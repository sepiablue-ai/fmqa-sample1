#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qa_bbo_krr_amplify_full.py
Fixstars Amplify を用いた QA-BBO 最小実装（フル版）

- カーネルリッジ回帰（多項式2次）をサロゲートとし、μ(x) を二次展開して QUBO 化
- Fixstars Amplify の Solver を使って QUBO を解く（上位 K 候補を取得）
- 取得候補は KRR の予測分散 s(x) を計算して UCB（μ - sqrt(beta)*s）で再ランク付け
- 上位をブラックボックスで評価して学習データに追加する BBO ループ

注意：
Amplify の実行には amplify-python-sdk のセットアップが必要です。
Amplify のクライアント設定（APIキー等）はユーザ側で事前に行ってください。
Amplify が利用できない場合はフォールバック実装で擬似的に動きます。
"""

import numpy as np
import math, random
from collections import defaultdict

# -------------------- 設定パラメータ --------------------
N_ITEMS = 64
MAX_POS = N_ITEMS
INIT_RANDOM = 100
TOP_K = 16
LAMBDA = 1e-3
POLY_COEF0 = 1.0
BETA0 = 1.0
W_REM = 1.0
W_COM = 0.3
W_ROW = 200.0
W_COL = 200.0

# Amplify 利用フラグ（環境に合わせて True/False）
USE_AMPLIFY = True

# -------------------- ヘルパー関数 --------------------
def idx(i, p):
    return i * MAX_POS + p

def order_to_onehot(order):
    x = np.zeros(N_ITEMS * MAX_POS, dtype=int)
    for p, i in enumerate(order):
        x[idx(i, p)] = 1
    return x

def onehot_to_order(x):
    order = [-1] * MAX_POS
    used = set()
    for p in range(MAX_POS):
        for i in range(N_ITEMS):
            if x[idx(i,p)] >= 1:
                if i in used:
                    continue
                order[p] = i
                used.add(i)
                break
    remaining = [i for i in range(N_ITEMS) if i not in used]
    cur = 0
    for p in range(MAX_POS):
        if order[p] == -1:
            order[p] = remaining[cur]; cur += 1
    return order

# -------------------- ブラックボックス（置き換えてください） --------------------
def blackbox_simulator(order, rotations=None):
    # ユーザの実環境シミュレータに置き換えてください。
    sizes = np.array([(i % 10) + 1 for i in range(N_ITEMS)])
    bin_capacity = int(N_ITEMS * 5.5)
    used = 0; packed = []
    for i in order:
        v = sizes[i]
        if used + v <= bin_capacity:
            used += v; packed.append(i)
        else:
            continue
    remaining = N_ITEMS - len(packed)
    centroid_norm = max(0.0, 1.0 - (used / float(bin_capacity)))
    return remaining, centroid_norm

# -------------------- 目的関数変換 --------------------
def compute_target(remaining, centroid_norm, q_mean):
    r_norm = remaining / float(N_ITEMS)
    Q = W_REM * r_norm + W_COM * centroid_norm
    y = - math.exp(- Q / q_mean)
    return y, Q

# -------------------- 多項式カーネル & KRR --------------------
def poly2_kernel(x, y, c0=POLY_COEF0):
    s = int(np.dot(x, y))
    return (c0 + s) ** 2

def build_kernel_matrix(X):
    n = X.shape[0]
    K = np.empty((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            K[i,j] = poly2_kernel(X[i], X[j])
            K[j,i] = K[i,j]
    return K

def fit_krr(X, y, lam=LAMBDA):
    X = np.array(X)
    y = np.array(y)
    K = build_kernel_matrix(X)
    alpha = np.linalg.solve(K + lam * np.eye(K.shape[0]), y)
    return {'X': X, 'y': y, 'K': K, 'alpha': alpha, 'lam': lam}

def predict_krr_expanded_qubo_terms(model):
    X = model['X']; alpha = model['alpha']; c0 = POLY_COEF0
    n_train, d = X.shape
    const = float(np.sum(alpha * (c0 ** 2)))
    h = np.zeros(d, dtype=float)
    for a in range(d):
        h[a] = 2.0 * c0 * np.dot(alpha, X[:, a])
    J = defaultdict(float)
    for j in range(n_train):
        xj = X[j]; coeff = alpha[j]
        idxs = np.nonzero(xj)[0]
        for ii in idxs:
            for jj in idxs:
                a = int(ii); b = int(jj)
                key = (a,b) if a<=b else (b,a)
                J[key] += coeff
    return const, h, J

def mu_predict_from_terms(const, h, J, x):
    val = const + np.dot(h, x)
    for (a,b), v in J.items():
        if a == b:
            val += v * x[a]
        else:
            val += v * x[a] * x[b]
    return val

def compute_predictive_std(model, x):
    X = model['X']; K = model['K']; lam = model['lam']
    n = X.shape[0]
    k_x = np.array([poly2_kernel(x, X[j]) for j in range(n)])
    Kreg = K + lam * np.eye(n)
    v = np.linalg.solve(Kreg, k_x)
    k_xx = poly2_kernel(x, x)
    s2 = max(0.0, k_xx - float(np.dot(k_x, v)))
    return math.sqrt(s2)

# -------------------- QUBO 組立 --------------------
def build_qubo_from_mu_terms(const, h_mu, J_mu, w_row=W_ROW, w_col=W_COL):
    d = len(h_mu)
    h = h_mu.copy().astype(float)
    J = dict(J_mu)
    for i in range(N_ITEMS):
        positions = [idx(i,p) for p in range(MAX_POS)]
        for p in positions:
            h[p] += w_row * (-1.0)
        for a_i in range(len(positions)):
            for b_i in range(a_i+1, len(positions)):
                a = positions[a_i]; b = positions[b_i]
                key = (a,b) if a<=b else (b,a)
                J[key] = J.get(key, 0.0) + w_row * 2.0
    for p in range(MAX_POS):
        items = [idx(i,p) for i in range(N_ITEMS)]
        for i_var in items:
            h[i_var] += w_col * (-1.0)
        for a_i in range(len(items)):
            for b_i in range(a_i+1, len(items)):
                a = items[a_i]; b = items[b_i]
                key = (a,b) if a<=b else (b,a)
                J[key] = J.get(key, 0.0) + w_col * 2.0
    const_pen = (w_row + w_col) * 1.0
    return h, J, const + const_pen

# -------------------- Amplify ソルバ呼び出し --------------------
def solve_qubo_amplify(h, J, top_k=TOP_K):
    try:
        from amplify import Solver, gen_symbols
    except Exception as e:
        print("[Info] amplify import failed:", e)
        return None
    d = len(h)
    vars = gen_symbols('b', d, type='Binary')
    obj = 0
    for i in range(d):
        obj += float(h[i]) * vars[i]
    for (a,b), v in J.items():
        obj += float(v) * vars[a] * vars[b]
    solver = Solver(client=None)
    result = solver.solve(obj)
    sols = []
    seen = set()
    for r in result:
        x = np.array([int(r[vars[i]]) for i in range(d)], dtype=int)
        key = tuple(x.tolist())
        if key in seen:
            continue
        seen.add(key)
        energy = float(np.dot(h, x))
        for (a,b), v in J.items():
            energy += float(v) * x[a] * x[b]
        sols.append({'x': x, 'energy': energy})
        if len(sols) >= top_k:
            break
    return sols

# -------------------- フォールバック：グリーディサンプラー --------------------
def fallback_sampler(h, J, top_k=TOP_K):
    print("[Info] Using fallback greedy feasible sampler.")
    sols = []
    for _ in range(max(top_k,200)):
        perm = list(range(N_ITEMS)); random.shuffle(perm)
        x = order_to_onehot(perm)
        sols.append({'x': x, 'energy': 0.0})
        if len(sols) >= top_k:
            break
    return sols

# -------------------- メイン BBO ループ --------------------
def main_bbo_loop(max_epochs=50, init_random=INIT_RANDOM):
    X_train = []; Q_raws = []
    print("[Info] Generating initial dataset...")
    for i in range(init_random):
        if i < init_random // 2:
            base_order = sorted(range(N_ITEMS), key=lambda k: -((k % 10) + 1))
            for _ in range(3):
                a = random.randrange(N_ITEMS); b = random.randrange(N_ITEMS)
                base_order[a], base_order[b] = base_order[b], base_order[a]
            order = base_order
        else:
            order = list(range(N_ITEMS)); random.shuffle(order)
        x = order_to_onehot(order)
        remaining, centroid_norm = blackbox_simulator(order)
        Q_raw = W_REM * (remaining/float(N_ITEMS)) + W_COM * centroid_norm
        X_train.append(x); Q_raws.append(Q_raw)
    X_train = np.array(X_train, dtype=int)
    Q_mean = float(np.mean(Q_raws))
    y_train = []
    for x in X_train:
        order = onehot_to_order(x)
        remaining, centroid_norm = blackbox_simulator(order)
        y, Qv = compute_target(remaining, centroid_norm, q_mean=Q_mean)
        y_train.append(y)
    y_train = np.array(y_train, dtype=float)

    model = fit_krr(X_train, y_train, lam=LAMBDA)

    best_real = None; best_order = None
    for epoch in range(1, max_epochs+1):
        print(f"[Epoch {epoch}] Fit KRR on {model['X'].shape[0]} samples...")
        const, h_mu, J_mu = predict_krr_expanded_qubo_terms(model)
        h, J, const_all = build_qubo_from_mu_terms(const, h_mu, J_mu, w_row=W_ROW, w_col=W_COL)
        sols = None
        if USE_AMPLIFY:
            try:
                sols = solve_qubo_amplify(h, J, top_k=TOP_K)
                if sols is None:
                    sols = fallback_sampler(h, J, top_k=TOP_K)
            except Exception as e:
                print("[Warn] Amplify solve failed:", e)
                sols = fallback_sampler(h, J, top_k=TOP_K)
        else:
            sols = fallback_sampler(h, J, top_k=TOP_K)
        print(f"[Epoch {epoch}] Obtained {len(sols)} candidates from QA.")
        cand_list = []
        for s in sols:
            x_cand = s['x']
            order_cand = onehot_to_order(x_cand)
            mu_est = mu_predict_from_terms(const, h_mu, J_mu, x_cand)
            s_val = compute_predictive_std(model, x_cand)
            beta_t = BETA0 * math.log(1 + epoch)
            ucb = mu_est - math.sqrt(beta_t) * s_val
            cand_list.append({'x': x_cand, 'order': order_cand, 'mu': mu_est, 's': s_val, 'ucb': ucb})
        cand_list = sorted(cand_list, key=lambda z: z['ucb'])
        num_eval = min(4, len(cand_list))
        for i_eval in range(num_eval):
            cand = cand_list[i_eval]
            remaining, centroid_norm = blackbox_simulator(cand['order'])
            y_real, Q_real = compute_target(remaining, centroid_norm, q_mean=Q_mean)
            print(f"  Eval {i_eval}: remaining={remaining}, centroid={centroid_norm:.3f}, Q={Q_real:.6f}, mu={cand['mu']:.6f}, s={cand['s']:.6f}, ucb={cand['ucb']:.6f}")
            xvec = cand['x']
            model['X'] = np.vstack([model['X'], xvec])
            model['y'] = np.hstack([model['y'], y_real])
            model['K'] = build_kernel_matrix(model['X'])
            model = fit_krr(model['X'], model['y'], lam=LAMBDA)
            if (best_real is None) or (Q_real < best_real):
                best_real = Q_real; best_order = cand['order']
                print("  New best real Q:", best_real)
    print("Done. Best real Q:", best_real)
    if best_order is not None:
        print("Best order (first 20):", best_order[:20])
    return best_real, best_order

# -------------------- 実行 --------------------
if __name__ == '__main__':
    main_bbo_loop(max_epochs=30)