# qa_bbo_krr_amplify.py
# QA BBO (Kernel Ridge Regression UCB版) 最小実装サンプル
# Fixstars Amplify を利用可能なら QUBOをQAで解き、なければ簡易サンプラーで代用します。

import numpy as np
import itertools
import logging

try:
    from amplify import Solver, BinarySymbolGenerator, decode_solution, FixstarsClient, Model
    USE_AMPLIFY = True
except ImportError:
    USE_AMPLIFY = False
    logging.warning("Amplify が見つからないため、ローカル簡易サンプラーで代用します。")

# ======================================================
# ブラックボックス関数 (ダミー実装)
# 実際にはあなたのローカルの積み込みシミュレータに置き換えてください。
# 引数: 順序 (list of int), 回転有無 (list of int)
# 戻り値: スコア (小さいほど良い)
# ======================================================
def blackbox_simulator(order, rotations):
    # ダミー: 積み残し個数 + 重心ズレ(乱数) を返す
    remaining = max(0, 64 - len(order))  # 実際にはシミュレータに依存
    centroid_deviation = np.random.rand()  # 0〜1の乱数
    score = remaining * 100 + centroid_deviation * 10
    return score

# ======================================================
# カーネルリッジ回帰 (多項式カーネル)
# ======================================================
class KernelRidge:
    def __init__(self, lam=1e-6, degree=2):
        self.lam = lam
        self.degree = degree
        self.X = None
        self.y = None
        self.alpha = None
        self.K = None

    def kernel(self, X1, X2):
        # 多項式カーネル
        return (1 + np.dot(X1, X2.T)) ** self.degree

    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        K = self.kernel(self.X, self.X)
        n = K.shape[0]
        self.alpha = np.linalg.solve(K + self.lam * np.eye(n), self.y)
        self.K = K

    def predict(self, X_new):
        Kx = self.kernel(np.array(X_new), self.X)
        mu = np.dot(Kx, self.alpha)
        return mu

    def predict_with_uncertainty(self, X_new):
        X_new = np.array(X_new)
        Kx = self.kernel(X_new, self.X)
        mu = np.dot(Kx, self.alpha)
        K_inv = np.linalg.inv(self.K + self.lam * np.eye(self.K.shape[0]))
        kxx = self.kernel(X_new, X_new)
        s2 = np.diag(kxx - np.dot(Kx, np.dot(K_inv, Kx.T)))
        s2 = np.maximum(s2, 1e-8)
        return mu, np.sqrt(s2)

# ======================================================
# 順序のワンホットエンコード
# ======================================================
def order_to_onehot(order, n_items=64):
    mat = np.zeros((n_items, n_items))
    for pos, item in enumerate(order):
        mat[pos, item] = 1
    return mat.flatten()

# ======================================================
# QUBO生成 (サロゲート予測を線形/二次項に分解)
# ======================================================
def build_qubo_from_weights(weights, n_items=64):
    # 簡易的に「重みをそのまま線形項」として構成
    # 実際には順序制約のため二次項制約も必要
    linear = {}
    quadratic = {}
    for i, w in enumerate(weights):
        linear[i] = w
    return linear, quadratic

# ======================================================
# QA BBO本体
# ======================================================
def qa_bbo(n_epochs=50, init_data=100, beta=2.0):
    # 初期データ生成
    X_data, y_data = [], []
    for _ in range(init_data):
        order = list(np.random.permutation(64))
        onehot = order_to_onehot(order)
        rotations = [0] * 64
        y = blackbox_simulator(order, rotations)
        X_data.append(onehot)
        y_data.append(y)

    # サロゲート初期学習
    model = KernelRidge()
    model.fit(X_data, y_data)

    best_score = min(y_data)
    print(f"初期ベストスコア: {best_score}")

    for epoch in range(n_epochs):
        # 乱数で候補生成
        order = list(np.random.permutation(64))
        onehot = order_to_onehot(order)

        mu, s = model.predict_with_uncertainty([onehot])
        acq = mu[0] - np.sqrt(beta) * s[0]  # UCB風

        # Amplify利用部分 (簡略化)
        if USE_AMPLIFY:
            gen = BinarySymbolGenerator()
            q = gen.array(shape=64*64)
            linear, quadratic = build_qubo_from_weights(onehot)
            # QUBOをAmplifyに送る（ここは簡易化して未実装にしています）
            # 実際には amplify.Model に linear/quadratic を追加して solver.solve(model)
            candidate = order
        else:
            candidate = order

        # ブラックボックス評価
        rotations = [0] * 64
        y_new = blackbox_simulator(candidate, rotations)
        if y_new < best_score:
            best_score = y_new
            print(f"[{epoch}] 新ベストスコア: {best_score}")

        # データ追加＆再学習
        X_data.append(order_to_onehot(candidate))
        y_data.append(y_new)
        model.fit(X_data, y_data)

    return best_score

# ======================================================
# 実行
# ======================================================
if __name__ == "__main__":
    best = qa_bbo()
    print("最終ベストスコア:", best)