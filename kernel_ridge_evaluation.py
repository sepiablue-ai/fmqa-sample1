import numpy as np
from scipy.stats import spearmanr

# =====================
# スコア計算（重心+指数変換）
# =====================
def compute_Q(remaining, utilization, centroid_dist,
              M=63, d_max=1.0,
              w_r=100.0, w_u=10.0, w_c=5.0):
    r_norm = remaining / float(M)
    u = 1.0 - utilization
    d_norm = min(1.0, centroid_dist / float(d_max))
    return w_r * r_norm + w_u * u + w_c * d_norm

def compute_transformed_score(Q, c_m):
    return -np.exp(-Q / c_m)

# =====================
# ダミー blackbox
# =====================
def blackbox_dummy(order_bin):
    remaining = np.random.randint(0, 5)  
    utilization = np.random.uniform(0.7, 0.99)
    centroid_dist = np.random.uniform(0.0, 0.5)
    pack_result = 0 if remaining == 0 else 1
    return pack_result, remaining, utilization, centroid_dist

# =====================
# カーネル計算（poly degree=2）
# =====================
def poly_kernel(X1, X2, gamma=1.0, coef0=0.0, degree=2):
    return (gamma * np.dot(X1, X2.T) + coef0) ** degree

# =====================
# 論文式カーネルQA学習
# =====================
def train_kernel_qa(X_train, y_train, lambda_reg=1e-6, gamma=1.0, coef0=0.0, degree=2):
    K = poly_kernel(X_train, X_train, gamma=gamma, coef0=coef0, degree=degree)
    N = K.shape[0]
    alpha = np.linalg.solve(K + lambda_reg * np.eye(N), y_train)
    return alpha, X_train

# =====================
# 予測
# =====================
def predict_kernel_qa(alpha, X_train, X_test, gamma=1.0, coef0=0.0, degree=2):
    K_test = poly_kernel(X_test, X_train, gamma=gamma, coef0=coef0, degree=degree)
    return np.dot(K_test, alpha)

# =====================
# サロゲート性能評価
# =====================
def evaluate_surrogate(y_true_trans, y_pred_trans, feas_flags_true):
    spearman_corr, _ = spearmanr(y_true_trans, y_pred_trans)

    threshold = np.median(y_pred_trans)
    feas_pred_flags = (y_pred_trans < threshold).astype(int)
    tp = np.sum((feas_pred_flags == 1) & (feas_flags_true == 1))
    fn = np.sum((feas_pred_flags == 0) & (feas_flags_true == 1))
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "spearman_corr": spearman_corr,
        "recall_feasible": recall
    }

# =====================
# 実行例
# =====================
if __name__ == "__main__":
    np.random.seed(0)
    n_features = 63*63 + 63
    n_train = 50
    n_test = 20

    # 初期データ生成
    X_train = np.random.randint(0, 2, size=(n_train, n_features))
    Q_list, feas_flags_train = [], []
    for x in X_train:
        pack_result, rem, util, cdist = blackbox_dummy(x)
        Q_list.append(compute_Q(rem, util, cdist))
        feas_flags_train.append(1 if pack_result == 0 else 0)
    Q_list = np.array(Q_list)
    feas_flags_train = np.array(feas_flags_train)
    Q_init_mean = np.mean(Q_list)
    y_train = compute_transformed_score(Q_list, Q_init_mean)

    # 学習
    alpha, X_train_ref = train_kernel_qa(X_train, y_train, lambda_reg=1e-6, gamma=1.0, coef0=0.0, degree=2)

    # テストデータ
    X_test = np.random.randint(0, 2, size=(n_test, n_features))
    Q_test_list, feas_flags_test = [], []
    for x in X_test:
        pack_result, rem, util, cdist = blackbox_dummy(x)
        Q_test_list.append(compute_Q(rem, util, cdist))
        feas_flags_test.append(1 if pack_result == 0 else 0)
    Q_test_list = np.array(Q_test_list)
    y_test = compute_transformed_score(Q_test_list, Q_init_mean)
    feas_flags_test = np.array(feas_flags_test)

    # 予測
    y_pred = predict_kernel_qa(alpha, X_train_ref, X_test, gamma=1.0, coef0=0.0, degree=2)

    # 評価
    metrics = evaluate_surrogate(y_test, y_pred, feas_flags_test)
    print("Q_init_mean:", Q_init_mean)
    print("Spearman順位相関:", metrics["spearman_corr"])
    print("可行性Recall:", metrics["recall_feasible"])