import numpy as np
import matplotlib.pyplot as plt

# ---- カーネル関数（例: RBFカーネル）----
def rbf_kernel(X1, X2, gamma=1.0):
    """
    X1: (n_samples1, n_features)
    X2: (n_samples2, n_features)
    gamma: ガウスカーネルのパラメータ
    """
    dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * np.dot(X1, X2.T)
    return np.exp(-gamma * dists)

# ---- カーネルQA（カーネルリッジ回帰）学習 ----
def kernel_qa_train(X, y, gamma=1.0, lam=1e-6):
    """
    X: (n_samples, n_features) 入力ベクトル（順序のワンホットなど）
    y: (n_samples,) 正解スコア
    gamma: RBFカーネルパラメータ
    lam: 正則化パラメータ
    """
    K = rbf_kernel(X, X, gamma=gamma)
    alpha = np.linalg.solve(K + lam * np.eye(K.shape[0]), y)
    return alpha, X

# ---- 予測 ----
def kernel_qa_predict(X_train, alpha, X_test, gamma=1.0):
    K_test = rbf_kernel(X_test, X_train, gamma=gamma)
    return np.dot(K_test, alpha)

# ---- 精度評価＆プロット ----
def evaluate_kernel_qa(order_list, score_list, gamma=1.0, lam=1e-6, title="KernelQA Evaluation"):
    """
    order_list: (n_samples, n_features) 例: 順序のワンホット＋回転変数
    score_list: (n_samples,) ブラックボックスのスコア（正解値）
    """
    X = np.array(order_list, dtype=float)
    y = np.array(score_list, dtype=float)

    # 学習
    alpha, X_train = kernel_qa_train(X, y, gamma=gamma, lam=lam)

    # 予測（ここでは学習データそのまま予測＝再現性評価）
    y_pred = kernel_qa_predict(X_train, alpha, X_train, gamma=gamma)

    # 散布図プロット
    plt.figure(figsize=(6, 6))
    plt.scatter(y, y_pred, alpha=0.7, edgecolors='k')
    plt.plot([min(y), max(y)], [min(y), max(y)], 'r--', label="y = x")
    plt.xlabel("True Score")
    plt.ylabel("Predicted Score")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

    # 相関係数など出力
    corr = np.corrcoef(y, y_pred)[0, 1]
    mse = np.mean((y - y_pred) ** 2)
    print(f"Correlation: {corr:.4f}")
    print(f"MSE: {mse:.6f}")

# ==== サンプル実行例 ====
if __name__ == "__main__":
    np.random.seed(0)
    n_samples = 30
    n_features = 10  # 順序のエンコード後の特徴次元

    # 仮の順序ベクトルとスコアを作成
    X_demo = np.random.randint(0, 2, size=(n_samples, n_features))  # バイナリ順序表現
    y_demo = np.sin(np.sum(X_demo, axis=1)) + 0.1 * np.random.randn(n_samples)  # 仮スコア

    evaluate_kernel_qa(X_demo, y_demo, gamma=0.5, lam=1e-6, title="KernelQA Predict vs True")