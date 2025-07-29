import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from collections import deque

# === コンテナとアイテムの定義 ===
container_size = (10, 10, 10)  # 大きな直方体サイズ（幅, 奥行, 高さ）

# 小さい直方体のリスト（幅, 奥行, 高さ）
item_list = [
    (2, 2, 2),
    (3, 2, 1),
    (1, 1, 3),
    (2, 3, 2),
    (1, 1, 1),
    (3, 3, 3),
    (2, 2, 1),
    (1, 2, 2)
]

item_queue = deque(item_list)

# === 積み込み位置の管理用 ===
placed_items = []  # [(x, y, z, w, d, h)]

# === 位置が衝突していないかを判定 ===
def does_overlap(pos1, size1, pos2, size2):
    x1, y1, z1 = pos1
    w1, d1, h1 = size1
    x2, y2, z2 = pos2
    w2, d2, h2 = size2
    return not (
        x1 + w1 <= x2 or x2 + w2 <= x1 or
        y1 + d1 <= y2 or y2 + d2 <= y1 or
        z1 + h1 <= z2 or z2 + h2 <= z1
    )

# === 配置可能かチェック ===
def can_place(x, y, z, size):
    w, d, h = size
    if x + w > container_size[0] or y + d > container_size[1] or z + h > container_size[2]:
        return False
    for item in placed_items:
        if does_overlap((x, y, z), size, (item[0], item[1], item[2]), (item[3], item[4], item[5])):
            return False
    return True

# === BLD法（Bottom-Left-Down法）による積み込み ===
def place_items():
    while item_queue:
        item = item_queue.popleft()
        placed = False
        # 3次元空間内で可能な限り左・手前・下に詰めていく
        for z in range(container_size[2]):
            for y in range(container_size[1]):
                for x in range(container_size[0]):
                    if can_place(x, y, z, item):
                        placed_items.append((x, y, z, *item))
                        placed = True
                        break
                if placed:
                    break
            if placed:
                break

# === 重心計算 ===
def evaluate_center_of_gravity():
    total_volume = 0
    weighted_sum = np.array([0.0, 0.0, 0.0])
    for x, y, z, w, d, h in placed_items:
        volume = w * d * h
        center = np.array([x + w / 2, y + d / 2, z + h / 2])
        weighted_sum += center * volume
        total_volume += volume
    cog = weighted_sum / total_volume
    container_center = np.array(container_size) / 2
    deviation = np.linalg.norm(cog - container_center)
    return cog, container_center, deviation

# === 3D 可視化 ===
def visualize():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # コンテナの枠線
    ax.plot([0, container_size[0]], [0, 0], [0, 0], color='gray')
    ax.plot([0, container_size[0]], [container_size[1], container_size[1]], [0, 0], color='gray')
    ax.plot([0, 0], [0, container_size[1]], [0, 0], color='gray')

    for x, y, z, w, d, h in placed_items:
        draw_box(ax, x, y, z, w, d, h)

    ax.set_xlim([0, container_size[0]])
    ax.set_ylim([0, container_size[1]])
    ax.set_zlim([0, container_size[2]])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("3Dパッキング結果")
    plt.show()

# 直方体を描画
def draw_box(ax, x, y, z, w, d, h):
    # 頂点座標を定義
    corners = np.array([
        [x, y, z],
        [x + w, y, z],
        [x + w, y + d, z],
        [x, y + d, z],
        [x, y, z + h],
        [x + w, y, z + h],
        [x + w, y + d, z + h],
        [x, y + d, z + h]
    ])
    # 面を構成する頂点の組み合わせ
    faces = [
        [corners[j] for j in [0, 1, 2, 3]],
        [corners[j] for j in [4, 5, 6, 7]],
        [corners[j] for j in [0, 1, 5, 4]],
        [corners[j] for j in [2, 3, 7, 6]],
        [corners[j] for j in [1, 2, 6, 5]],
        [corners[j] for j in [4, 7, 3, 0]]
    ]
    box = Poly3DCollection(faces, alpha=0.6)
    box.set_edgecolor('k')
    ax.add_collection3d(box)

# === メイン処理 ===
if __name__ == "__main__":
    place_items()
    cog, center, deviation = evaluate_center_of_gravity()
    print(f"重心位置: {cog}")
    print(f"コンテナ中心: {center}")
    print(f"重心の中心からのずれ: {deviation:.2f}")
    visualize()