import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from collections import deque

# === コンテナとアイテムの定義 ===
container_size = (10, 10, 10)  # 幅, 奥行, 高さ

# 積み込む小箱（幅, 奥行, 高さ）のキュー
item_list = [
    (2, 2, 2),
    (3, 2, 1),
    (1, 1, 3),
    (2, 3, 2),
    (1, 1, 1),
    (3, 3, 3),
    (2, 2, 1),
    (1, 2, 2),
]
item_queue = deque(item_list)

# 配置された箱 [(x, y, z, w, d, h)]
placed_items = []

# === 衝突判定 ===
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

# === 支持面積が指定以上あるかを判定 ===
def has_support(x, y, z, w, d, support_ratio):
    if z == 0:
        return True  # 地面は常に支持されているとみなす
    support_area = 0
    for item in placed_items:
        ix, iy, iz, iw, id, ih = item
        if iz + ih == z:  # 高さがぴったり一致している箱
            overlap_x = max(0, min(x + w, ix + iw) - max(x, ix))
            overlap_y = max(0, min(y + d, iy + id) - max(y, iy))
            support_area += overlap_x * overlap_y
    return support_area >= support_ratio * w * d

# === 指定位置に置けるか ===
def can_place(x, y, z, size, support_ratio):
    w, d, h = size
    if x + w > container_size[0] or y + d > container_size[1] or z + h > container_size[2]:
        return False
    if not has_support(x, y, z, w, d, support_ratio):
        return False
    for item in placed_items:
        if does_overlap((x, y, z), size, (item[0], item[1], item[2]), (item[3], item[4], item[5])):
            return False
    return True

# === 指定XY位置に置ける最も低いZを探索 ===
def find_lowest_valid_z(x, y, item_size, support_ratio):
    w, d, h = item_size
    for z in range(container_size[2] - h + 1):  # 下から上へ
        if can_place(x, y, z, item_size, support_ratio):
            return z
    return None

# === 重力落下型BLD法による積み込み ===
def place_items_gravity(support_ratio=1.0):
    while item_queue:
        item = item_queue.popleft()
        w, d, h = item
        best_pos = None

        for y in range(container_size[1] - d + 1):
            for x in range(container_size[0] - w + 1):
                z = find_lowest_valid_z(x, y, item, support_ratio)
                if z is not None:
                    best_pos = (x, y, z)
                    break  # 最初に見つけた位置に配置（BLD）
            if best_pos:
                break

        if best_pos:
            x, y, z = best_pos
            placed_items.append((x, y, z, w, d, h))
        else:
            print(f"積み込みできなかったアイテム: {item}")

# === 重心計算 ===
def evaluate_center_of_gravity():
    total_volume = 0
    weighted_sum = np.array([0.0, 0.0, 0.0])
    for x, y, z, w, d, h in placed_items:
        volume = w * d * h
        center = np.array([x + w/2, y + d/2, z + h/2])
        weighted_sum += center * volume
        total_volume += volume
    cog = weighted_sum / total_volume
    container_center = np.array(container_size) / 2
    deviation = np.linalg.norm(cog - container_center)
    return cog, container_center, deviation

# === 3D可視化 ===
def visualize():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for x, y, z, w, d, h in placed_items:
        draw_box(ax, x, y, z, w, d, h)
    ax.set_xlim([0, container_size[0]])
    ax.set_ylim([0, container_size[1]])
    ax.set_zlim([0, container_size[2]])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("重力型3Dパッキング")
    plt.show()

# === 箱を描画 ===
def draw_box(ax, x, y, z, w, d, h):
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
    faces = [
        [corners[i] for i in [0, 1, 2, 3]],
        [corners[i] for i in [4, 5, 6, 7]],
        [corners[i] for i in [0, 1, 5, 4]],
        [corners[i] for i in [2, 3, 7, 6]],
        [corners[i] for i in [1, 2, 6, 5]],
        [corners[i] for i in [4, 7, 3, 0]],
    ]
    box = Poly3DCollection(faces, alpha=0.6, edgecolor='k')
    ax.add_collection3d(box)

# === メイン実行 ===
if __name__ == "__main__":
    # ここで支持率を変更できる（例: 0.9 → 90%支持）
    support_threshold = 0.9
    place_items_gravity(support_ratio=support_threshold)
    cog, center, deviation = evaluate_center_of_gravity()
    print(f"\n重心位置: {cog}")
    print(f"コンテナ中心: {center}")
    print(f"重心のずれ: {deviation:.2f}")
    visualize()