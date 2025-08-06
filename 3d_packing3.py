import random

# ===========================
# データ構造
# ===========================
class Box:
    def __init__(self, w, d, h, id):
        self.w = w
        self.d = d
        self.h = h
        self.id = id

class PlacedBox(Box):
    def __init__(self, x, y, z, w, d, h, id):
        super().__init__(w, d, h, id)
        self.x = x
        self.y = y
        self.z = z

    def position(self):
        return (self.x, self.y, self.z)

# ===========================
# 判定関数
# ===========================
def does_overlap(pb1, pb2):
    return (pb1.x < pb2.x + pb2.w and pb1.x + pb1.w > pb2.x and
            pb1.y < pb2.y + pb2.d and pb1.y + pb1.d > pb2.y and
            pb1.z < pb2.z + pb2.h and pb1.z + pb1.h > pb2.z)

def is_inside_container(box, container_size):
    x, y, z = box.x, box.y, box.z
    W, D, H = container_size
    return (x + box.w <= W and y + box.d <= D and z + box.h <= H)

def has_support(box, placed_boxes, min_support_ratio=0.5):
    if box.z == 0:
        return True  # 床に接していればOK

    support_area = 0
    required_area = box.w * box.d * min_support_ratio

    for other in placed_boxes:
        if (other.z + other.h == box.z):  # すぐ下にある
            x_overlap = max(0, min(box.x + box.w, other.x + other.w) - max(box.x, other.x))
            y_overlap = max(0, min(box.y + box.d, other.y + other.d) - max(box.y, other.y))
            support_area += x_overlap * y_overlap
            if support_area >= required_area:
                return True

    return False

# ===========================
# 重力落下処理
# ===========================
def apply_gravity(candidate, placed_boxes, container_height, min_support_ratio):
    # 上から下へ z を 1 ずつ下げながら試す
    for new_z in range(candidate.z, -1, -1):
        test_box = PlacedBox(candidate.x, candidate.y, new_z, candidate.w, candidate.d, candidate.h, candidate.id)
        if is_inside_container(test_box, (float('inf'), float('inf'), container_height)) and \
           not any(does_overlap(test_box, pb) for pb in placed_boxes) and \
           has_support(test_box, placed_boxes, min_support_ratio):
            return test_box
    return None  # 落下できない場合（ありえないが）

# ===========================
# メイン配置アルゴリズム
# ===========================
def pack_boxes(boxes, container_size, min_support_ratio=0.5):
    placed_boxes = []
    extreme_points = [(0, 0, 0)]  # 初期候補点

    for box in boxes:
        best_candidate = None
        best_z = float('inf')

        for ep in extreme_points:
            initial_candidate = PlacedBox(ep[0], ep[1], ep[2], box.w, box.d, box.h, box.id)
            if not is_inside_container(initial_candidate, container_size):
                continue
            if any(does_overlap(initial_candidate, pb) for pb in placed_boxes):
                continue

            # 重力に従って下に落下させる
            dropped_box = apply_gravity(initial_candidate, placed_boxes, container_size[2], min_support_ratio)
            if dropped_box and dropped_box.z < best_z:
                best_candidate = dropped_box
                best_z = dropped_box.z

        if best_candidate:
            placed_boxes.append(best_candidate)
            # 新たな候補点（右・奥・上）
            extreme_points.append((best_candidate.x + best_candidate.w, best_candidate.y, best_candidate.z))
            extreme_points.append((best_candidate.x, best_candidate.y + best_candidate.d, best_candidate.z))
            extreme_points.append((best_candidate.x, best_candidate.y, best_candidate.z + best_candidate.h))
        else:
            print(f"箱 {box.id} は配置できませんでした")

    return placed_boxes

# ===========================
# 実行
# ===========================
def main():
    container = (2230, 610, 670)

    # テスト用：ランダムな64個の箱（サイズは現実的に調整）
    boxes = []
    for i in range(64):
        w = random.randint(100, 400)
        d = random.randint(100, 300)
        h = random.randint(100, 300)
        boxes.append(Box(w, d, h, i+1))

    # パッキング実行
    result = pack_boxes(boxes, container, min_support_ratio=0.5)

    # 結果出力
    print("\n配置結果:")
    for b in result:
        print(f"箱 {b.id}: pos=({b.x},{b.y},{b.z}), size=({b.w},{b.d},{b.h})")

    print(f"\n合計 {len(result)} 箱を配置できました。")

if __name__ == "__main__":
    main()