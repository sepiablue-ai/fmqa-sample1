"""
Best-Fit Decreasing の流れ（割当フェーズ）
	1.	段ボールを体積（大きさ）順にソート
	•	降順に並べる（大きいものから順に割り当てる）
	2.	2箱の目標容量を計算
	•	箱Aの容量 / (箱A+箱Bの容量) → Aの目標比率
	•	箱Bも同様
	•	例：箱A=600、箱B=400 → Aの目標=60%、B=40%
	3.	段ボールを順に、目標との差が小さくなる方へ置く
	•	箱Aに入れた場合と箱Bに入れた場合で、それぞれ「割当体積 − 目標体積」の差を計算
	•	差が小さい方へ置く
"""
import random

def best_fit_decreasing(volumes, capA, capB, seed=None):
    """
    Best-Fit Decreasing に基づき、段ボールを2箱(A,B)に割り当てる
    volumes : list of float/int 段ボール体積のリスト
    capA, capB : float/int 輸送箱A,Bの容量
    """
    if seed is not None:
        random.seed(seed)
    
    # --- 1) 段ボールを大きい順にソート ---
    boxes = sorted(volumes, reverse=True)

    # --- 2) 目標体積 ---
    total_vol = sum(boxes)
    targetA = total_vol * (capA / (capA + capB))
    targetB = total_vol * (capB / (capA + capB))

    # --- 3) 割当処理 ---
    assignA, assignB = [], []
    volA, volB = 0, 0

    for v in boxes:
        # 仮にAに入れた場合の差
        diffA = abs((volA + v) - targetA) + abs(volB - targetB)
        # 仮にBに入れた場合の差
        diffB = abs(volA - targetA) + abs((volB + v) - targetB)

        if diffA <= diffB:
            # Aに入れる
            assignA.append(v)
            volA += v
        else:
            # Bに入れる
            assignB.append(v)
            volB += v

    # --- 4) 容量超過チェック ---
    feasible = (volA <= capA and volB <= capB)

    return {
        "assignA": assignA,
        "assignB": assignB,
        "volA": volA,
        "volB": volB,
        "targetA": targetA,
        "targetB": targetB,
        "feasible": feasible
    }

# ==== 使用例 ====
if __name__ == "__main__":
    # 段ボール体積サンプル（120個）
    volumes = [random.randint(5, 50) for _ in range(120)]
    capA, capB = 1500, 1000  # 輸送箱A,Bの容量

    result = best_fit_decreasing(volumes, capA, capB, seed=42)

    print("箱A 個数:", len(result["assignA"]), "容量:", result["volA"], "/", capA)
    print("箱B 個数:", len(result["assignB"]), "容量:", result["volB"], "/", capB)
    print("可行解か？:", result["feasible"])