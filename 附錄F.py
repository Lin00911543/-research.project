import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# 模擬 AI 系統自動紀錄的動作數據
# 欄位: 學生ID, 測試階段(前測/後測), 肩角誤差(°), 錯誤次數
data = [
    ["S001", "Pre", 15, 8],
    ["S001", "Post", 7, 3],
    ["S002", "Pre", 18, 10],
    ["S002", "Post", 9, 4],
    ["S003", "Pre", 20, 11],
    ["S003", "Post", 8, 5]
]
df = pd.DataFrame(data, columns=["StudentID", "Stage", "ShoulderAngleError", "ErrorCount"])

# 前後測資料分離
pre = df[df["Stage"] == "Pre"]["ShoulderAngleError"]
post = df[df["Stage"] == "Post"]["ShoulderAngleError"]

# 成對樣本 t 檢定
t_stat, p_value = stats.ttest_rel(pre, post)
print(f"T檢定結果: t = {t_stat:.3f}, p = {p_value:.4f}")
if p_value < 0.05:
    print("結論: 差異具有統計顯著性，系統有效改善學習成效")
else:
    print("結論: 差異不顯著，需進一步驗證")

# 視覺化誤差改善情況
avg_data = df.groupby("Stage")["ShoulderAngleError"].mean()
avg_data.plot(kind="bar", color=["red", "green"])
plt.title("肩角誤差前後測比較")
plt.ylabel("平均誤差(°)")
plt.xticks(rotation=0)
plt.show()

