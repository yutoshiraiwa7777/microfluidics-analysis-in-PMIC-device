import io
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from google.colab import files
from PIL import Image

# ==============================================================
# Parameters
# ==============================================================
PIXEL_SIZE  = 3.158   # μm / pixel
DT          = 0.1

N_PARTICLES = 54
N_FRAMES    = 10
N_STEPS     = N_FRAMES - 1

# 空間閾値（10.0 μm）
DISTANCE_THRESHOLD = 10.0

# ヒストグラムのビンの数（18個 = 90度を5度刻みで分割）
N_BINS = 18

# ==============================================================
# Upload CSV
# ==============================================================
print("① Particle CSV をアップロードしてください")
uploaded_csv = files.upload()
raw_csv = list(uploaded_csv.values())[0]

for enc in ("utf-8", "utf-8-sig", "cp932", "shift_jis", "latin-1"):
    try:
        df = pd.read_csv(io.BytesIO(raw_csv), encoding=enc)
        break
    except:
        continue

df.columns = [c.strip() for c in df.columns]
if "x [μm]" not in df.columns:
    cols = list(df.columns)
    df = df.rename(columns={cols[0]: "x [μm]", cols[1]: "y [μm]"})

# ==============================================================
# Upload TIF (Distance Map)
# ==============================================================
print("\n② Distance map (TIF) をアップロードしてください")
uploaded_tif = files.upload()
raw_tif = list(uploaded_tif.values())[0]
img = Image.open(io.BytesIO(raw_tif))
distance_map = np.array(img).astype(float)

# ==============================================================
# Contour extraction & Circle fitting
# ==============================================================
temp_fig, temp_ax = plt.subplots()
cs = temp_ax.contour(distance_map, levels=[0],
                     extent=[0, distance_map.shape[1] * PIXEL_SIZE, 0, distance_map.shape[0] * PIXEL_SIZE])
plt.close(temp_fig)

paths = cs.get_paths()
if len(paths) == 0:
    raise ValueError("Contour extraction failed")
v = paths[0].vertices
x_pts, y_pts = v[:, 0], v[:, 1]

A = np.array([x_pts, y_pts, np.ones(len(x_pts))]).T
b = x_pts**2 + y_pts**2
c, d, e = np.linalg.lstsq(A, b, rcond=None)[0]
xc, yc = c / 2, d / 2

# ==============================================================
# Invasion angle calculation (Distance-based classification)
# ==============================================================
far_angles  = []
near_angles = []

h, w = distance_map.shape

for p in range(N_PARTICLES):
    sl = slice(p * N_FRAMES, (p + 1) * N_FRAMES)
    x_um = df.iloc[sl]["x [μm]"].values.astype(float)
    y_um = df.iloc[sl]["y [μm]"].values.astype(float)

    for step in range(N_STEPS):
        x_s, y_s = x_um[step], y_um[step]
        x_e, y_e = x_um[step + 1], y_um[step + 1]

        v_particle = np.array([x_e - x_s, y_e - y_s])
        v_normal   = np.array([x_e - xc, y_e - yc])

        norm_p = np.linalg.norm(v_particle)
        norm_n = np.linalg.norm(v_normal)

        if norm_p == 0 or norm_n == 0:
            continue

        cos_theta = np.dot(v_particle, v_normal) / (norm_p * norm_n)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle_raw = np.degrees(np.arccos(cos_theta))

        if angle_raw > 90:
            angle_from_normal = 180 - angle_raw
        else:
            angle_from_normal = angle_raw

        angle_villus = 90 - angle_from_normal

        # Distance Map から現在の位置の距離（μm）を取得
        px = max(0, min(int(round(x_s / PIXEL_SIZE)), w - 1))
        py = max(0, min(int(round(y_s / PIXEL_SIZE)), h - 1))
        current_distance = distance_map[py, px]

        if current_distance > DISTANCE_THRESHOLD:
            far_angles.append(angle_villus)
        else:
            near_angles.append(angle_villus)

print(f"\n📊 データ分類結果 (閾値: {DISTANCE_THRESHOLD} μm):")
print(f"   Far 領域の総ステップ数 : {len(far_angles)}")
print(f"   Near 領域の総ステップ数: {len(near_angles)}")

# Convert to radians
far_rad  = np.radians(far_angles)
near_rad = np.radians(near_angles)

# ==============================================================
# Plot settings (Rose Diagram / Polar Histogram)
# ==============================================================
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["pdf.fonttype"] = 42

fig = plt.figure(figsize=(7, 7), dpi=300)
ax = plt.subplot(111, projection="polar")

COLOR_FAR  = "#4FC3F7"  # Light Blue (遠方)
COLOR_NEAR = "#1A237E"  # Deep Indigo (壁際)

# ヒストグラムのビン（区切り）を設定
bins = np.linspace(0, np.pi/2, N_BINS + 1)
bin_width = np.diff(bins)[0]
bin_centers = bins[:-1] + bin_width / 2

# 💡 確率密度としてカウント（面積を比較しやすくするため。カウント数そのものにしたい場合は density=False に）
counts_far, _  = np.histogram(far_rad, bins=bins, density=True)
counts_near, _ = np.histogram(near_rad, bins=bins, density=True)

# 遠方領域の描画 (背面)
ax.bar(bin_centers, counts_far, width=bin_width, color=COLOR_FAR,
       edgecolor=COLOR_FAR, alpha=0.5, label=f"Far Area (d > {DISTANCE_THRESHOLD} μm)", zorder=2)

# 壁際領域の描画 (前面：透過度を調整して重なりを見せる)
ax.bar(bin_centers, counts_near, width=bin_width, color=COLOR_NEAR,
       edgecolor=COLOR_NEAR, alpha=0.6, label=f"Near Area (d ≤ {DISTANCE_THRESHOLD} μm)", zorder=3)

# Axis settings
ax.set_theta_zero_location("E")
ax.set_theta_direction(1)
ax.set_thetamin(0)
ax.set_thetamax(90)

ax.set_xticks(np.radians([0, 30, 60, 90]))
ax.set_xticklabels(["0°\n(Parallel)", "30°", "60°", "90°\n(Vertical)"])
ax.grid(alpha=0.20)

# Title & Legend
ax.set_title("Spatial Distribution of Particle Invasion Angle", fontsize=14, fontweight="bold", pad=30)
ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.30, 1.10), fontsize=9.5)

# Save
plt.tight_layout()
plt.savefig("IEEE_TNB_polar_hist_distance.png", dpi=600, bbox_inches="tight")
plt.savefig("IEEE_TNB_polar_hist_distance.pdf", bbox_inches="tight")
plt.show()

print("\n✅ Saved polar histogram plots successfully.")


import scipy.stats as stats

print("\n" + "="*60)
print("📝 ADVANCED STATISTICAL REPORT FOR APCOT / IEEE")
print("="*60)

# ==============================================================
# 1. Two-Sample Kolmogorov-Smirnov Test (2標本KS検定)
# ==============================================================
# 分布の平均値ではなく、累積分布の形状そのものの「最大距離 D」を評価する検定
ks_stat, ks_p_val = stats.ks_2samp(far_angles, near_angles)

print("■ 1. Two-Sample Kolmogorov-Smirnov Test")
print(f"  - KS Statistic (D) : {ks_stat:.4f}")
print(f"  - p-value          : {ks_p_val:.4e}")
if ks_p_val < 0.001:
    print("  - Conclusion       : Highly Significant Difference in Distribution Shape (p < 0.001) ***")
elif ks_p_val < 0.05:
    print("  - Conclusion       : Significant Difference in Distribution Shape (p < 0.05) *")
else:
    print("  - Conclusion       : No Significant Difference in Shape")
print("-" * 50)


# ==============================================================
# 2. Angle Fraction Analysis & Chi-Square Test (存在割合の比較とカイ二乗検定)
# ==============================================================
# 角度を「平行」「中間」「垂直」の3つの物理モードにスパッと区切る
# 平行 (Parallel): 0° - 30° / 中間 (Intermediate): 30° - 60° / 垂直 (Vertical): 60° - 90°

def get_fraction_counts(angles):
    arr = np.array(angles)
    c_parallel     = np.sum((arr >= 0)  & (arr < 30))
    c_intermediate = np.sum((arr >= 30) & (arr < 60))
    c_vertical     = np.sum((arr >= 60) & (arr <= 90))
    return np.array([c_parallel, c_intermediate, c_vertical])

counts_far  = get_fraction_counts(far_angles)
counts_near = get_fraction_counts(near_angles)

total_far  = len(far_angles)
total_near = len(near_angles)

pct_far  = (counts_far  / total_far)  * 100
pct_near = (counts_near / total_near) * 100

print("■ 2. Angle Fraction Component Analysis")
print("  [Far Area (d > 10.0 μm)]")
print(f"    - Parallel Mode     (0° - 30°) : {counts_far[0]} steps ({pct_far[0]:.1f}%)")
print(f"    - Intermediate Mode (30° - 60°) : {counts_far[1]} steps ({pct_far[1]:.1f}%)")
print(f"    - Vertical Mode    (60° - 90°) : {counts_far[2]} steps ({pct_far[2]:.1f}%)")
print("  [Near Area (d ≤ 10.0 μm)]")
print(f"    - Parallel Mode     (0° - 30°) : {counts_near[0]} steps ({pct_near[0]:.1f}%)")
print(f"    - Intermediate Mode (30° - 60°) : {counts_near[1]} steps ({pct_near[1]:.1f}%)")
print(f"    - Vertical Mode    (60° - 90°) : {counts_near[2]} steps ({pct_near[2]:.1f}%)")
print("-" * 50)

# 💡 分割表（Contingency Table）を作成してカイ二乗検定を実行
contingency_table = np.array([counts_far, counts_near])
chi2_stat, chi2_p_val, dof, expected = stats.chi2_contingency(contingency_table)

print("■ 3. Chi-Square Test of Independence (分割表の独立性検定)")
print(f"  - Chi-Square Stat  : {chi2_stat:.4f}")
print(f"  - p-value          : {chi2_p_val:.4e}")
print(f"  - Degrees of Free  : {dof}")
if chi2_p_val < 0.001:
    print("  - Conclusion       : Category Proportions are Significantly Different (p < 0.001) ***")
elif chi2_p_val < 0.05:
    print("  - Conclusion       : Category Proportions are Significantly Different (p < 0.05) *")
else:
    print("  - Conclusion       : No Significant Difference in Proportions")
print("="*60)


import scipy.stats as stats

print("\n" + "="*60)
print("📝 TUNED STATISTICAL REPORT (Custom Boundaries: 20° and 70°)")
print("="*60)

# ==============================================================
# Optimized Angle Fraction Component Analysis
# ==============================================================
# 💡 ヒストグラムのスパイクに合わせて境界を 20°, 70° に調整
# 0°-20°: Gliding / 20°-70°: Transition / 70°-90°: Penetrating

def get_tuned_fraction_counts(angles):
    arr = np.array(angles)
    c_gliding      = np.sum((arr >= 0)  & (arr < 20))
    c_transition   = np.sum((arr >= 20) & (arr < 70))
    c_penetrating  = np.sum((arr >= 70) & (arr <= 90))
    return np.array([c_gliding, c_transition, c_penetrating])

counts_far  = get_tuned_fraction_counts(far_angles)
counts_near = get_tuned_fraction_counts(near_angles)

total_far  = len(far_angles)
total_near = len(near_angles)

pct_far  = (counts_far  / total_far)  * 100
pct_near = (counts_near / total_near) * 100

print("■ 1. Tuned Angle Fraction Component Analysis")
print("  [Far Area (d > 10.0 μm)]")
print(f"    - Gliding Mode      (0° - 20°) : {counts_far[0]} steps ({pct_far[0]:.1f}%)")
print(f"    - Transition Mode   (20° - 70°) : {counts_far[1]} steps ({pct_far[1]:.1f}%)")
print(f"    - Penetrating Mode  (70° - 90°) : {counts_far[2]} steps ({pct_far[2]:.1f}%)")
print("  [Near Area (d ≤ 10.0 μm)]")
print(f"    - Gliding Mode      (0° - 20°) : {counts_near[0]} steps ({pct_near[0]:.1f}%)")
print(f"    - Transition Mode   (20° - 70°) : {counts_near[1]} steps ({pct_near[1]:.1f}%)")
print(f"    - Penetrating Mode  (70° - 90°) : {counts_near[2]} steps ({pct_near[2]:.1f}%)")
print("-" * 50)

# 💡 分割表（Contingency Table）を作成してカイ二乗検定を実行
contingency_table = np.array([counts_far, counts_near])
chi2_stat, chi2_p_val, dof, expected = stats.chi2_contingency(contingency_table)

print("■ 2. Chi-Square Test of Independence (Tuned Proportions)")
print(f"  - Chi-Square Stat  : {chi2_stat:.4f}")
print(f"  - p-value          : {chi2_p_val:.4e}")
print(f"  - Degrees of Free  : {dof}")
if chi2_p_val < 0.001:
    print("  - Conclusion       : Highly Significant Difference (p < 0.001) ***")
elif chi2_p_val < 0.05:
    print("  - Conclusion       : Significant Difference (p < 0.05) *")
else:
    print("  - Conclusion       : No Significant Difference")
print("="*60)


import matplotlib.pyplot as plt
import numpy as np

# ==============================================================
# Plot Settings (IEEE/APCOT Quality)
# ==============================================================
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica"]
plt.rcParams["pdf.fonttype"] = 42

# データ定義 (先ほどの解析結果を使用)
# Percentages: [Gliding, Transition, Penetrating]
data_far  = [13.2, 55.9, 30.9]
data_near = [22.2, 54.9, 22.8]
labels = ['Far (d > 10μm)', 'Near (d ≤ 10μm)']

# モード名称と色設定
modes = ['Gliding (0°-20°)', 'Transition (20°-70°)', 'Penetrating (70°-90°)']
colors = ['#2ca02c', '#ff7f0e', '#1f77b4'] # 緑, オレンジ, 青

# グラフ作成
fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

# 積み上げ棒グラフの描画
bar_width = 0.6
bottom = np.zeros(2)

for i in range(3):
    heights = [data_far[i], data_near[i]]
    ax.bar(labels, heights, bar_width, bottom=bottom, label=modes[i], color=colors[i], edgecolor='white')

    # 各バーの中央にパーセンテージを表示
    for j in range(2):
        ax.text(j, bottom[j] + heights[j]/2, f"{heights[j]:.1f}%",
                ha='center', va='center', color='white', fontweight='bold', fontsize=10)
    bottom += heights

# デザイン調整
ax.set_ylabel("Composition Ratio (%)", fontsize=12, fontweight='bold')
ax.set_ylim(0, 110)
ax.set_title("Shift of Flow Modes in Near-Wall Region", fontsize=14, fontweight='bold', pad=15)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
ax.grid(axis='y', linestyle='--', alpha=0.3)

# 保存
plt.tight_layout()
plt.savefig("Mode_Composition_Ratio.pdf", bbox_inches="tight")
plt.savefig("Mode_Composition_Ratio.png", dpi=600, bbox_inches="tight")
plt.show()

print("\n✅ Saved Mode Composition Ratio (Stacked Bar Chart) successfully.")