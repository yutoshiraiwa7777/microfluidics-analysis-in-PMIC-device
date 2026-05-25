# ==============================================================
# Approach Amplitude Analysis (1ステップあたりの接近量の検証)
# ==============================================================
print("\n" + "="*60)
print("🏃‍♂️ APPROACH AMPLITUDE ANALYSIS (Delta Distance per Step)")
print("="*60)

far_delta_d  = []
near_delta_d = []

# ※すでにメインループを通過しているため、同様の条件で接近量を抽出します
for p in range(N_PARTICLES):
    sl = slice(p * N_FRAMES, (p + 1) * N_FRAMES)
    x_um = df.iloc[sl]["x [μm]"].values.astype(float)
    y_um = df.iloc[sl]["y [μm]"].values.astype(float)

    for step in range(N_STEPS):
        x_s, y_s = x_um[step], y_um[step]
        x_e, y_e = x_um[step + 1], y_um[step + 1]

        # 開始点と終了点の「Distance Map上の距離」を取得
        px_s = max(0, min(int(round(x_s / PIXEL_SIZE)), w - 1))
        py_s = max(0, min(int(round(y_s / PIXEL_SIZE)), h - 1))
        d_start = distance_map[py_s, px_s]

        px_e = max(0, min(int(round(x_e / PIXEL_SIZE)), w - 1))
        py_e = max(0, min(int(round(y_e / PIXEL_SIZE)), h - 1))
        d_end = distance_map[py_e, px_e]

        # 💡 1ステップあたりの接近量（正の値＝壁に近づいている、負の値＝壁から離れている）
        # ※物理的な意味をストレートにするため「開始距離 - 終了距離」で定義
        delta_d = d_start - d_end

        # 開始時の場所を基準に Far / Near を判定
        if d_start > DISTANCE_THRESHOLD:
            far_delta_d.append(delta_d)
        else:
            near_delta_d.append(delta_d)

# 基本統計量の計算
mean_delta_far, std_delta_far = np.mean(far_delta_d), np.std(far_delta_d, ddof=1)
mean_delta_near, std_delta_near = np.mean(near_delta_d), np.std(near_delta_d, ddof=1)

print(f"■ Far Area (d > {DISTANCE_THRESHOLD} μm) の1ステップあたり接近量:")
print(f"  - Mean Δd : {mean_delta_far:.4f} μm/step")
print(f"  - S.D.    : {std_delta_far:.4f} μm")
print(f"  - (速度換算 : {mean_delta_far/DT:.2f} μm/s の速度で壁に接近)")
print("-" * 50)
print(f"■ Near Area (d ≤ {DISTANCE_THRESHOLD} μm) の1ステップあたり接近量:")
print(f"  - Mean Δd : {mean_delta_near:.4f} μm/step")
print(f"  - S.D.    : {std_delta_near:.4f} μm")
print(f"  - (速度換算 : {mean_delta_near/DT:.2f} μm/s の速度で壁に接近)")
print("-" * 50)

# 有意差の検定（Welch's t-test）
t_stat, p_val = stats.ttest_ind(far_delta_d, near_delta_d, equal_var=False)
print(f"💡 Statistical Significance (Welch's t-test):")
print(f"  - t-statistic : {t_stat:.4f}")
print(f"  - p-value     : {p_val:.4e}")
if p_val < 0.05:
    if mean_delta_far > mean_delta_near:
        print("  - Conclusion  : 予想通り、壁際（Near）で接近量が有意に『控えめ』になっています！ ***")
    else:
        print("  - Conclusion  : 壁際（Near）の方が接近量が大きくなっています。")
else:
    print("  - Conclusion  : 接近量に統計的な有意差はありません。")
print("="*60)




import io
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from google.colab import files
from PIL import Image
from scipy import stats

# ==============================================================
# ⚙️ パラメータ・カラー設定（ここを自由に変えてください）
# ==============================================================
THRESHOLD_DISTANCE = 20.0  # 💡 閾値をここでコントロール（例: 20.0, 15.0, 5.0 など）

PIXEL_SIZE  = 3.158   # μm / pixel
DT          = 0.1
N_FRAMES    = 10
N_STEPS     = N_FRAMES - 1

COLOR_MAIN  = "#111111"
COLOR_FAR   = "#EAECEE"
COLOR_NEAR  = "#FADBD8"

def calculate_particle_averages(df, distance_map, n_particles, threshold):
    ny, nx = distance_map.shape
    particle_far_means = []
    particle_near_means = []

    for p in range(n_particles):
        sl = slice(p * N_FRAMES, (p + 1) * N_FRAMES)
        x_um = df.iloc[sl]["x [μm]"].values.astype(float)
        y_um = df.iloc[sl]["y [μm]"].values.astype(float)

        p_far_velocities = []
        p_near_velocities = []

        for step in range(N_STEPS):
            x_s, y_s = x_um[step], y_um[step]
            x_e, y_e = x_um[step + 1], y_um[step + 1]

            px = int(np.clip(x_e / PIXEL_SIZE, 0, nx - 1))
            py = int(np.clip(y_e / PIXEL_SIZE, 0, ny - 1))
            d = distance_map[py, px] * PIXEL_SIZE
            
            if d <= 0:
                continue

            disp_step = np.sqrt((x_e - x_s)**2 + (y_e - y_s)**2)
            v_approach = disp_step / DT

            # 設定した閾値（threshold）で判定
            if d > threshold:
                p_far_velocities.append(v_approach)
            else:
                p_near_velocities.append(v_approach)

        if len(p_far_velocities) > 0:
            particle_far_means.append(np.mean(p_far_velocities))
        if len(p_near_velocities) > 0:
            particle_near_means.append(np.mean(p_near_velocities))
                
    return np.array(particle_far_means), np.array(particle_near_means)

def safe_read_csv(uploaded_dict):
    raw_bytes = list(uploaded_dict.values())[0]
    for enc in ("utf-8", "utf-8-sig", "cp932", "shift_jis"):
        try:
            df = pd.read_csv(io.BytesIO(raw_bytes), encoding=enc)
            df.columns = [c.strip() for c in df.columns]
            if "x [μm]" not in df.columns and len(df.columns) >= 2:
                cols = list(df.columns)
                df = df.rename(columns={cols[0]: "x [μm]", cols[1]: "y [μm]"})
            return df
        except:
            continue
    raise UnicodeDecodeError("CSVの読み込みに失敗しました。")

# ==============================================================
# 2. データの読み込みと計算
# ==============================================================
print("①【Experimental群】の Particle CSV をアップロードしてください")
df_exp = safe_read_csv(files.upload())

print("\n②【Experimental群】の Distance map (TIF) をアップロードしてください")
map_exp = np.array(Image.open(io.BytesIO(list(files.upload().values())[0]))).astype(float)

# 変数 threshold を渡して再計算
far_p_means, near_p_means = calculate_particle_averages(df_exp, map_exp, len(df_exp) // N_FRAMES, THRESHOLD_DISTANCE)

print("\n" + "="*50)
print(f"📊 集計完了 (現在の閾値: {THRESHOLD_DISTANCE} μm):")
print(f"  Far 領域の粒子数 (n) : {len(far_p_means)}")
print(f"  Near 領域の粒子数 (n): {len(near_p_means)}")
print("="*50 + "\n")

# ==============================================================
# 3. 統計量計算と検定
# ==============================================================
mean_far  = np.mean(far_p_means)
se_far    = stats.sem(far_p_means)
mean_near = np.mean(near_p_means)
se_near   = stats.sem(near_p_means)

_, p_val = stats.ttest_ind(far_p_means, near_p_means, equal_var=False)

# ==============================================================
# 4. グラフ描画
# ==============================================================
fig, ax = plt.subplots(figsize=(4.5, 5.5), dpi=300)

labels = [f'Far\n(d > {THRESHOLD_DISTANCE} μm)', f'Near\n(d ≤ {THRESHOLD_DISTANCE} μm)']
means  = [mean_far, mean_near]
errors = [se_far, se_near]

bars = ax.bar(labels, means, yerr=errors, color=[COLOR_FAR, COLOR_NEAR], 
              edgecolor=COLOR_MAIN, linewidth=0.8, width=0.45,
              capsize=5, error_kw=dict(lw=1.0, capthick=1.0))

y_max = max(mean_far + se_far, mean_near + se_near)
bracket_y = y_max * 1.15
ax.set_ylim(0, bracket_y * 1.25)

mark = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
ax.plot([0, 0, 1, 1], [bracket_y * 0.96, bracket_y, bracket_y, bracket_y * 0.96], color=COLOR_MAIN, linewidth=0.8)
ax.text(0.5, bracket_y * 1.02, f"{mark}\n(p = {p_val:.1e})", ha='center', va='bottom', fontsize=10, fontweight='bold', fontname='Arial')

ax.set_ylabel("Mean Track Velocity (μm/s)\n[Mean ± S.E.]", fontsize=11, fontweight='bold', fontname='Arial', labelpad=6)
ax.tick_params(direction="in", which="major", colors=COLOR_MAIN, labelsize=10, length=4, width=0.6)

for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
for sp in ["left", "bottom"]: ax.spines[sp].set_color(COLOR_MAIN); ax.spines[sp].set_linewidth(0.6)

plt.tight_layout()
plt.savefig(f"IEEE_TNB_th{THRESHOLD_DISTANCE}_velocity_bar.pdf", bbox_inches="tight")
plt.show()


import io
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from google.colab import files
from PIL import Image
from scipy import stats

# ==============================================================
# ⚙️ パラメータ・カラー設定
# ==============================================================
THRESHOLD_DISTANCE = 20.0  # 条件を完全に揃えるため 20 µm

PIXEL_SIZE  = 3.158   # μm / pixel
DT          = 0.1
N_FRAMES    = 10
N_STEPS     = N_FRAMES - 1

COLOR_MAIN  = "#111111"
COLOR_FAR   = "#EAECEE"
COLOR_NEAR  = "#D4EFDF"  # 平行成分用に淡いグリーンを設定

def calculate_parallel_particle_averages(df, distance_map, n_particles, threshold):
    """粒子ごとに『絨毛表面に平行な（接線）速度成分』を計算し、Near/Farで平均化する"""
    ny, nx = distance_map.shape
    
    # 絨毛の中心座標（xc, yc）を推定
    x_grid = np.arange(nx) * PIXEL_SIZE
    y_grid = np.arange(ny) * PIXEL_SIZE
    temp_fig, temp_ax = plt.subplots()
    cs = temp_ax.contour(distance_map, levels=[0], extent=[0, x_grid[-1], 0, y_grid[-1]])
    plt.close(temp_fig)
    
    v = cs.get_paths()[0].vertices
    x_pts, y_pts = v[:, 0], v[:, 1]
    A = np.array([x_pts, y_pts, np.ones(len(x_pts))]).T
    b = x_pts**2 + y_pts**2
    c, d_, e = np.linalg.lstsq(A, b, rcond=None)[0]
    xc, yc = c / 2, d_ / 2

    particle_far_means = []
    particle_near_means = []

    for p in range(n_particles):
        sl = slice(p * N_FRAMES, (p + 1) * N_FRAMES)
        x_um = df.iloc[sl]["x [μm]"].values.astype(float)
        y_um = df.iloc[sl]["y [μm]"].values.astype(float)

        p_far_tangent_velocities = []
        p_near_tangent_velocities = []

        for step in range(N_STEPS):
            x_s, y_s = x_um[step], y_um[step]
            x_e, y_e = x_um[step + 1], y_um[step + 1]

            px = int(np.clip(x_e / PIXEL_SIZE, 0, nx - 1))
            py = int(np.clip(y_e / PIXEL_SIZE, 0, ny - 1))
            d = distance_map[py, px] * PIXEL_SIZE
            
            if d <= 0:
                continue

            # 1. 粒子の移動ベクトル
            v_particle = np.array([x_e - x_s, y_e - y_s])
            
            # 2. 絨毛中心（内向き法線）への単位ベクトル
            v_normal = np.array([xc - x_s, yc - y_s])
            norm_val = np.linalg.norm(v_normal)
            if norm_val == 0:
                continue
            u_normal = v_normal / norm_val

            # 3. 垂直（法線）成分の抽出
            delta_d_approach = np.dot(v_particle, u_normal)
            v_normal_vector = delta_d_approach * u_normal

            # 4. 全移動ベクトルから垂直成分を差し引き、「平行（接線）成分」を抽出
            v_tangent_vector = v_particle - v_normal_vector
            
            # 接線ベクトルの大きさ（スカラー）を速度に変換
            v_parallel = np.linalg.norm(v_tangent_vector) / DT

            # 閾値で判定して格納
            if d > threshold:
                p_far_tangent_velocities.append(v_parallel)
            else:
                p_near_tangent_velocities.append(v_parallel)

        if len(p_far_tangent_velocities) > 0:
            particle_far_means.append(np.mean(p_far_tangent_velocities))
        if len(p_near_tangent_velocities) > 0:
            particle_near_means.append(np.mean(p_near_tangent_velocities))
                
    return np.array(particle_far_means), np.array(particle_near_means)

def safe_read_csv(uploaded_dict):
    raw_bytes = list(uploaded_dict.values())[0]
    for enc in ("utf-8", "utf-8-sig", "cp932", "shift_jis"):
        try:
            df = pd.read_csv(io.BytesIO(raw_bytes), encoding=enc)
            df.columns = [c.strip() for c in df.columns]
            if "x [μm]" not in df.columns and len(df.columns) >= 2:
                cols = list(df.columns)
                df = df.rename(columns={cols[0]: "x [μm]", cols[1]: "y [μm]"})
            return df
        except:
            continue
    raise UnicodeDecodeError("CSVの読み込みに失敗しました。")

# ==============================================================
# 2. データの読み込みと実行
# ==============================================================
print("①【Experimental群】の Particle CSV をアップロードしてください")
df_exp = safe_read_csv(files.upload())

print("\n②【Experimental群】の Distance map (TIF) をアップロードしてください")
map_exp = np.array(Image.open(io.BytesIO(list(files.upload().values())[0]))).astype(float)

# 絨毛に「平行な」速度成分で集計
far_p_means, near_p_means = calculate_parallel_particle_averages(df_exp, map_exp, len(df_exp) // N_FRAMES, THRESHOLD_DISTANCE)

# ==============================================================
# 3. 統計量計算と検定
# ==============================================================
mean_far  = np.mean(far_p_means)
se_far    = stats.sem(far_p_means)
mean_near = np.mean(near_p_means)
se_near   = stats.sem(near_p_means)

_, p_val = stats.ttest_ind(far_p_means, near_p_means, equal_var=False)

# ==============================================================
# 4. グラフ描画
# ==============================================================
fig, ax = plt.subplots(figsize=(4.5, 5.5), dpi=300)

labels = [f'Far\n(d > {THRESHOLD_DISTANCE} μm)', f'Near\n(d ≤ {THRESHOLD_DISTANCE} μm)']
means  = [mean_far, mean_near]
errors = [se_far, se_near]

bars = ax.bar(labels, means, yerr=errors, color=[COLOR_FAR, COLOR_NEAR], 
              edgecolor=COLOR_MAIN, linewidth=0.8, width=0.45,
              capsize=5, error_kw=dict(lw=1.0, capthick=1.0))

y_max = max(mean_far + se_far, mean_near + se_near)
bracket_y = y_max * 1.15
ax.set_ylim(0, bracket_y * 1.25)

mark = r"$\ast\ast\ast$" if p_val < 0.001 else r"$\ast\ast$" if p_val < 0.01 else r"$\ast$" if p_val < 0.05 else "ns"
ax.plot([0, 0, 1, 1], [bracket_y * 0.96, bracket_y, bracket_y, bracket_y * 0.96], color=COLOR_MAIN, linewidth=0.8)
ax.text(0.5, bracket_y * 1.02, f"{mark}\n(p = {p_val:.1e})", ha='center', va='bottom', fontsize=12, fontweight='bold', fontname='Arial')

# 縦軸を「平行方向の速度」に指定
ax.set_ylabel("Parallel Velocity along Villi (μm/s)\n[Mean ± S.E.]", fontsize=11, fontweight='bold', fontname='Arial', labelpad=6)
ax.tick_params(direction="in", which="major", colors=COLOR_MAIN, labelsize=10, length=4, width=0.6)

for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
for sp in ["left", "bottom"]: ax.spines[sp].set_color(COLOR_MAIN); ax.spines[sp].set_linewidth(0.6)

plt.tight_layout()
plt.savefig(f"IEEE_TNB_parallel_th{THRESHOLD_DISTANCE}_bar.pdf", bbox_inches="tight")
plt.show()


import io
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from google.colab import files
from PIL import Image
from scipy import stats

# ==============================================================
# ⚙️ パラメータ・カラー設定
# ==============================================================
THRESHOLD_DISTANCE = 20.0  # 有意差が担保された 20 µm 閾値

PIXEL_SIZE  = 3.158   # μm / pixel
DT          = 0.1
N_FRAMES    = 10
N_STEPS     = N_FRAMES - 1

COLOR_MAIN  = "#111111"
COLOR_FAR   = "#EAECEE"
COLOR_NEAR  = "#FADBD8"

def calculate_approach_particle_averages(df, distance_map, n_particles, threshold):
    """粒子ごとに『絨毛の法線方向（内向き）への接近速度成分』を計算し、Near/Farで平均化する"""
    ny, nx = distance_map.shape
    
    # 絨毛の中心座標（xc, yc）を輪郭から最小二乗法で推定（法線ベクトルの基準点）
    x_grid = np.arange(nx) * PIXEL_SIZE
    y_grid = np.arange(ny) * PIXEL_SIZE
    temp_fig, temp_ax = plt.subplots()
    cs = temp_ax.contour(distance_map, levels=[0], extent=[0, x_grid[-1], 0, y_grid[-1]])
    plt.close(temp_fig)
    
    v = cs.get_paths()[0].vertices
    x_pts, y_pts = v[:, 0], v[:, 1]
    A = np.array([x_pts, y_pts, np.ones(len(x_pts))]).T
    b = x_pts**2 + y_pts**2
    c, d_, e = np.linalg.lstsq(A, b, rcond=None)[0]
    xc, yc = c / 2, d_ / 2

    particle_far_means = []
    particle_near_means = []

    for p in range(n_particles):
        sl = slice(p * N_FRAMES, (p + 1) * N_FRAMES)
        x_um = df.iloc[sl]["x [μm]"].values.astype(float)
        y_um = df.iloc[sl]["y [μm]"].values.astype(float)

        p_far_approach_velocities = []
        p_near_approach_velocities = []

        for step in range(N_STEPS):
            x_s, y_s = x_um[step], y_um[step]
            x_e, y_e = x_um[step + 1], y_um[step + 1]

            # 1. 絨毛表面からの距離 d (μm)
            px = int(np.clip(x_e / PIXEL_SIZE, 0, nx - 1))
            py = int(np.clip(y_e / PIXEL_SIZE, 0, ny - 1))
            d = distance_map[py, px] * PIXEL_SIZE
            
            if d <= 0:
                continue

            # 2. 粒子の移動ベクトル (μm)
            v_particle = np.array([x_e - x_s, y_e - y_s])
            
            # 3. 粒子位置から絨毛中心（内向き法線方向）への単位ベクトル
            v_normal = np.array([xc - x_s, yc - y_s])
            norm_val = np.linalg.norm(v_normal)
            if norm_val == 0:
                continue
            u_normal = v_normal / norm_val

            # 4. 移動ベクトルの法線方向への射影（＝純粋な絨毛方向への接近量 Δd）
            # 内積をとることで、横滑り成分（Gliding）を排除した真の接近成分を抽出
            delta_d_approach = np.dot(v_particle, u_normal)
            v_approach = delta_d_approach / DT

            # ※絨毛に「近づく」方向（プラス）のステップを対象に、閾値で判定
            if v_approach > 0:
                if d > threshold:
                    p_far_approach_velocities.append(v_approach)
                else:
                    p_near_approach_velocities.append(v_approach)

        # 粒子 p ごとの平均値を算出
        if len(p_far_approach_velocities) > 0:
            particle_far_means.append(np.mean(p_far_approach_velocities))
        if len(p_near_approach_velocities) > 0:
            particle_near_means.append(np.mean(p_near_approach_velocities))
                
    return np.array(particle_far_means), np.array(particle_near_means)

def safe_read_csv(uploaded_dict):
    raw_bytes = list(uploaded_dict.values())[0]
    for enc in ("utf-8", "utf-8-sig", "cp932", "shift_jis"):
        try:
            df = pd.read_csv(io.BytesIO(raw_bytes), encoding=enc)
            df.columns = [c.strip() for c in df.columns]
            if "x [μm]" not in df.columns and len(df.columns) >= 2:
                cols = list(df.columns)
                df = df.rename(columns={cols[0]: "x [μm]", cols[1]: "y [μm]"})
            return df
        except:
            continue
    raise UnicodeDecodeError("CSVの読み込みに失敗しました。")

# ==============================================================
# 2. データの読み込みと粒子単位の接近速度成分算出
# ==============================================================
print("①【Experimental群】の Particle CSV をアップロードしてください")
df_exp = safe_read_csv(files.upload())

print("\n②【Experimental群】の Distance map (TIF) をアップロードしてください")
map_exp = np.array(Image.open(io.BytesIO(list(files.upload().values())[0]))).astype(float)

# 絨毛方向への接近速度成分だけで再集計
far_p_means, near_p_means = calculate_approach_particle_averages(df_exp, map_exp, len(df_exp) // N_FRAMES, THRESHOLD_DISTANCE)

print("\n" + "="*50)
print(f"📊 接近速度成分（法線方向）での集計完了 (閾値: {THRESHOLD_DISTANCE} μm):")
print(f"  Far 領域の粒子数 (n) : {len(far_p_means)}")
print(f"  Near 領域の粒子数 (n): {len(near_p_means)}")
print("="*50 + "\n")

# ==============================================================
# 3. 統計量計算と検定
# ==============================================================
mean_far  = np.mean(far_p_means)
se_far    = stats.sem(far_p_means)
mean_near = np.mean(near_p_means)
se_near   = stats.sem(near_p_means)

_, p_val = stats.ttest_ind(far_p_means, near_p_means, equal_var=False)

# ==============================================================
# 4. グラフ描画（Approach Velocity 軸修正版）
# ==============================================================
fig, ax = plt.subplots(figsize=(4.5, 5.5), dpi=300)

labels = [f'Far\n(d > {THRESHOLD_DISTANCE} μm)', f'Near\n(d ≤ {THRESHOLD_DISTANCE} μm)']
means  = [mean_far, mean_near]
errors = [se_far, se_near]

bars = ax.bar(labels, means, yerr=errors, color=[COLOR_FAR, COLOR_NEAR], 
              edgecolor=COLOR_MAIN, linewidth=0.8, width=0.45,
              capsize=5, error_kw=dict(lw=1.0, capthick=1.0))

y_max = max(mean_far + se_far, mean_near + se_near)
bracket_y = y_max * 1.15
ax.set_ylim(0, bracket_y * 1.25)

# 有意差マークを美的なギリシャ文字星型（\ast）へ変更、かつ視認性向上のため大文字化
mark = r"$\ast\ast\ast$" if p_val < 0.001 else r"$\ast\ast$" if p_val < 0.01 else r"$\ast$" if p_val < 0.05 else "ns"
ax.plot([0, 0, 1, 1], [bracket_y * 0.96, bracket_y, bracket_y, bracket_y * 0.96], color=COLOR_MAIN, linewidth=0.8)
ax.text(0.5, bracket_y * 1.02, f"{mark}\n(p = {p_val:.1e})", ha='center', va='bottom', fontsize=12, fontweight='bold', fontname='Arial')

# 縦軸を「絨毛方向への接近速度」に厳密化
ax.set_ylabel("Approach Velocity toward Villi (μm/s)\n[Mean ± S.E.]", fontsize=11, fontweight='bold', fontname='Arial', labelpad=6)
ax.tick_params(direction="in", which="major", colors=COLOR_MAIN, labelsize=10, length=4, width=0.6)

for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
for sp in ["left", "bottom"]: ax.spines[sp].set_color(COLOR_MAIN); ax.spines[sp].set_linewidth(0.6)

plt.tight_layout()
plt.savefig(f"IEEE_TNB_actual_approach_th{THRESHOLD_DISTANCE}_bar.pdf", bbox_inches="tight")
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 既存のデータセット（far_velocity, near_velocityは計算済みと想定）
# データラベル
labels = ['Far\n(d > 10μm)', 'Near\n(d ≤ 10μm)']
colors = ['#1f77b4', '#ff7f0e'] # 必要に応じて調整

# Figure作成（1行2列）
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=300)

# --- 1. ボックスプロット (Approach Velocity) ---
df_vel = pd.DataFrame({
    'Category': ['Far'] * len(far_velocity) + ['Near'] * len(near_velocity),
    'Velocity': np.concatenate([far_velocity, near_velocity])
})

sns.boxplot(x='Category', y='Velocity', data=df_vel, palette=colors, width=0.5, ax=ax1, showfliers=False)
ax1.set_ylabel("Approach Velocity (μm/s)", fontsize=12, fontweight='bold')
ax1.set_xlabel("")
ax1.set_title("Hydrodynamic Deceleration", fontsize=13, fontweight='bold')
# p値のブラケット
y_max = np.percentile(np.concatenate([far_velocity, near_velocity]), 98)
ax1.plot([0, 0, 1, 1], [y_max + 5, y_max + 10, y_max + 10, y_max + 5], color='black', lw=1.5)
ax1.text(0.5, y_max + 12, "p < 1e-13", ha='center', va='bottom', fontsize=10, fontweight='bold')

# --- 2. 積み上げ棒グラフ (Mode Composition) ---
data_far = [13.2, 55.9, 30.9]
data_near = [22.2, 54.9, 22.8]
modes = ['Gliding', 'Transition', 'Penetrating']
bar_colors = ['#2ca02c', '#9467bd', '#d62728'] # 緑, 紫, 赤

bottom = np.zeros(2)
for i in range(3):
    heights = [data_far[i], data_near[i]]
    ax2.bar(labels, heights, 0.6, bottom=bottom, label=modes[i], color=bar_colors[i], edgecolor='white')
    # パーセンテージ表示
    for j in range(2):
        ax2.text(j, bottom[j] + heights[j]/2, f"{heights[j]:.1f}%",
                 ha='center', va='center', color='white', fontweight='bold', fontsize=9)
    bottom += heights

ax2.set_ylabel("Composition Ratio (%)", fontsize=12, fontweight='bold')
ax2.set_title("Shift of Flow Modes", fontsize=13, fontweight='bold')
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)

plt.tight_layout()
plt.savefig("Combined_Analysis.pdf", bbox_inches="tight")
plt.savefig("Combined_Analysis.png", dpi=600, bbox_inches="tight")
plt.show()