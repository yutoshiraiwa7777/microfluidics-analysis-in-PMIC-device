import io, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from google.colab import files
from PIL import Image

# ── 1. パラメータ設定 ──────────────────────────────────────────
PIXEL_SIZE   = 3.158   # μm / pixel
DT           = 0.1     # 秒 / フレーム
N_FRAMES     = 10
TIME_AXIS    = np.arange(N_FRAMES) * DT  # 0.0, 0.1, … , 0.9 秒

# 各グループの追跡粒子数
N_PARTICLES_CTRL = 50
N_PARTICLES_EXP  = 54

# ── 2. データ処理関数の定義 ────────────────────────────────────
def process_condition_data(condition_name, n_particles):
    print(f"\n=======================================================")
    print(f" 【{condition_name}】のデータを入力してください（追跡粒子数: {n_particles}）")
    print(f"=======================================================")
    
    # ① 固有の座標データの読み込み
    print("① 粒子の座標データ（CSVファイル）をアップロードしてください …")
    uploaded_csv = files.upload()
    raw_csv = list(uploaded_csv.values())[0]

    for enc in ("utf-8", "utf-8-sig", "cp932", "shift_jis", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(raw_csv), encoding=enc)
            break
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue

    df.columns = [c.strip() for c in df.columns]
    if "x [μm]" not in df.columns:
        cols = list(df.columns)
        df = df.rename(columns={cols[0]: "x [μm]", cols[1]: "y [μm]"})

    # ② この条件固有のDistance Mapの読み込み
    print(f"② この条件に対応する 絨毛の Distance Map（TIF画像）をアップロードしてください …")
    uploaded_tif = files.upload()
    raw_tif = list(uploaded_tif.values())[0]
    img = Image.open(io.BytesIO(raw_tif))
    distance_map = np.array(img).astype(float)

    dist_matrix = np.zeros((n_particles, N_FRAMES))

    for p in range(n_particles):
        sl = slice(p * N_FRAMES, (p + 1) * N_FRAMES)
        x_um = df.iloc[sl]["x [μm]"].values.astype(float)
        y_um = df.iloc[sl]["y [μm]"].values.astype(float)
        
        col_idx = np.clip(np.round(x_um / PIXEL_SIZE).astype(int), 0, distance_map.shape[1] - 1)
        row_idx = np.clip(np.round(y_um / PIXEL_SIZE).astype(int), 0, distance_map.shape[0] - 1)
        
        # それぞれ固有のDistance Mapから距離を抽出
        raw_distances = distance_map[row_idx, col_idx] * PIXEL_SIZE
        
        clamped_distances = np.copy(raw_distances)
        inside_flag = False
        for f in range(N_FRAMES):
            if inside_flag:
                clamped_distances[f] = 0.0
            elif raw_distances[f] <= 0.0:
                clamped_distances[f] = 0.0
                inside_flag = True  
                
            dist_matrix[p] = clamped_distances

    # 各グループの初期位置を基準とした累積接近変位の計算
    disp_matrix = np.zeros((n_particles, N_FRAMES))
    for p in range(n_particles):
        disp_matrix[p] = dist_matrix[p][0] - dist_matrix[p]
        
    return disp_matrix

# ── 3. 別々のファイルセットを順次アップロード・処理 ────────────
# それぞれ独立したCSVとTIF画像が適用されます
disp_control = process_condition_data("No deformation (変形なし)", N_PARTICLES_CTRL)
disp_deform  = process_condition_data("Deformation (変形あり)", N_PARTICLES_EXP)

# ── 4. 統計量（各フレームでの平均・標準偏差）の計算 ───────────
mean_control = np.mean(disp_control, axis=0)
sd_control   = np.std(disp_control, axis=0, ddof=1)

mean_deform  = np.mean(disp_deform, axis=0)
sd_deform    = np.std(disp_deform, axis=0, ddof=1)

# ── 5. プロットの作成 ──────────────────────────────────────────
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
plt.rcParams['pdf.fonttype'] = 42

COLOR_CTRL_INDIV = "#CFD8DC"  # 変形なし（個々の粒子：薄いグレー）
COLOR_CTRL_MAIN  = "#78909C"  # 変形なし（平均ライン：濃いグレー）
COLOR_DEF_INDIV  = "#FFCDD2"  # 変形あり（個々の粒子：薄い赤）
COLOR_DEF_MAIN   = "#D32F2F"  # 変形あり（平均ライン：濃い赤）

fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=150)

# 1. 個々の粒子の累積接近推移をプロット
for p in range(N_PARTICLES_CTRL):
    ax.plot(TIME_AXIS, disp_control[p], color=COLOR_CTRL_INDIV, alpha=0.3, linewidth=0.8, zorder=1)

for p in range(N_PARTICLES_EXP):
    ax.plot(TIME_AXIS, disp_deform[p], color=COLOR_DEF_INDIV, alpha=0.3, linewidth=0.8, zorder=2)

# 2. 変形なし（No deformation）の平均ラインとエラーバー
ax.plot(TIME_AXIS, mean_control, color=COLOR_CTRL_MAIN, linewidth=2.0, 
        label=f"No deformation (Mean, n={N_PARTICLES_CTRL})", zorder=3)
ax.errorbar(TIME_AXIS, mean_control, yerr=sd_control, fmt='o', color=COLOR_CTRL_MAIN,
            markersize=4, elinewidth=1.2, capsize=3.0, capthick=1.2, zorder=5)

# 3. 変形あり（Deformation）の平均ラインとエラーバー
ax.plot(TIME_AXIS, mean_deform, color=COLOR_DEF_MAIN, linewidth=2.0, 
        label=f"Deformation (Mean, n={N_PARTICLES_EXP})", zorder=4)
ax.errorbar(TIME_AXIS, mean_deform, yerr=sd_deform, fmt='o', color=COLOR_DEF_MAIN,
            markersize=4, elinewidth=1.2, capsize=3.0, capthick=1.2, zorder=6)

# ── 6. 軸・外枠のデザイン最適化 ─────────────────────────────────
ax.set_xlim(0.0, 0.95)
ax.set_xticks(TIME_AXIS)
ax.set_xlabel("Time (s)", fontsize=10, color="#111111", labelpad=6)
ax.set_ylabel("Cumulative approach displacement ($\mu$m)", fontsize=10, color="#111111", labelpad=6)

ax.legend(loc="upper left", frameon=False, fontsize=9)

ax.tick_params(direction="in", which="major", colors="#111111", labelsize=9, length=4, width=0.7)
ax.grid(False)

for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
for sp in ["left", "bottom"]:
    ax.spines[sp].set_color("#111111")
    ax.spines[sp].set_linewidth(0.7)

plt.savefig("all_frames_independent_maps_plot.png", bbox_inches="tight", dpi=300)
plt.show()

print("✅ all_frames_independent_maps_plot.png 保存完了。それぞれのDistance Mapを適用した比較グラフが出力されました。")






import io, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import urllib.request

# ── 0. Arialフォントを直接ダウンロードしてMatplotlibに強制登録 ──────────
font_url = "https://github.com/matomo-org/travis-scripts/raw/master/fonts/Arial.ttf"
font_path = "Arial.ttf"

if not os.path.exists(font_path):
    print("⏳ Arialフォントファイルを直接ダウンロード中...")
    urllib.request.urlretrieve(font_url, font_path)
    import matplotlib.font_manager as fm
    fm.fontManager.addfont(font_path)
    print("✅ Arialフォントの強制登録が完了しました。")

from google.colab import files
from PIL import Image
from scipy import stats

# ── 1. パラメータ設定 ──────────────────────────────────────────
PIXEL_SIZE   = 3.158   # μm / pixel
DT           = 0.1     # 秒 / フレーム
N_FRAMES     = 10      # 全10フレーム（0.0s〜0.9s）

# 各グループの追跡粒子数
N_PARTICLES_CTRL = 50
N_PARTICLES_EXP  = 54

# ── 2. データ処理関数の定義 ────────────────────────────────────
def process_condition_data(condition_name, n_particles):
    print(f"\n=======================================================")
    print(f" 【{condition_name}】のデータを入力してください（追跡粒子数: {n_particles}）")
    print(f"=======================================================")
    
    print("① 粒子の座標データ（CSVファイル）をアップロードしてください …")
    uploaded_csv = files.upload()
    raw_csv = list(uploaded_csv.values())[0]

    for enc in ("utf-8", "utf-8-sig", "cp932", "shift_jis", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(raw_csv), encoding=enc)
            break
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue

    df.columns = [c.strip() for c in df.columns]
    if "x [μm]" not in df.columns:
        cols = list(df.columns)
        df = df.rename(columns={cols[0]: "x [μm]", cols[1]: "y [μm]"})

    print(f"② この条件に対応する 絨毛の Distance Map（TIF画像）をアップロードしてください …")
    uploaded_tif = files.upload()
    raw_tif = list(uploaded_tif.values())[0]
    img = Image.open(io.BytesIO(raw_tif))
    distance_map = np.array(img).astype(float)

    dist_matrix = np.zeros((n_particles, N_FRAMES))

    for p in range(n_particles):
        sl = slice(p * N_FRAMES, (p + 1) * N_FRAMES)
        x_um = df.iloc[sl]["x [μm]"].values.astype(float)
        y_um = df.iloc[sl]["y [μm]"].values.astype(float)
        
        col_idx = np.clip(np.round(x_um / PIXEL_SIZE).astype(int), 0, distance_map.shape[1] - 1)
        row_idx = np.clip(np.round(y_um / PIXEL_SIZE).astype(int), 0, distance_map.shape[0] - 1)
        
        raw_distances = distance_map[row_idx, col_idx] * PIXEL_SIZE
        
        clamped_distances = np.copy(raw_distances)
        inside_flag = False
        for f in range(N_FRAMES):
            if inside_flag:
                clamped_distances[f] = 0.0
            elif raw_distances[f] <= 0.0:
                clamped_distances[f] = 0.0
                inside_flag = True  
                
            dist_matrix[p] = clamped_distances

    disp_matrix = np.zeros((n_particles, N_FRAMES))
    for p in range(n_particles):
        disp_matrix[p] = dist_matrix[p][0] - dist_matrix[p]
        
    return disp_matrix

# ── 3. 有意差ブラケットを描画する専用関数の定義 ────────────────────
def add_significance_bracket(ax, x1, x2, y, mark, mark_font={'fontname':'Arial', 'fontsize':10, 'fontweight':'bold'}):
    hige_len = 0.5 
    y_hige = y - hige_len
    ax.plot([x1, x1, x2, x2], [y_hige, y, y, y_hige], color="#111111", linewidth=0.7, zorder=5)
    if mark != "":
        ax.text((x1 + x2) / 2, y + 0.1, mark, ha='center', va='bottom', color="#111111", zorder=6, **mark_font)

# ── 4. データの読み込みと計算 ──────────────────────────────────
disp_control = process_condition_data("No deformation (変形なし)", N_PARTICLES_CTRL)
disp_deform  = process_condition_data("Deformation (変形あり)", N_PARTICLES_EXP)

# ── 5. 0.0秒のデータを排除し、0.1秒〜0.9秒（インデックス1〜9）を抽出 ──
# これにより、プロット対象は全9ステップになります
PLOT_FRAMES = np.arange(1, N_FRAMES)  # 1, 2, ... , 9
TIME_AXIS   = PLOT_FRAMES * DT        # 0.1, 0.2, ... , 0.9 秒

mean_control = np.mean(disp_control[:, PLOT_FRAMES], axis=0)
se_control   = np.std(disp_control[:, PLOT_FRAMES], axis=0, ddof=1) / np.sqrt(N_PARTICLES_CTRL)

mean_deform  = np.mean(disp_deform[:, PLOT_FRAMES], axis=0)
se_deform    = np.std(disp_deform[:, PLOT_FRAMES], axis=0, ddof=1) / np.sqrt(N_PARTICLES_EXP)

# ── 6. 複数並列棒グラフの作成 ──────────────────────────────────
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['pdf.fonttype'] = 42

COLOR_MAIN   = "#111111"
COLOR_BAR_CTRL = "#ECEFF1"  
COLOR_BAR_DEF  = "#FFCDD2"  

fig, ax = plt.subplots(figsize=(9.0, 5.5), dpi=150)

# 横軸の棒の位置インデックス（0.1sが配列の0番目、0.9sが8番目になるよう設定）
x_indices = np.arange(len(PLOT_FRAMES))
bar_width = 0.35

# 1. No deformation の棒とエラーバー
ax.bar(x_indices - bar_width/2, mean_control, yerr=se_control, 
       width=bar_width, color=COLOR_BAR_CTRL, edgecolor=COLOR_MAIN, linewidth=0.7,
       error_kw={'elinewidth': 0.8, 'capsize': 2.0, 'capthick': 0.8, 'ecolor': COLOR_MAIN},
       label=f"Baseline flow (Control) (n={N_PARTICLES_CTRL})", zorder=2)

# 2. Deformation の棒とエラーバー
ax.bar(x_indices + bar_width/2, mean_deform, yerr=se_deform, 
       width=bar_width, color=COLOR_BAR_DEF, edgecolor=COLOR_MAIN, linewidth=0.7,
       error_kw={'elinewidth': 0.8, 'capsize': 2.0, 'capthick': 0.8, 'ecolor': COLOR_MAIN},
       label=f"Deformation-induced approach flow (n={N_PARTICLES_EXP})", zorder=2)

# ── 7. Welchのt検定の実行と有り差ブラケットの描画 ──────────────────
print("\n===== Welch's t-test Results (0.1s - 0.9s) =====")
for idx, f in enumerate(PLOT_FRAMES):
    ctrl_data = disp_control[:, f]
    exp_data  = disp_deform[:, f]
    
    t_stat, p_val = stats.ttest_ind(ctrl_data, exp_data, equal_var=False)
    print(f"Time {f*DT:.1f} s: t-statistic = {t_stat:.4f}, p-value = {p_val:.4e}")
    
    if p_val < 0.001:
        significance_mark = "***"
    elif p_val < 0.01:
        significance_mark = "**"
    elif p_val < 0.05:
        significance_mark = "*"
    else:
        significance_mark = ""
        
    highest_top = max(mean_control[idx] + se_control[idx], mean_deform[idx] + se_deform[idx])
    bracket_y = highest_top + 1.2 
    
    bracket_x_ctrl = idx - bar_width/2
    bracket_x_deform = idx + bar_width/2
    
    add_significance_bracket(ax, bracket_x_ctrl, bracket_x_deform, bracket_y, significance_mark)

# ── 8. 軸・外枠のデザイン最適化（原点合わせの調整） ──────────────────
xtick_labels = [f"{t:.1f}" for t in TIME_AXIS]

ax.set_xticks(x_indices)
ax.set_xticklabels(xtick_labels, fontsize=9, color=COLOR_MAIN, fontname='Arial')
ax.set_xlabel("Time (s)", fontsize=10, color=COLOR_MAIN, labelpad=6, fontname='Arial')
ax.set_ylabel("Cumulative approach displacement (Mean $\pm$ S.E., $\mu$m)", fontsize=10, color=COLOR_MAIN, labelpad=6, fontname='Arial')

# 【重要】0.1秒の棒の左側に適度な余白を作りつつ、Y軸（原点）と近づける設定
ax.set_xlim(left=-0.6, right=len(PLOT_FRAMES) - 0.4)
ax.set_ylim(bottom= -10)

# 上部余白の調整
current_ylim_top = ax.get_ylim()[1]
ax.set_ylim(top=current_ylim_top * 1.12)

ax.legend(loc="upper left", frameon=False, fontsize=9, prop={'family': 'Arial'})

ax.tick_params(direction="in", which="major", colors=COLOR_MAIN, labelsize=9, length=4, width=0.7)
ax.grid(False)

for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
for sp in ["left", "bottom"]:
    ax.spines[sp].set_color(COLOR_MAIN)
    ax.spines[sp].set_linewidth(0.7)

plt.savefig("all_frames_barplot_no_zero_frame.tif", bbox_inches="tight", dpi=300)
plt.show()

print("\n✅ all_frames_barplot_no_zero_frame.tif 保存完了。")
print("0.0秒を排除し、0.1秒始まりで原点に合わせた論文用グラフが完成しました。")


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
# ⚙️ 固定パラメータ（3.0 μm 閾値）
# ==============================================================
THRESHOLD_DISTANCE = 3.0  
PIXEL_SIZE  = 3.158   # μm / pixel
N_FRAMES    = 10

COLOR_MAIN  = "#111111"
COLOR_CTRL  = "#EAECEE"  # Control群（グレー）
COLOR_EXP   = "#EC7063"  # Experimental群（赤系）

def process_group_include_all(df, distance_map, group_name=""):
    """
    初期位置の除外をせず、10フレーム内で一度でも閾値内に入った粒子をカウントする
    """
    ny, nx = distance_map.shape
    n_particles = len(df) // N_FRAMES
    
    reached_count = 0
    total_count = n_particles  # 分母は全粒子数
    
    for p in range(n_particles):
        sl = slice(p * N_FRAMES, (p + 1) * N_FRAMES)
        p_data = df.iloc[sl]
        
        x_um = p_data["x [μm]"].values.astype(float)
        y_um = p_data["y [μm]"].values.astype(float)
        
        # 💡 10フレーム（Index 0〜9）の全期間をチェック
        has_reached = False
        for frame in range(N_FRAMES):
            px = int(np.clip(x_um[frame] / PIXEL_SIZE, 0, nx - 1))
            py = int(np.clip(y_um[frame] / PIXEL_SIZE, 0, ny - 1))
            
            raw_d = distance_map[py, px]
            d = raw_d * PIXEL_SIZE
            
            # 距離が 3.0 μm 以下、またはピクセル値が 0（壁面・壁内）なら接触判定
            if d <= THRESHOLD_DISTANCE or raw_d == 0:
                has_reached = True
                break  # 一度でも接触したらその粒子は判定終了
                
        if has_reached:
            reached_count += 1
            
    failed_count = total_count - reached_count
    print(f"⚙️ [{group_name}] 総粒子数(分母): {total_count}個 | 接触成功(分子): {reached_count}個 | 未接触: {failed_count}個")
    return reached_count, total_count

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
# 2. データの順次読み込み
# ==============================================================
print("①【Control群】の Particle CSV を選択してください")
df_ctrl = safe_read_csv(files.upload())
print("②【Control群】の Distance map (TIF) を選択してください")
map_ctrl = np.array(Image.open(io.BytesIO(list(files.upload().values())[0]))).astype(float)

print("\n------------------------------------------------------------")
print("③【Experimental群】の Particle CSV を選択してください")
df_exp = safe_read_csv(files.upload())
print("④【Experimental群】の Distance map (TIF) を選択してください")
map_exp = np.array(Image.open(io.BytesIO(list(files.upload().values())[0]))).astype(float)

# 初期位置を含めて全粒子を愚直に計算
ctrl_reached, ctrl_total = process_group_include_all(df_ctrl, map_ctrl, "Control")
exp_reached, exp_total   = process_group_include_all(df_exp, map_exp, "Experimental")

ctrl_failed = ctrl_total - ctrl_reached
exp_failed  = exp_total - exp_reached

# ==============================================================
# 3. 統計検定（フィッシャーの正確確率検定）
# ==============================================================
contingency_table = [[exp_reached, exp_failed], [ctrl_reached, ctrl_failed]]
oddsratio, p_val = stats.fisher_exact(contingency_table)

rate_ctrl = (ctrl_reached / ctrl_total) * 100 if ctrl_total > 0 else 0
rate_exp  = (exp_reached / exp_total) * 100 if exp_total > 0 else 0

print("\n" + "="*60)
print(f"📊 【初期位置含む・3.0 μm判定】比較結果:")
print(f"  Control群      : 接触 {ctrl_reached} / 全体 {ctrl_total} ({rate_ctrl:.1f} %)")
print(f"  Experimental群 : 接触 {exp_reached} / 全体 {exp_total} ({rate_exp:.1f} %)")
print(f"  👉 フィッシャー検定 p値: p = {p_val:.4e}")
print("="*60 + "\n")

# ==============================================================
# 4. グラフ描画
# ==============================================================
fig, ax = plt.subplots(figsize=(4.5, 5.5), dpi=300)

labels = ['Control', 'Experimental']
rates  = [rate_ctrl, rate_exp]

bars = ax.bar(labels, rates, color=[COLOR_CTRL, COLOR_EXP], 
              edgecolor=COLOR_MAIN, linewidth=0.8, width=0.45)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{height:.1f}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold', fontname='Arial')

y_max = max(rates) if max(rates) > 0 else 10
bracket_y = y_max * 1.1
ax.set_ylim(0, max(15, bracket_y * 1.3))

mark = r"$\ast\ast\ast$" if p_val < 0.001 else r"$\ast\ast$" if p_val < 0.01 else r"$\ast$" if p_val < 0.05 else "ns"
ax.plot([0, 0, 1, 1], [bracket_y * 0.96, bracket_y, bracket_y, bracket_y * 0.96], color=COLOR_MAIN, linewidth=0.8)
ax.text(0.5, bracket_y * 1.02, f"{mark}\n(p = {p_val:.2e})", ha='center', va='bottom', fontsize=11, fontweight='bold', fontname='Arial')

ax.set_ylabel(f"Bacterial Contact Rate within {THRESHOLD_DISTANCE} μm (%)\n[Including all observed particles]", fontsize=10, fontweight='bold', fontname='Arial', labelpad=6)
ax.tick_params(direction="in", which="major", colors=COLOR_MAIN, labelsize=11, length=4, width=0.6)

for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
for sp in ["left", "bottom"]: ax.spines[sp].set_color(COLOR_MAIN); ax.spines[sp].set_linewidth(0.6)

plt.tight_layout()
plt.show()