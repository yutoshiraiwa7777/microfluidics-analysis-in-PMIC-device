import io, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from google.colab import files
import matplotlib.ticker as ticker

# ==============================================================
#  ユーザー調整用パラメータ
# ==============================================================
# 既存の基本パラメータ
PIXEL_SIZE  = 3.158   # μm / pixel
DT          = 0.1     # 秒 / フレーム
N_PARTICLES = 50      # Control群の正しい粒子数
N_FRAMES    = 10      # 1粒子あたりのフレーム数

# 表示する視野のサイズ（縦横μm四方：中心の0から上下左右に350μmずつ広がります）
VIEW_SPAN = 700

# ── 2. CSV ファイルのアップロード ──────────────────────────────
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

# ── 3. Distance Map（TIF画像）の読み込み ──────────────────────
print("\n② 絨毛の Distance Map をアップロードしてください …")
from PIL import Image
uploaded_tif = files.upload()
raw_tif = list(uploaded_tif.values())[0]
img = Image.open(io.BytesIO(raw_tif))
distance_map = np.array(img).astype(float)

# 絨毛の重心（中心座標）を自動で計算するロジック
villus_indices = np.argwhere(distance_map <= np.min(distance_map) + 0.1)
if len(villus_indices) > 0:
    center_y_um = np.mean(villus_indices[:, 0]) * PIXEL_SIZE
    center_x_um = np.mean(villus_indices[:, 1]) * PIXEL_SIZE
    print(f"🔎 絨毛の実中心座標: x = {center_x_um:.1f} μm, y = {center_y_um:.1f} μm")
else:
    center_x_um = (distance_map.shape[1] * PIXEL_SIZE) / 2.0
    center_y_um = (distance_map.shape[0] * PIXEL_SIZE) / 2.0

# ── 4. プロットの作成 ──────────────────────────────────────────
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['pdf.fonttype'] = 42

COLOR_VILLUS = "#78909C"  
COLOR_TRACK  = "#455A64"  
COLOR_START  = "#2E7D32"  
COLOR_END    = "#C62828"  
COLOR_TEXT   = "#111111"

fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=150)

# ── 💡 絨毛表面の点群から円近似を行うブロック ────────────────────
# 1. 一時的に等高線を抽出して座標を取得
temp_fig, temp_ax = plt.subplots()
cs = temp_ax.contour(distance_map, levels=[0], 
                     extent=[0, distance_map.shape[1] * PIXEL_SIZE, 0, distance_map.shape[0] * PIXEL_SIZE])
plt.close(temp_fig)

# Matplotlib 3.8以降の仕様に対応してパスを取得
paths = cs.get_paths()
if len(paths) > 0:
    path = paths[0]
    v = path.vertices
    x_pts = v[:, 0]
    y_pts = v[:, 1]
    
    # 2. 最小二乗法による円フィッティング
    A = np.array([x_pts, y_pts, np.ones(len(x_pts))]).T
    b = x_pts**2 + y_pts**2
    c, d, e = np.linalg.lstsq(A, b, rcond=None)[0]
    
    # 中心座標 (xc, yc) と 半径 R の算出
    xc = c / 2
    yc = d / 2
    R = np.sqrt(e + xc**2 + yc**2)
    
    # 3. 近似された円弧をプロット (0〜360度を高解像度で描写)
    theta = np.linspace(0, 2 * np.pi, 500)
    x_fit = xc + R * np.cos(theta)
    y_fit = yc + R * np.sin(theta)
    
    # メイングラフに完全に滑らかな円として描画
    ax.plot(x_fit, y_fit, color=COLOR_VILLUS, linewidth=1.5, linestyle="-", zorder=1)
    print(f"📊 絨毛表面を円近似しました: 中心({xc:.1f}, {yc:.1f}) μm, 半径 R = {R:.1f} μm")
else:
    # 輪郭抽出に失敗した場合のバックアップ（従来の等高線描画）
    ax.contour(distance_map, levels=[0], colors=COLOR_VILLUS, linewidths=1.2, 
               extent=[0, distance_map.shape[1] * PIXEL_SIZE, 0, distance_map.shape[0] * PIXEL_SIZE],
               zorder=1)

# 💡 絨毛中心（相対原点0）を通る基準線を薄く描画 (論文スタイル)
ax.axhline(center_y_um, color="#DDDDDD", linewidth=0.8, linestyle="-", zorder=0) # Y=0の横線
ax.axvline(center_x_um, color="#DDDDDD", linewidth=0.8, linestyle="-", zorder=0) # X=0の縦線

# 全粒子の軌跡をプロット
valid_plots = 0
for p in range(N_PARTICLES):
    sl = slice(p * N_FRAMES, (p + 1) * N_FRAMES)
    
    if sl.start >= len(df):
        break
        
    x_um = df.iloc[sl]["x [μm]"].values.astype(float)
    y_um = df.iloc[sl]["y [μm]"].values.astype(float)
    
    if len(x_um) == 0 or len(y_um) == 0:
        continue
    
    # 軌跡の線
    ax.plot(x_um, y_um, color=COLOR_TRACK, linewidth=0.8, alpha=0.4, zorder=2)
    # 始点
    ax.scatter(x_um[0], y_um[0], color=COLOR_START, s=7, alpha=0.7, zorder=3, 
               label="Start ($t$ = 0 s)" if valid_plots == 0 else "")
    # 終点
    ax.scatter(x_um[-1], y_um[-1], color=COLOR_END, s=7, alpha=0.8, zorder=4, 
               label="End ($t$ = 0.9 s)" if valid_plots == 0 else "")
    
    valid_plots += 1

# ── 5. 軸・外枠のデザイン最適化（中心原点化） ─────────────────────────
# ① 絨毛中心をグラフの物理的な中央に固定
ax.set_xlim(center_x_um - VIEW_SPAN / 2.0, center_x_um + VIEW_SPAN / 2.0)
ax.set_ylim(center_y_um - VIEW_SPAN / 2.0, center_y_um + VIEW_SPAN / 2.0)

# ② 💡 絨毛中心（真ん中）から正確に100μm刻みで目盛りを配置する
half_span = VIEW_SPAN / 2.0
tick_offsets = np.arange(0, half_span + 1, 100)
relative_ticks = np.unique(np.concatenate([-tick_offsets, tick_offsets]))

x_ticks = center_x_um + relative_ticks
y_ticks = center_y_um + relative_ticks

ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)

# ③ 💡 物理目盛りの値を、文字上だけ「中心からの相対距離（0基準の100刻み）」に変換する
def relative_x_formatter(x, pos):
    val = x - center_x_um
    return f"{val:.0f}"

def relative_y_formatter(y, pos):
    val = y - center_y_um
    return f"{val:.0f}"

ax.xaxis.set_major_formatter(ticker.FuncFormatter(relative_x_formatter))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(relative_y_formatter))

# 縦横比を完全に 1:1 に固定
ax.set_aspect('equal', adjustable='box')

ax.set_xlabel("$x$ ($\mu$m)", fontsize=10, color=COLOR_TEXT, labelpad=5)
ax.set_ylabel("$y$ ($\mu$m)", fontsize=10, color=COLOR_TEXT, labelpad=5)
ax.set_title("Particle Trajectories around Villus (Control Group)", fontsize=11, color=COLOR_TEXT, fontweight="semibold", pad=10)

ax.tick_params(direction="in", which="major", colors=COLOR_TEXT, labelsize=9, length=4, width=0.7)
ax.grid(False)

for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
for sp in ["left", "bottom"]:
    ax.spines[sp].set_color(COLOR_TEXT)
    ax.spines[sp].set_linewidth(0.7)

if valid_plots > 0:
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")

plt.savefig("villus_perfect_fit_trajectory_map.png", bbox_inches="tight", dpi=300)
plt.show()

print("✅ 保存完了。絨毛表面を滑らかな円でフィッティングし、中心原点0の基準線（X=0, Y=0）を追加したグラフを出力しました。")




import io, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from google.colab import files
import matplotlib.ticker as ticker

# ==============================================================
#  ユーザー調整用パラメータ
# ==============================================================
# 既存の基本パラメータ
PIXEL_SIZE  = 3.158   # μm / pixel
DT          = 0.1     # 秒 / フレーム
N_PARTICLES = 54     # Control群の正しい粒子数
N_FRAMES    = 10      # 1粒子あたりのフレーム数

# 表示する視野のサイズ（縦横μm四方：中心の0から上下左右に350μmずつ広がります）
VIEW_SPAN = 600

# 💡 軌跡を抽出する判定エリアのサイズ（縦横400μm四方：中心から±200μm）
FILTER_SPAN = 600

# ── 2. CSV ファイルのアップロード ──────────────────────────────
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

# ── 3. Distance Map（TIF画像）の読み込み ──────────────────────
print("\n② 絨毛の Distance Map をアップロードしてください …")
from PIL import Image
uploaded_tif = files.upload()
raw_tif = list(uploaded_tif.values())[0]
img = Image.open(io.BytesIO(raw_tif))
distance_map = np.array(img).astype(float)

# 絨毛の重心（中心座標）を自動で計算するロジック
villus_indices = np.argwhere(distance_map <= np.min(distance_map) + 0.1)
if len(villus_indices) > 0:
    center_y_um = np.mean(villus_indices[:, 0]) * PIXEL_SIZE
    center_x_um = np.mean(villus_indices[:, 1]) * PIXEL_SIZE
    print(f"🔎 絨毛の実中心座標: x = {center_x_um:.1f} μm, y = {center_y_um:.1f} μm")
else:
    center_x_um = (distance_map.shape[1] * PIXEL_SIZE) / 2.0
    center_y_um = (distance_map.shape[0] * PIXEL_SIZE) / 2.0

# ── 4. プロットの作成 ──────────────────────────────────────────
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['pdf.fonttype'] = 42

COLOR_VILLUS = "#78909C"  
COLOR_TRACK  = "#455A64"  
COLOR_START  = "#2E7D32"  
COLOR_END    = "#C62828"  
COLOR_TEXT   = "#111111"

fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=150)

# ── 絨毛表面の点群から円近似を行うブロック ────────────────────
temp_fig, temp_ax = plt.subplots()
cs = temp_ax.contour(distance_map, levels=[0], 
                     extent=[0, distance_map.shape[1] * PIXEL_SIZE, 0, distance_map.shape[0] * PIXEL_SIZE])
plt.close(temp_fig)

paths = cs.get_paths()
if len(paths) > 0:
    path = paths[0]
    v = path.vertices
    x_pts = v[:, 0]
    y_pts = v[:, 1]
    
    A = np.array([x_pts, y_pts, np.ones(len(x_pts))]).T
    b = x_pts**2 + y_pts**2
    c, d, e = np.linalg.lstsq(A, b, rcond=None)[0]
    
    xc = c / 2
    yc = d / 2
    R = np.sqrt(e + xc**2 + yc**2)
    
    theta = np.linspace(0, 2 * np.pi, 500)
    x_fit = xc + R * np.cos(theta)
    y_fit = yc + R * np.sin(theta)
    
    ax.plot(x_fit, y_fit, color=COLOR_VILLUS, linewidth=1.5, linestyle="-", zorder=1)
else:
    ax.contour(distance_map, levels=[0], colors=COLOR_VILLUS, linewidths=1.2, 
               extent=[0, distance_map.shape[1] * PIXEL_SIZE, 0, distance_map.shape[0] * PIXEL_SIZE],
               zorder=1)

# 絨毛中心（相対原点0）を通る基準線を薄く描画
ax.axhline(center_y_um, color="#DDDDDD", linewidth=0.8, linestyle="-", zorder=0)
ax.axvline(center_x_um, color="#DDDDDD", linewidth=0.8, linestyle="-", zorder=0)

# 💡 400μm四方のフィルター判定とプロット
valid_plots = 0
excluded_plots = 0

# 判定用の境界値（中心から±200μm）
x_min_limit, x_max_limit = center_x_um - FILTER_SPAN / 2.0, center_x_um + FILTER_SPAN / 2.0
y_min_limit, y_max_limit = center_y_um - FILTER_SPAN / 2.0, center_y_um + FILTER_SPAN / 2.0

for p in range(N_PARTICLES):
    sl = slice(p * N_FRAMES, (p + 1) * N_FRAMES)
    
    if sl.start >= len(df):
        break
        
    x_um = df.iloc[sl]["x [μm]"].values.astype(float)
    y_um = df.iloc[sl]["y [μm]"].values.astype(float)
    
    if len(x_um) == 0 or len(y_um) == 0:
        continue
    
    # 💡 軌跡の全フレーム座標が400μm四方（±200μm）に収まっているか判定
    in_x = np.all((x_um >= x_min_limit) & (x_um <= x_max_limit))
    in_y = np.all((y_um >= y_min_limit) & (y_um <= y_max_limit))
    
    if in_x and in_y:
        # 条件内のみプロット
        ax.plot(x_um, y_um, color=COLOR_TRACK, linewidth=0.8, alpha=0.4, zorder=2)
        ax.scatter(x_um[0], y_um[0], color=COLOR_START, s=7, alpha=0.7, zorder=3, 
                   label="Start ($t$ = 0 s)" if valid_plots == 0 else "")
        ax.scatter(x_um[-1], y_um[-1], color=COLOR_END, s=7, alpha=0.8, zorder=4, 
                   label="End ($t$ = 0.9 s)" if valid_plots == 0 else "")
        valid_plots += 1
    else:
        excluded_plots += 1

# ── 5. 軸・外枠のデザイン最適化（中心原点化） ─────────────────────────
ax.set_xlim(center_x_um - VIEW_SPAN / 2.0, center_x_um + VIEW_SPAN / 2.0)
ax.set_ylim(center_y_um - VIEW_SPAN / 2.0, center_y_um + VIEW_SPAN / 2.0)

half_span = VIEW_SPAN / 2.0
tick_offsets = np.arange(0, half_span + 1, 100)
relative_ticks = np.unique(np.concatenate([-tick_offsets, tick_offsets]))

x_ticks = center_x_um + relative_ticks
y_ticks = center_y_um + relative_ticks

ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)

def relative_x_formatter(x, pos):
    val = x - center_x_um
    return f"{val:.0f}"

def relative_y_formatter(y, pos):
    val = y - center_y_um
    return f"{val:.0f}"

ax.xaxis.set_major_formatter(ticker.FuncFormatter(relative_x_formatter))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(relative_y_formatter))

ax.set_aspect('equal', adjustable='box')

ax.set_xlabel("$x$ ($\mu$m)", fontsize=10, color=COLOR_TEXT, labelpad=5)
ax.set_ylabel("$y$ ($\mu$m)", fontsize=10, color=COLOR_TEXT, labelpad=5)
ax.set_title("Filtered Particle Trajectories around Villus", fontsize=11, color=COLOR_TEXT, fontweight="semibold", pad=10)

ax.tick_params(direction="in", which="major", colors=COLOR_TEXT, labelsize=9, length=4, width=0.7)
ax.grid(False)

for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
for sp in ["left", "bottom"]:
    ax.spines[sp].set_color(COLOR_TEXT)
    ax.spines[sp].set_linewidth(0.7)

if valid_plots > 0:
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")

plt.savefig("filtered_trajectory_map_400um.png", bbox_inches="tight", dpi=300)
plt.show()

# ── 6. 統計データの書き出し ────────────────────────────────────
print("\n📊 💡 軌跡のフィルタリング結果")
print(f"  - 全粒子数: {valid_plots + excluded_plots} 個")
print(f"  - プロットされた粒子数 (400μm四方内): {valid_plots} 個")
print(f"  - 除外された粒子数 (一部でもエリア外に出たもの): {excluded_plots} 個")