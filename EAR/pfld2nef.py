import ktc, onnx, shutil, os
import numpy as np
from PIL import Image

# --- Config ---
ONNX_PATH = "pfld-sim_compat.onnx"       # 使用我們剛剛產出的簡化版模型
DATA_DIR  = "images"        # ⭐️ 請建立這個資料夾，裡面放「裁切好的臉部特寫照」
MODEL_ID, MODEL_VER, CHIP = 20010, "0001", "730" # ID 設為 20010 避免衝突
OUT_DIR   = "pfld_output"         # 輸出目錄
IMG_SIZE  = 112                   # ⭐️ PFLD 的標準輸入尺寸是 112x112

# --- Helpers ---
def ensure_dir(p):
    if not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)

def list_images(root):
    return [os.path.join(r, f) for r, _, fs in os.walk(root)
            for f in fs if f.lower().endswith((".jpg",".jpeg",".png",".bmp"))]

def preprocess(path):
    """
    PFLD 的前處理非常簡單：
    1. 讀取並轉 RGB
    2. 直接 Resize 成 112x112 (因為假設輸入已經是裁切好的臉部了)
    3. 歸一化除以 255.0
    """
    im = Image.open(path).convert("RGB")
    im = im.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(im, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[None]  # HWC -> CHW -> NCHW
    return arr

def safe_copy_to_dir(src_path, dst_dir):
    ensure_dir(dst_dir)
    dst_path = os.path.join(dst_dir, os.path.basename(src_path))
    if os.path.abspath(src_path) == os.path.abspath(dst_path):
        return dst_path
    shutil.copy(src_path, dst_path)
    return dst_path

# --- Pipeline ---
print(f"[1/4] 載入並優化 ONNX: {ONNX_PATH}")
m = onnx.load(ONNX_PATH)
m = ktc.onnx_optimizer.onnx2onnx_flow(m)
print(f"   ONNX 模型載入成功")

print("[2/4] 建立 KL730 的 ModelConfig...")
km = ktc.ModelConfig(MODEL_ID, MODEL_VER, CHIP, onnx_model=m)

print(f"[3/4] 從 {DATA_DIR} 讀取大頭照進行量化 (Quantization)...")
imgs = list_images(DATA_DIR)
assert imgs, f"找不到圖片！請在 {DATA_DIR} 資料夾中放入幾張「裁切好的臉部特寫照片」。"
arrs = [preprocess(p) for p in imgs]
input_name = m.graph.input[0].name
bie_path = km.analysis({input_name: arrs})
safe_copy_to_dir(bie_path, OUT_DIR)
print(f"   ✅ BIE 儲存至: {bie_path}")

print("[4/4] 編譯成 NEF...")
nef_path = ktc.compile([km])
safe_copy_to_dir(nef_path, OUT_DIR)
print(f"   ✅ NEF 儲存至: {nef_path}")
print(f"\n✨ 大功告成！PFLD 模型已安穩地躺在 {OUT_DIR}/ 囉！")