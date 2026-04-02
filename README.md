# AI健身激勵教練（AI Fitness Motivational Coach）

## 專案背景與動機
基於邊緣運算平台開發的「AI 健身激勵教練」，能透過攝影機即時辨識使用者的臉部發力特徵，並利用大型語言模型 (LLM) 生成貼近真人專業教練的激勵語，最後透過語音合成 (TTS) 即時播放。

---

## 專案亮點
- **全離線運算**：所有推論（視覺、語言模型、語音）皆在 Kneo Pi 上完成，無需連網，保護隱私。
- **資源動態調度**：以「20 秒循環策略」，解決邊緣設備 NPU 與 CPU 的資源競爭問題。
- **多模態整合**：結合電腦視覺 (CV)、大型語言模型 (LLM) 與語音合成 (TTS) 技術。

---

## 系統架構
本專案採用異步多線程處理，將不同性質的任務分配至最適合的運算單元：

| 任務類型 | 使用模型 | 運算單元 | 說明 |
| :--- | :--- | :--- | :--- |
| **臉部/關鍵點偵測** | BlazeFace + PFLD | **NPU** | 提供穩定且低延遲的視覺推論 (13+ Infer FPS)。 |
| **激勵文字生成** | Gemma 3 270M | **CPU** | 透過 `llama-server` 以 API 形式提供推論服務。 |
| **語音合成 (TTS)** | Piper TTS | **CPU** | 獨立線程處理，確保語音流暢不卡頓。 |

---

## 專案檔案結構
本專案的檔案結構包含主程式、模型資源以及視覺模型的轉檔腳本目錄（Gemma3 270M 轉檔腳本目錄詳見 `llama_cpp_kneopi` 資料夾）。以下為核心檔案配置：
```
AI_Fitness_Coach_KneoPi/
├── combo_realtime_Step2.py       # 主程式 (結合 CV、LLM 請求與 TTS 播放)
├── anchors.npy                   # BlazeFace 邊界框解碼所需的錨點資料
├── en_US-lessac-medium.onnx      # Piper TTS 語音模型
├── en_US-lessac-medium.onnx.json # Piper TTS 模型設定檔
└── res/                          # NPU 資源目錄 (程式預設讀取路徑)
    └── models/
        └── KL730/
            └── Blazeface_Pfld_combo/
                └── Blazeface_Pfld_combo/
                    └── blazeface_pfld.nef  # 合併完成的視覺模型 (BlazeFace + PFLD)
```

---

## 核心邏輯
### 1. 發力判定 (Exertion Detection)
系統利用 PFLD 提取 98 個臉部關鍵點，計算 **EAR (Eye Aspect Ratio)** 來量化受訓者的疲勞與發力程度。
- **EAR 公式**：計算眼部垂直與水平特徵點的距離比例。
- **判定標準**：
    - `EAR < 0.20`: 極度發力 (Exhausted)。
    - `0.20 <= EAR <= 0.25`: 穩定發力 (Sweet Spot)。
    - `EAR > 0.25`: 挑戰不足 (No Challenge)。

### 2. 20 秒循環資源管理 (Resource Scheduling)
為了避免 CPU/NPU 負載過高，系統將運作分為兩個階段：
- **前 10 秒（取樣階段）**：NPU 專注於視覺推論，統計 EAR 數值分佈。
- **後 10 秒（決策執行）**：
    1. **暫停 NPU 推論**，釋放 CPU 影像前後處理資源。
    2. 根據統計結果向 `llama-server` 請求生成激勵語。
    3. 執行 Piper TTS 合成並播放音訊。

---

## 模型轉檔與部署
該部分詳細記錄了將 PyTorch 模型（BlazeFace 與 PFLD）轉換為 Kneron KL730 NPU 可執行之 `.nef` 模型的過程。

### 1. 環境依賴
根據終端機輸出，我們使用了特定的虛擬環境。主要的套件包含：
* `onnx` (1.20.1)
* `onnx-ir` (0.2.0)
* `onnxruntime` (1.24.2)
* `onnxsim` (0.6.1)
* `torch` (2.10.0)
* `torchvision` (0.25.0)
* `Pillow` (12.1.1)

### 2. BlazeFace 模型轉檔流程 (位於 `Face_Detection` 資料夾)
參考 [BlazeFace-PyTorch](https://github.com/hollance/BlazeFace-PyTorch)，轉檔包含三個主要步驟：

#### 2.1 從 PyTorch 匯出 ONNX (`export_onnx.py`)
將預訓練好的 PyTorch 權重 (`blazeface.pth`) 轉換為 ONNX 格式 (`blazeface.onnx`)。
- **輸入尺寸設定**：使用 `(1, 3, 128, 128)` 的虛擬張量（Dummy Input）以建立運算流程圖。
- **輸出節點被設定為**：`regressors` (偏移量) 與 `classificators` (置信度)。
- **參數設定**：設定 `opset_version=11` 以達到較好的 Kneron Toolchain 相容性。

#### 2.2 ONNX IR 版本降級 (`LowerIRto8.py`)
為確保與 Kneron Docker 的相容性，必須將產生的 ONNX 模型的 IR 版本降級。
- 讀取 `blazeface.onnx`。
- 強制設定 `model.ir_version = 8`。
- 儲存為相容版 `blazeface_compat.onnx`。

#### 2.3 ONNX 編譯至 Kneron NEF (`blazeface2nef.py`)
利用 Kneron Toolchain (`ktc`) 執行量化（Quantization）與編譯。
- **參數配置**：
  - 輸入模型：`blazeface_compat.onnx`
  - Model ID：`20009` (避免衝突的自訂 ID)
  - 晶片 (CHIP)：`730` (代表 KL730)
- **量化設定**：
  - 需要在 `images` 目錄下準備臉部照片作為校正集 (Calibration Data)。
  - 圖片前處理：先經過 Resize (Letterbox) 轉換至 `128x128`，並轉為 RGB 排列，然後將數值正規化 (`/ 255.0`)。
- **輸出結果**：完成後，將在 `blazeface_output/` 生成 `*.nef` 與分析用的 `.bie` 檔案。

### 3. PFLD 模型轉檔流程 (位於 `EAR` 資料夾)
參考 [PFLD-pytorch](https://github.com/polarisZhao/PFLD-pytorch)，PFLD 轉檔的流程與 BlazeFace 類似，步驟如下：

#### 3.1 原始 ONNX 模型準備
我們從 `pfld-sim.onnx` 這個簡化版的 ONNX 模型開始進行後續處理。

#### 3.2 ONNX IR 版本降級 (`LowerIRto8.py`)
與 BlazeFace 在轉換過程面臨的狀況相同，先透過腳本這一步進行相容性調整：
- 載入 `pfld-sim.onnx` 並強制設定 `model.ir_version = 8`。
- 產出兼融版本 `pfld-sim_compat.onnx`。

#### 3.3 ONNX 編譯至 Kneron NEF (`pfld2nef.py`)
使用 Kneron Toolchain (`ktc`) 進行編譯與優化。
- **參數配置**：
  - 輸入模型：`pfld-sim_compat.onnx`
  - Model ID：`20010` (與 BlazeFace 的 ID 區隔)
  - 晶片 (CHIP)：`730`
- **量化設定**：
  - 需要在 `images` 目錄下提供數張「裁切好的臉部特寫照片」。
  - 影像前處理：PFLD 不需要 Letterbox 操作，而是直接 Resize 成標準輸入尺寸 `112x112`，再轉為 RGB 並將數值正規化 (`/ 255.0`)。
- **輸出結果**：順利編譯完後，可以在 `pfld_output/` 中取得可部署至 KL730 執行推論的編譯結果檔案（如 `*.nef` 和 `.bie`）。

### 4. 合併 `.nef` 模型 (位於 `EARandFace_Detection_test` 資料夾)
為了在單一 Session 中同時調用兩個視覺模型，參考 Kneron Document Center [NEF Combine (Optional)](https://doc.kneron.com/docs/#toolchain/manual_5_nef/#53-nef-combine-optional)，使用 Kneron Toolchain (`ktc`) 將多個 `.nef` 檔案合併，執行 `combine.py`：
```python
import ktc

# 將編譯完成的 BlazeFace 與 PFLD nef 檔案合併
ktc.combine_nef(
    ['./blazeface.nef', './pfld.nef'], 
    output_path = "./combined_models"
)
```

---

## LLM 模型部署 (Gemma3 270M)
* **模型設定**：詳見`llama_cpp_kneopi`資料夾中的 `README.txt`。
* **啟動 `llama-server` 提供背景服務**：
```bash
./llama-server -m gemma-3-270m-it-Q8_0.nef -ngl 0 -c 1024 --port 8080
```

---

## 執行與 Demo

### 執行指令
1. **環境設定**：建立虛擬環境並安裝相依套件。
```bash
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install requests piper-tts opencv-python
```
2. **啟動背景服務**：開啟一個終端機執行 llama-server，強制使用 CPU (-ngl 0) 避免與視覺模型的 NPU 衝突。
```bash
./llama-server -m gemma-3-270m-it-Q8_0.nef -ngl 0 -c 1024 --port 8080
```
3. **執行主程式**：
```bash
python combo_realtime_Step2.py --src /dev/video0
```
(請根據您的 Webcam 裝置編號更改 `--src` 參數)
4. **關閉系統**：在顯示影像的視窗上點擊並按下鍵盤的 `q` 鍵，即可安全關閉主程式與攝影機連線。

### 互動體驗說明 (20 秒動態循環)
程式啟動後，畫面上會顯示即時的系統狀態，請依照以下 20 秒的節奏進行體驗：
* **前 10 秒（取樣階段 Phase: SAMPLING）**
    * 請對著鏡頭做出不同的表情。你可以嘗試「面無表情」或是「咬牙切齒/用力瞇眼」。
    * 觀察左下角的 Stats，系統會即時更新你這 10 秒內的發力百分比（Low / Mid / High）。
    * 觀察左上角的 Infer FPS，應穩定維持在 13 左右。
* **後 10 秒（決策與激勵階段 Phase: DECISION & TTS）**
    * **NPU 暫停運作**：你會發現 `Infer FPS` 降為 0.0，藉此釋放硬體資源。
    * **畫面文字**：畫面正下方會顯示紅字（如 `Result: EXHAUSTED (<0.20)`），代表教練對你上一組動作的判定。
    * **語音反饋**：約 1~2 秒後，系統會透過喇叭播放 Gemma 3 生成的專屬激勵語（例如："Keep pushing! You got this!"）。
* **循環重啟**
    * 滿 20 秒後，系統會自動清空計數器並恢復 NPU 推論，準備觀察你的下一組動作。

---

## 未來展望
* **動作標準度分析**：計畫整合 YOLOv8-Pose 進行全身姿勢評估。
* **個人化訓練記憶**：引入 RAG 技術，讓 AI 教練能根據使用者的歷史紀錄提供更精準的激勵。
* **商用化擴充**：將系統推廣至智慧健身房或數位孿生場域管理系統。