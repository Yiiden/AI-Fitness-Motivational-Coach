import torch
from blazeface import BlazeFace

def main():
    print("正在初始化 BlazeFace 模型...")
    # 1. 實例化模型 (預設為 128x128 的架構)
    net = BlazeFace()

    # 2. 載入預訓練權重
    # 請確保 'blazeface.pth' 檔案與此腳本在同一個目錄下
    net.load_weights("blazeface.pth")
    
    # 將模型切換到推論 (Evaluation) 模式，這對匯出 ONNX 非常重要
    # 這樣可以關閉 Dropout 或 Batch Normalization 的訓練行為
    net.eval()

    # 3. 建立一個虛擬輸入張量 (Dummy Input)
    # 形狀為 (Batch Size, Channels, Height, Width) -> (1, 3, 128, 128)
    # 這個張量是用來讓 PyTorch 「跑一次」模型，藉此記錄運算流程圖 (Graph)
    dummy_input = torch.randn(1, 3, 128, 128)

    onnx_filename = "blazeface.onnx"
    
    print("開始匯出 ONNX 檔案...")
    # 4. 執行 ONNX 匯出
    torch.onnx.export(
        net,                           # 要匯出的模型
        dummy_input,                   # 虛擬輸入
        onnx_filename,                 # 儲存的檔案名稱
        export_params=True,            # 將權重一起匯出
        opset_version=11,              # 設定 opset 為 11 (對 Kneron Toolchain 相容性極佳)
        do_constant_folding=True,      # 執行常數摺疊優化，提升推論速度
        input_names=['input'],         # 定義輸入節點名稱
        output_names=['regressors', 'classificators'] # BlazeFace 會輸出這兩個陣列：偏移量與置信度
    )

    print(f"✅ 匯出大功告成！請檢查目錄下是否生成了 '{onnx_filename}'。")

if __name__ == "__main__":
    main()