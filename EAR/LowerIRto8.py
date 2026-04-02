import onnx

# 1. 載入你剛產出的深層斷頭模型
model_path = "pfld-sim.onnx"
model = onnx.load(model_path)

# 2. 強行將 IR 版本降級為 8 (Kneron Docker 相容版本)
model.ir_version = 8

# 3. 儲存為相容版本
compat_model_path = "pfld-sim_compat.onnx"
onnx.save(model, compat_model_path)

print(f"✅ 降級完成！請使用 {compat_model_path} 進行 Docker 轉換。")