import ktc

# 1. 把你已經擁有的 nef 檔案路徑放進 List
nef_list = ["blazeface.nef", "pfld.nef"] # 請確認檔名與你的實際檔案一致

# 2. 設定輸出資料夾
output_dir = "./combo_nef"

print("🔄 開始合併 NEF...")
# 3. 呼叫你截圖中的神祕 API
ktc.combine_nef(nef_list, output_path=output_dir)

print(f"✨ 合併完成！請去 {output_dir} 資料夾領取你的雙核大腦！")