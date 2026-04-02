import os, sys, cv2, kp, math
import numpy as np
import argparse, time, threading, platform
import requests
import wave
import subprocess
from piper import PiperVoice

# ==========================================
# 1. 數學與工具函數
# ==========================================
def pt_dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_ear(eye_pts):
    """計算 WFLW 8點格式的 EAR (Eye Aspect Ratio)"""
    v1 = pt_dist(eye_pts[1], eye_pts[7])
    v2 = pt_dist(eye_pts[2], eye_pts[6])
    v3 = pt_dist(eye_pts[3], eye_pts[5])
    h = pt_dist(eye_pts[0], eye_pts[4])
    if h == 0: return 0.0
    return (v1 + v2 + v3) / (3.0 * h)

def letterbox(im, new_shape=(128, 128)):
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114,114,114))
    return im, r, (left, top)

def decode_boxes(raw_boxes, anchors):
    x_center = raw_boxes[..., 0] / 128.0 * anchors[:, 2] + anchors[:, 0]
    y_center = raw_boxes[..., 1] / 128.0 * anchors[:, 3] + anchors[:, 1]
    w = raw_boxes[..., 2] / 128.0 * anchors[:, 2]
    h = raw_boxes[..., 3] / 128.0 * anchors[:, 3]
    return np.stack([y_center - h/2.0, x_center - w/2.0, y_center + h/2.0, x_center + w/2.0], axis=-1)

def overlap_similarity(box, other_boxes):
    max_xy = np.minimum(box[2:], other_boxes[:, 2:])
    min_xy = np.maximum(box[:2], other_boxes[:, :2])
    inter_dims = np.clip((max_xy - min_xy), a_min=0, a_max=None)
    inter_area = inter_dims[:, 0] * inter_dims[:, 1]
    area_a = (box[2] - box[0]) * (box[3] - box[1])
    area_b = (other_boxes[:, 2] - other_boxes[:, 0]) * (other_boxes[:, 3] - other_boxes[:, 1])
    return inter_area / (area_a + area_b - inter_area)

def nms(boxes, scores, score_threshold=0.6, iou_threshold=0.3):
    if len(boxes) == 0: return [], [], []
    mask = scores > score_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    if len(boxes) == 0: return [], [], []
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1: break
        ovr = overlap_similarity(boxes[i], boxes[order[1:]])
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return keep, boxes, scores

def open_cap(src):
    is_windows = platform.system().lower().startswith("win")
    backend = cv2.CAP_DSHOW if is_windows else cv2.CAP_V4L2
    return cv2.VideoCapture(int(src) if src.isdigit() else src, backend)

# ==========================================
# 2. 雙核 AI 執行緒 (結合已驗證的 PFLD 邏輯)
# ==========================================
shared_frame = None
shared_results = None
lock = threading.Lock()
is_running = True
global_anchors = None
# --- 新增 FPS 變數 ---
inference_fps = 0.0
# 新增：用來在後 10 秒暫停 NPU 推論，釋放資源給 LLM/TTS
pause_inference = False

BLAZE_ID = 20009
PFLD_ID = 20010

def inference_worker(device_group):
    global shared_frame, shared_results, is_running, global_anchors, inference_fps # 加上 inference_fps
    prev_infer_time = time.time() # 初始化時間

    while is_running:
        # 如果進入決策與播放階段，暫停 NPU 推論，省下 CPU 資源
        if pause_inference:
            time.sleep(0.1)
            # 暫停時將 inference_fps 歸零，方便在畫面上辨識
            inference_fps = 0.0 
            prev_infer_time = time.time()
            continue

        with lock:
            img_work = shared_frame.copy() if shared_frame is not None else None
        
        if img_work is None:
            time.sleep(0.01)
            continue
        
        # 紀錄開始推論時間
        start_time = time.time()

        orig_h, orig_w = img_work.shape[:2]

        # --- 階段一：呼叫 BlazeFace 找臉 ---
        img_pad, ratio, (pw, ph) = letterbox(img_work, (128, 128))
        img_565 = cv2.cvtColor(img_pad, cv2.COLOR_BGR2BGR565)
        
        kp.inference.generic_image_inference_send(
            device_group=device_group, 
            generic_inference_input_descriptor=kp.GenericImageInferenceDescriptor(
                model_id=BLAZE_ID, 
                input_node_image_list=[kp.GenericInputNodeImage(image=img_565, image_format=kp.ImageFormat.KP_IMAGE_FORMAT_RGB565)]
            )
        )
        res_blaze = kp.inference.generic_image_inference_receive(device_group=device_group)
        
        out0 = kp.inference.generic_inference_retrieve_float_node(0, res_blaze, kp.ChannelOrdering.KP_CHANNEL_ORDERING_CHW).ndarray
        out1 = kp.inference.generic_inference_retrieve_float_node(1, res_blaze, kp.ChannelOrdering.KP_CHANNEL_ORDERING_CHW).ndarray
        raw_scores, raw_boxes = (out0[0,:,0], out1[0]) if out0.shape[-1] == 1 else (out1[0,:,0], out0[0])
        
        scores = 1 / (1 + np.exp(-np.clip(raw_scores, -15, 15)))
        decoded_boxes = decode_boxes(raw_boxes, global_anchors)
        keep_indices, filtered_boxes, _ = nms(decoded_boxes, scores)

        face_data = None

        if keep_indices:
            ymin, xmin, ymax, xmax = filtered_boxes[keep_indices[0]] * 128
            x1 = max(0, int((xmin - pw) / ratio))
            y1 = max(0, int((ymin - ph) / ratio))
            x2 = min(orig_w, int((xmax - pw) / ratio))
            y2 = min(orig_h, int((ymax - ph) / ratio))

            if x2 > x1 and y2 > y1:
                # --- 階段二：裁切並呼叫 PFLD 抓點 ---
                face_crop = img_work[y1:y2, x1:x2]
                face_112 = cv2.resize(face_crop, (112, 112))
                face_565 = cv2.cvtColor(face_112, cv2.COLOR_BGR2BGR565)

                kp.inference.generic_image_inference_send(
                    device_group=device_group, 
                    generic_inference_input_descriptor=kp.GenericImageInferenceDescriptor(
                        model_id=PFLD_ID, 
                        input_node_image_list=[kp.GenericInputNodeImage(image=face_565, image_format=kp.ImageFormat.KP_IMAGE_FORMAT_RGB565)]
                    )
                )
                res_pfld = kp.inference.generic_image_inference_receive(device_group=device_group)
                
                # [💡 已驗證的 PFLD 節點解析邏輯]
                target_pts = None
                for i in range(res_pfld.header.num_output_node):
                    node_data = kp.inference.generic_inference_retrieve_float_node(
                        i, res_pfld, kp.ChannelOrdering.KP_CHANNEL_ORDERING_CHW
                    ).ndarray
                    if node_data.size == 196:
                        target_pts = node_data
                        break

                if target_pts is not None:
                    flat_pts = target_pts.flatten() 
                    if np.max(flat_pts) > 1.5:  
                        flat_pts = flat_pts / 112.0
                    pts = flat_pts.reshape(98, 2)
                    
                    # 映射回原始畫面
                    abs_pts = []
                    for px, py in pts:
                        abs_x = int(x1 + px * (x2 - x1))
                        abs_y = int(y1 + py * (y2 - y1))
                        abs_pts.append((abs_x, abs_y))

                    # --- 階段三：計算 EAR ---
                    left_ear = calculate_ear(abs_pts[60:68])
                    right_ear = calculate_ear(abs_pts[68:76])
                    avg_ear = (left_ear + right_ear) / 2.0

                    face_data = {
                        'box': (x1, y1, x2, y2),
                        'pts': abs_pts,
                        'ear': avg_ear
                    }

        with lock:
            shared_results = face_data
        # 計算推論 FPS
        curr_infer_time = time.time()
        inference_fps = 1.0 / (curr_infer_time - prev_infer_time)
        prev_infer_time = curr_infer_time

def coach_action(prompt, voice_model):
    """這是一個獨立的執行緒，負責向 Gemma 3 拿文字並用 Piper 播放"""
    print(f"\n[教練思考中...] 傳送 Prompt: {prompt}")
    
    # 1. 呼叫本地端的 llama-server API
    url = "http://127.0.0.1:8080/completion"
    headers = {"Content-Type": "application/json"}
    
    # 【修改重點 1】將 Prompt 包裝成 Gemma 3 Instruct 認識的格式
    formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    
    data = {
        "prompt": formatted_prompt,
        "n_predict": 20,       # 稍微放寬到 20 個 Token
        "temperature": 0.7,
        # 【修改重點 2】移除 "stop": ["\n"]，讓模型完整說完
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        response.raise_for_status()
        
        # 取得文字，並清理掉模型可能輸出的特殊符號
        coach_words = response.json().get('content', '').strip()
        coach_words = coach_words.replace('<end_of_turn>', '').replace('*', '')
        
        # 【修改重點 3】保護機制：萬一模型發神經回傳空字串，給個預設句
        if not coach_words:
            coach_words = "Come on! Keep pushing!"
            
        print(f"[教練說]: {coach_words}")
        
    except Exception as e:
        print(f"[API 錯誤]: {e}")
        coach_words = "Keep going! You are doing great!" 
        
    # 2. 交給 Piper 進行語音合成
    audio_file = "coach_response.wav"
    try:
        with wave.open(audio_file, "wb") as wav_file:
            voice_model.synthesize_wav(coach_words, wav_file)
            
        # 3. 播放聲音
        current_os = platform.system().lower()
        if current_os == "darwin":
            subprocess.run(["afplay", audio_file])
        else:
            subprocess.run(["aplay", audio_file])
            
    except Exception as e:
         print(f"[TTS 錯誤]: {e}")

# ==========================================
# 3. 主程式入口
# ==========================================
def main():
    global shared_frame, shared_results, is_running, global_anchors
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default='/dev/video0') # ⚠️ 請記得確認 Kneo Pi 的攝影機代號
    parser.add_argument('-p', '--port_id', default=0, type=int)
    parser.add_argument('-m', '--model', default='./res/models/KL730/Blazeface_Pfld_combo/blazeface_pfld.nef')
    parser.add_argument('-fw', '--firmware', default='./res/firmware/KL730/kp_firmware.tar')
    args = parser.parse_args()

    global_anchors = np.load('anchors.npy')

    print("[1/3] 初始化 Kneo Pi 雙核引擎...")
    device_group = kp.core.connect_devices(usb_port_ids=[args.port_id])
    kp.core.load_firmware_from_file(device_group=device_group, scpu_fw_path=args.firmware, ncpu_fw_path="")
    kp.core.load_model_from_file(device_group=device_group, file_path=args.model)
    
    threading.Thread(target=inference_worker, args=(device_group,), daemon=True).start()

    print("[1.5/3] 載入 Piper TTS 模型...")
    # ⚠️ 請確保路徑正確指向你下載的 onnx 模型
    voice = PiperVoice.load("./en_US-lessac-medium.onnx")

    print("[2/3] 開啟鏡頭...")
    cap = open_cap(args.src)

    # 狀態紀錄變數
    ear_threshold = 0.20
    closed_frames = 0
    alarm_triggered = False

    print("[3/3] 即時系統啟動！(按 Q 離開)")
    prev_disp_time = time.time() # 初始化顯示時間

    # --- 新增：20秒循環控制變數 ---
    CYCLE_DURATION = 20.0
    SAMPLE_DURATION = 10.0
    cycle_start_time = time.time()
    
    # 統計用變數
    ear_counts = {'low': 0, 'mid': 0, 'high': 0}
    total_samples = 0
    decision_made = False
    decision_text = "Waiting..."
    current_phase = "SAMPLING"

    # 定義你的 Prompt (準備給下一步 Gemma 3 使用)
    system_prompts = {
        'low': "You're a fitness trainer. Briefly shout one powerful sentence(under 10 words) to motivate your trainee who feels exhausted.",
        'mid': "You're a fitness trainer. Briefly say a sentence(under 10 words) to motivate your trainee who is hitting the sweet spot.",
        'high': "You're a fitness trainer. Briefly say a sentence(under 10 words) to motivate your trainee who feels no challenge."
    }

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 計算顯示 FPS
        curr_disp_time = time.time()
        display_fps = 1.0 / (curr_disp_time - prev_disp_time)
        prev_disp_time = curr_disp_time
        
        with lock:
            shared_frame = frame.copy()
            data = shared_results

        # --- 時間與循環邏輯 ---
        elapsed = time.time() - cycle_start_time
        countdown = CYCLE_DURATION - elapsed

        if elapsed < SAMPLE_DURATION:
            # 【前 10 秒：取樣階段】
            current_phase = "SAMPLING"
            if data:
                ear = data['ear']
                if ear < 0.20:
                    ear_counts['low'] += 1
                elif 0.20 <= ear <= 0.25:
                    ear_counts['mid'] += 1
                else:
                    ear_counts['high'] += 1
                total_samples += 1
                
        elif SAMPLE_DURATION <= elapsed < CYCLE_DURATION:
            # 【後 10 秒：決策與激勵階段】
            current_phase = "DECISION & TTS"
            
            if not decision_made:
                global pause_inference
                pause_inference = True # 暫停 NPU 推論
                
                # 計算百分比並決策
                if total_samples > 0:
                    p_low = ear_counts['low'] / total_samples
                    p_mid = ear_counts['mid'] / total_samples
                    
                    if p_low >= 0.25:
                        decision_text = "EXHAUSTED (<0.20)"
                        selected_prompt = system_prompts['low']
                    elif p_mid >= 0.40:
                        decision_text = "SWEET SPOT (0.20-0.25)"
                        selected_prompt = system_prompts['mid']
                    else:
                        decision_text = "NO CHALLENGE (>0.25)"
                        selected_prompt = system_prompts['high']
                else:
                    decision_text = "NO FACE DETECTED"
                    selected_prompt = None
                
                decision_made = True
                
                # 這裡未來會啟動一個獨立執行緒去呼叫 llama-server 和 piper
                # threading.Thread(target=call_llm_and_tts, args=(selected_prompt,)).start()
                if selected_prompt:
                    threading.Thread(target=coach_action, args=(selected_prompt, voice), daemon=True).start()

        else:
            # 【滿 20 秒：重置循環】
            cycle_start_time = time.time()
            ear_counts = {'low': 0, 'mid': 0, 'high': 0}
            total_samples = 0
            decision_made = False
            decision_text = "Waiting..."
            pause_inference = False # 恢復 NPU 推論


        # --- 畫面 UI 繪製 ---
        if data and not pause_inference:
            # 只有在取樣階段才畫臉部網格與即時 EAR
            x1, y1, x2, y2 = data['box']
            pts = data['pts']
            #cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
            #for i in range(60, 76):
                #cv2.circle(frame, pts[i], 2, (0, 255, 255), -1)
            #cv2.putText(frame, f"EAR: {data['ear']:.3f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 顯示系統狀態 (左上角)
        cv2.putText(frame, f"Phase: {current_phase}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        cv2.putText(frame, f"Timer: {countdown:.1f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        cv2.putText(frame, f"Disp FPS🥟: {display_fps:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Infer FPS: {inference_fps:.1f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 顯示統計數據 (左下角)
        if total_samples > 0:
            p_low = (ear_counts['low'] / total_samples) * 100
            p_mid = (ear_counts['mid'] / total_samples) * 100
            p_high = (ear_counts['high'] / total_samples) * 100
            cv2.putText(frame, f"Stats - Low:{p_low:.0f}% Mid:{p_mid:.0f}% High:{p_high:.0f}%", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 顯示最終決策 (正中央偏下)
        if decision_made:
            text_size = cv2.getTextSize(f"Result: {decision_text}", cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            cv2.putText(frame, f"Result: {decision_text}", (text_x, frame.shape[0] - 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        cv2.imshow("Kneo Pi - AI Coach", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            is_running = False
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
