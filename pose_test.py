import cv2
import mediapipe as mp
import math 
import time 

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils

# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
# カメラ番号を 1 に設定
cap = cv2.VideoCapture(1) 
# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("--- ECO Advisor Personalization Setup ---")
# ユーザーの快適な基準を入力してもらう
try:
    user_comfortable_room_temp = float(input("Please enter your 'comfortable' room temperature (e.g., 25.0): "))
    user_baseline_skin_temp = float(input("Please enter your 'baseline' (normal) skin temperature (e.g., 36.5): "))
except ValueError:
    print("Invalid input. Using default values (25.0C, 36.5C).")
    user_comfortable_room_temp = 25.0
    user_baseline_skin_temp = 36.5

print("-------------------------------------------")
print(f"Setup complete: Comfortable Room Temp {user_comfortable_room_temp}C, Baseline Skin Temp {user_baseline_skin_temp}C")
print("Camera started. Press 'q' in the window to quit.")
print("--- Key Controls ---")
print("r / f : Adjust virtual 'Room Temp' Up / Down")
print("t / g : Adjust virtual 'Skin Temp' Up / Down")

# --- 仮想センサーの変数を、入力された値で初期化 ---
room_temp = user_comfortable_room_temp  # 仮想の室温 (℃)
skin_temp = user_baseline_skin_temp  # 仮想の皮膚温度 (℃)
skin_temp_baseline = user_baseline_skin_temp # 平常時の皮膚温度として保存

# --- Advisorの判定閾値を、入力された値に基づいて設定 ---
ROOM_TEMP_HOT = user_comfortable_room_temp + 2.0  # 快適＋2℃で「暑い」
ROOM_TEMP_COLD = user_comfortable_room_temp - 2.0 # 快適－2℃で「寒い」
SKIN_TEMP_HOT = skin_temp_baseline + 0.5 
SKIN_TEMP_COLD = skin_temp_baseline - 0.5

# 2点間の距離を計算する関数
def get_distance(p1, p2, image_width, image_height):
    if not (p1 and p2): 
        return float('inf') 
    p1_x = int(p1.x * image_width)
    p1_y = int(p1.y * image_height)
    p2_x = int(p2.x * image_width)
    p2_y = int(p2.y * image_height)
    return math.hypot(p1_x - p2_x, p1_y - p2_y)

# 3点の座標から角度を計算する関数
def calculate_angle(a, b, c, image_width, image_height):
    if not (a and b and c): 
        return 181 
    a_x = int(a.x * image_width)
    a_y = int(a.y * image_height)
    b_x = int(b.x * image_width)
    b_y = int(b.y * image_height)
    c_x = int(c.x * image_width)
    c_y = int(c.y * image_height)
    ba_x = a_x - b_x
    ba_y = a_y - b_y
    bc_x = c_x - b_x
    bc_y = c_y - b_y
    angle_ba = math.atan2(ba_y, ba_x)
    angle_bc = math.atan2(bc_y, bc_x)
    angle_rad = angle_bc - angle_ba
    angle_deg = math.degrees(angle_rad)
    angle_deg = abs(angle_deg)
    if angle_deg > 180:
        angle_deg = 360 - angle_deg
    return angle_deg

# --- ループの外で変数を初期化 ---
prev_wrist_x_R, prev_wrist_y_R = 0, 0 # 右手首
prev_wrist_x_L, prev_wrist_y_L = 0, 0 # 左手首
wrist_speed_R = 0
wrist_speed_L = 0

fanning_start_time = None
fanning_duration = 1.0  # 「あおぎ」状態を1.0秒維持

crossing_arms_start_time = None
crossing_arms_duration = 2.0 # 「腕組み」状態を2.0秒維持

warming_hands_start_time = None
warming_hands_duration = 2.0 # 「手を温める」状態を2.0秒維持

wiping_sweat_start_time = None
wiping_sweat_duration = 1.5 # 「汗を拭う」状態を1.5秒維持

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("ERROR: Could not read frame.")
        break

    image = cv2.flip(image, 1) # 最初に反転
    image_height, image_width, _ = image.shape
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    current_action_status = "---" 
    action_status_display = "---" 
    advisor_status = "---" 

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        try:
            # 各ランドマークがNoneでないことを確認するヘルパー関数
            def get_landmark(landmark_enum):
                lm = landmarks[landmark_enum.value]
                return lm if lm.visibility > 0.6 else None 
            
            # --- ランドマークの取得 ---
            nose = get_landmark(mp_pose.PoseLandmark.NOSE)
            right_shoulder = get_landmark(mp_pose.PoseLandmark.RIGHT_SHOULDER)
            left_shoulder = get_landmark(mp_pose.PoseLandmark.LEFT_SHOULDER)
            right_elbow = get_landmark(mp_pose.PoseLandmark.RIGHT_ELBOW)
            left_elbow = get_landmark(mp_pose.PoseLandmark.LEFT_ELBOW)
            right_wrist = get_landmark(mp_pose.PoseLandmark.RIGHT_WRIST)
            left_wrist = get_landmark(mp_pose.PoseLandmark.LEFT_WRIST)
            left_eye = get_landmark(mp_pose.PoseLandmark.LEFT_EYE)
            right_eye = get_landmark(mp_pose.PoseLandmark.RIGHT_EYE)

            # --- 1. 「腕組み」検出 (最優先) ---
            is_crossing_arms = False
            if all([right_shoulder, left_shoulder, right_elbow, left_elbow, right_wrist, left_wrist]):
                shoulder_width = get_distance(right_shoulder, left_shoulder, image_width, image_height)
                dist_R_wrist_L_elbow = get_distance(right_wrist, left_elbow, image_width, image_height)
                dist_L_wrist_R_elbow = get_distance(left_wrist, right_elbow, image_width, image_height)
                
                threshold_cold = shoulder_width * 0.55 
                
                is_crossing_arms = (dist_R_wrist_L_elbow < threshold_cold) and \
                                   (dist_L_wrist_R_elbow < threshold_cold)
            
            if is_crossing_arms:
                current_action_status = "Crossing Arms (COLD)"
                crossing_arms_start_time = time.time() 
            elif (crossing_arms_start_time is not None) and \
                 (time.time() - crossing_arms_start_time < crossing_arms_duration):
                current_action_status = "Crossing Arms (COLD)"
            else:
                crossing_arms_start_time = None
            
            # --- 2. 「手を温める」検出 (優先度2) ---
            if current_action_status == "---":
                is_warming_hands = False
                dist_R_wrist_nose = float('inf') 
                dist_L_wrist_nose = float('inf') 
                
                if all([nose, right_wrist, left_wrist]):
                    dist_R_wrist_nose = get_distance(nose, right_wrist, image_width, image_height)
                    dist_L_wrist_nose = get_distance(nose, left_wrist, image_width, image_height)
                    
                    threshold_warm_hands_dist = 300 
                    is_in_front = (right_wrist.z < nose.z) and (left_wrist.z < nose.z)
                    
                    is_warming_hands = (dist_R_wrist_nose < threshold_warm_hands_dist) and \
                                       (dist_L_wrist_nose < threshold_warm_hands_dist) and \
                                       is_in_front 

                if is_warming_hands:
                    current_action_status = "Warming Hands (COLD)"
                    warming_hands_start_time = time.time()
                elif (warming_hands_start_time is not None) and \
                     (time.time() - warming_hands_start_time < warming_hands_duration):
                    current_action_status = "Warming Hands (COLD)"
                else:
                    warming_hands_start_time = None
            
            # --- 3. 「汗を拭う」検出 (優先度3、左右の手に対応) ---
            if current_action_status == "---":
                is_wiping_sweat = False
                
                # --- 右手首の速度を計算 ---
                current_wrist_x_R = 0
                current_wrist_y_R = 0
                if all([right_wrist]):
                    current_wrist_x_R = int(right_wrist.x * image_width)
                    current_wrist_y_R = int(right_wrist.y * image_height)
                if prev_wrist_x_R == 0 and prev_wrist_y_R == 0: 
                    wrist_speed_R = 0
                else:
                    wrist_speed_R = math.hypot(current_wrist_x_R - prev_wrist_x_R, current_wrist_y_R - prev_wrist_y_R)
                prev_wrist_x_R, prev_wrist_y_R = current_wrist_x_R, current_wrist_y_R
                
                # --- 左手首の速度を計算 ---
                current_wrist_x_L = 0
                current_wrist_y_L = 0
                if all([left_wrist]):
                    current_wrist_x_L = int(left_wrist.x * image_width)
                    current_wrist_y_L = int(left_wrist.y * image_height)
                if prev_wrist_x_L == 0 and prev_wrist_y_L == 0: 
                    wrist_speed_L = 0
                else:
                    wrist_speed_L = math.hypot(current_wrist_x_L - prev_wrist_x_L, current_wrist_y_L - prev_wrist_y_L)
                prev_wrist_x_L, prev_wrist_y_L = current_wrist_x_L, current_wrist_y_L

                threshold_speed_sweat = 10
                is_wiping_sweat_R = False
                is_wiping_sweat_L = False

                # --- 右手で汗を拭う動作のチェック ---
                if all([right_wrist, left_eye, right_eye]):
                    wrist_y_px = int(right_wrist.y * image_height)
                    eye_l_y_px = int(left_eye.y * image_height)
                    eye_r_y_px = int(right_eye.y * image_height)
                    eye_y_avg = (eye_l_y_px + eye_r_y_px) / 2
                    
                    wrist_x_px = int(right_wrist.x * image_width)
                    eye_l_x_px = int(left_eye.x * image_width)
                    eye_r_x_px = int(right_eye.x * image_width)

                    is_above_eyes = (wrist_y_px < eye_y_avg)
                    is_between_eyes = (min(eye_l_x_px, eye_r_x_px) < wrist_x_px < max(eye_l_x_px, eye_r_x_px))
                    is_moving = (wrist_speed_R > threshold_speed_sweat) # 右手の速度
                    is_wiping_sweat_R = is_above_eyes and is_between_eyes and is_moving

                # --- 左手で汗を拭う動作のチェック ---
                if all([left_wrist, left_eye, right_eye]):
                    wrist_y_px = int(left_wrist.y * image_height) # 左手首
                    eye_l_y_px = int(left_eye.y * image_height)
                    eye_r_y_px = int(right_eye.y * image_height)
                    eye_y_avg = (eye_l_y_px + eye_r_y_px) / 2
                    
                    wrist_x_px = int(left_wrist.x * image_width) # 左手首
                    eye_l_x_px = int(left_eye.x * image_width)
                    eye_r_x_px = int(right_eye.x * image_width)

                    is_above_eyes = (wrist_y_px < eye_y_avg)
                    is_between_eyes = (min(eye_l_x_px, eye_r_x_px) < wrist_x_px < max(eye_l_x_px, eye_r_x_px))
                    is_moving = (wrist_speed_L > threshold_speed_sweat) # 左手の速度
                    is_wiping_sweat_L = is_above_eyes and is_between_eyes and is_moving
                
                # --- 最終判定 (どちらかの手でOK) ---
                is_wiping_sweat = is_wiping_sweat_R or is_wiping_sweat_L
                
                if is_wiping_sweat:
                    current_action_status = "Wiping Sweat (HOT)"
                    wiping_sweat_start_time = time.time()
                elif (wiping_sweat_start_time is not None) and \
                     (time.time() - wiping_sweat_start_time < wiping_sweat_duration):
                    current_action_status = "Wiping Sweat (HOT)"
                else:
                    wiping_sweat_start_time = None

            # --- 4. ★★★「あおぐ」検出 (優先度4、左右の手に対応) ★★★ ---
            if current_action_status == "---":
                is_fanning_R = False
                is_fanning_L = False
                
                threshold_angle = 140  
                threshold_speed_fanning = 35 # あおぐ動作は速度35以上
                threshold_dist = 850   
                
                # --- 右手であおぐ動作のチェック ---
                if all([nose, right_shoulder, right_elbow, right_wrist]): 
                    elbow_angle_R = calculate_angle(right_shoulder, right_elbow, right_wrist, image_width, image_height)
                    dist_wrist_nose_f_R = get_distance(nose, right_wrist, image_width, image_height)
                    
                    # wrist_speed_R は「汗を拭う」セクションで計算済み
                    is_fanning_R = (elbow_angle_R < threshold_angle) and \
                                   (wrist_speed_R > threshold_speed_fanning) and \
                                   (dist_wrist_nose_f_R < threshold_dist)
                
                # --- 左手であおぐ動作のチェック ---
                if all([nose, left_shoulder, left_elbow, left_wrist]): 
                    elbow_angle_L = calculate_angle(left_shoulder, left_elbow, left_wrist, image_width, image_height)
                    dist_wrist_nose_f_L = get_distance(nose, left_wrist, image_width, image_height)
                    
                    # wrist_speed_L は「汗を拭う」セクションで計算済み
                    is_fanning_L = (elbow_angle_L < threshold_angle) and \
                                   (wrist_speed_L > threshold_speed_fanning) and \
                                   (dist_wrist_nose_f_L < threshold_dist)
                
                # --- 最終判定 (どちらかの手でOK) ---
                is_fanning = is_fanning_R or is_fanning_L
                
                if is_fanning:
                    current_action_status = "Fanning (HOT)"
                    fanning_start_time = time.time()
                elif (fanning_start_time is not None) and (time.time() - fanning_start_time < fanning_duration):
                    current_action_status = "Fanning (HOT)"
                else:
                    fanning_start_time = None
            
            # タイマーのリセット処理
            if current_action_status != "Fanning (HOT)":
                fanning_start_time = None
            if current_action_status != "Warming Hands (COLD)":
                warming_hands_start_time = None
            if current_action_status != "Crossing Arms (COLD)": 
                crossing_arms_start_time = None
            if current_action_status != "Wiping Sweat (HOT)": 
                wiping_sweat_start_time = None

            action_status_display = current_action_status 

        except Exception as e:
            action_status_display = "Cannot detect points"
            # print(f"ERROR: {e}") 

        # ポーズランドマークを描画
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )
    
    # ★★★ 2. 「ECO Advisor」の最終判定ロジック ★★★
    
    # [判定A] 暑い状況 (センサーが「暑い」)
    if (room_temp > ROOM_TEMP_HOT) and (skin_temp > SKIN_TEMP_HOT):
        if action_status_display == "Fanning (HOT)" or action_status_display == "Wiping Sweat (HOT)":
            advisor_status = "ALERT: Turn down AC (High Confidence)"
        elif action_status_display == "Crossing Arms (COLD)" or action_status_display == "Warming Hands (COLD)":
            advisor_status = "WARNING: Contradiction! (Data Mismatch)"
        else: # (action_status_display == "---")
            advisor_status = "User seems hot (Low Confidence)"
            
    # [判定B] 寒い状況 (センサーが「寒い」)
    elif (room_temp < ROOM_TEMP_COLD) and (skin_temp < SKIN_TEMP_COLD):
        if action_status_display == "Crossing Arms (COLD)" or action_status_display == "Warming Hands (COLD)":
            advisor_status = "ALERT: Turn on Heater! (High Confidence)"
        elif action_status_display == "Fanning (HOT)" or action_status_display == "Wiping Sweat (HOT)":
            advisor_status = "WARNING: Contradiction! (Data MMismatch)"
        else: # (action_status_display == "---")
            advisor_status = "User seems cold (Low Confidence)"
            
    # [判定C] 快適な状況 (センサーは「快適」)
    else:
        if action_status_display == "Fanning (HOT)" or action_status_display == "Wiping Sweat (HOT)":
            advisor_status = "User seems hot (Low Confidence)" 
        elif action_status_display == "Crossing Arms (COLD)" or action_status_display == "Warming Hands (COLD)":
            advisor_status = "User seems cold (Low Confidence)" 
        else: # (action_status_display == "---")
            advisor_status = "OK: Comfortable"

    # --- 画面表示 ---
    
    cv2.putText(image, f'Room Temp (r/f): {room_temp:.1f} C', (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(image, f'Skin Temp (t/g): {skin_temp:.1f} C', (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    cv2.putText(image, f'Action: {action_status_display}', (10, 190), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.putText(image, f'Advisor: {advisor_status}', (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    # デバッグ情報を削除（必要な場合は前のコードを参照）

    # --- キー操作の受付 ---
    key = cv2.waitKey(5) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('r'): # Room Temp Up
        room_temp += 0.5
    elif key == ord('f'): # Room Temp Down
        room_temp -= 0.5
    elif key == ord('t'): # Skin Temp Up
        skin_temp += 0.1
    elif key == ord('g'): # Skin Temp Down
        skin_temp -= 0.1

    cv2.imshow('ECO Advisor System Prototype', image)

cap.release()
cv2.destroyAllWindows()
pose.close()

print("Program finished.")
