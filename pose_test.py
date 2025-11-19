import cv2
import mediapipe as mp
import math 
import time
import requests 

# --- ネットワークカメラの設定 ---
THERMAL_SERVER_URL = "http://pigeon02.local:7878"

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2, # 高精度モード
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils

# カメラ設定 (MacのWebカメラ)
cap = cv2.VideoCapture(1) 

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("--- ECO Advisor System ---")
print(f"Connecting to thermal camera at: {THERMAL_SERVER_URL}")

# --- 初期設定 ---
# ★★★ 基準値を変更したい場合はここを編集 ★★★
user_comfortable_room_temp = 24.0
user_baseline_skin_temp = 36.5

room_temp = user_comfortable_room_temp 
skin_temp = user_baseline_skin_temp 
skin_temp_baseline = user_baseline_skin_temp

# 判定閾値
ROOM_TEMP_HOT = user_comfortable_room_temp + 2.0
ROOM_TEMP_COLD = user_comfortable_room_temp - 2.0
SKIN_TEMP_HOT = skin_temp_baseline + 0.5 
SKIN_TEMP_COLD = skin_temp_baseline - 0.5

# --- 関数定義 ---

def get_thermal_data():
    """
    pigeon02.local から温度データを取得し、異常値を除去した上で最高温度(体温)を返す
    """
    try:
        response = requests.get(THERMAL_SERVER_URL, timeout=0.5)
        
        if response.status_code == 200:
            rows = response.text.strip().split('\n')
            max_temp_raw = -99999
            
            for row in rows:
                cols = row.split(',')
                for val in cols:
                    try:
                        t = int(val) # データは「温度 × 100」の整数値
                        
                        # ★★★ 異常値フィルター ★★★
                        # 0℃(0) 〜 60℃(6000) の範囲外は「ノイズ」として無視する
                        if 0 <= t <= 6000: 
                            if t > max_temp_raw:
                                max_temp_raw = t
                                
                    except ValueError:
                        continue
            
            if max_temp_raw > -99999:
                return float(max_temp_raw) / 100.0
            
    except Exception as e:
        pass
    
    return None

def get_distance(p1, p2, image_width, image_height):
    if not (p1 and p2): return float('inf') 
    p1_x = int(p1.x * image_width); p1_y = int(p1.y * image_height)
    p2_x = int(p2.x * image_width); p2_y = int(p2.y * image_height)
    return math.hypot(p1_x - p2_x, p1_y - p2_y)

def calculate_angle(a, b, c, image_width, image_height):
    if not (a and b and c): return 181 
    a_x = int(a.x * image_width); a_y = int(a.y * image_height)
    b_x = int(b.x * image_width); b_y = int(b.y * image_height)
    c_x = int(c.x * image_width); c_y = int(c.y * image_height)
    ba_x = a_x - b_x; ba_y = a_y - b_y
    bc_x = c_x - b_x; bc_y = c_y - b_y
    angle_ba = math.atan2(ba_y, ba_x)
    angle_bc = math.atan2(bc_y, bc_x)
    angle_deg = abs(math.degrees(angle_bc - angle_ba))
    if angle_deg > 180: angle_deg = 360 - angle_deg
    return angle_deg

# --- 変数初期化 ---
prev_wrist_x_R, prev_wrist_y_R = 0, 0
prev_wrist_x_L, prev_wrist_y_L = 0, 0
wrist_speed_R = 0; wrist_speed_L = 0

fanning_start_time = None; fanning_duration = 1.0
crossing_arms_start_time = None; crossing_arms_duration = 2.0
warming_hands_start_time = None; warming_hands_duration = 2.0
wiping_sweat_start_time = None; wiping_sweat_duration = 1.5

last_thermal_update = 0
thermal_update_interval = 1.0 # 1秒ごとに温度更新

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    image = cv2.flip(image, 1)
    h, w, _ = image.shape
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # --- 温度データの更新 ---
    if time.time() - last_thermal_update > thermal_update_interval:
        new_skin_temp = get_thermal_data()
        if new_skin_temp is not None:
            skin_temp = new_skin_temp 
        last_thermal_update = time.time()

    current_action_status = "---" 
    action_status_display = "---" 
    advisor_status = "---" 

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        def get_lm(enum): return landmarks[enum.value] if landmarks[enum.value].visibility > 0.6 else None
        
        nose = get_lm(mp_pose.PoseLandmark.NOSE)
        r_sho = get_lm(mp_pose.PoseLandmark.RIGHT_SHOULDER); l_sho = get_lm(mp_pose.PoseLandmark.LEFT_SHOULDER)
        r_elb = get_lm(mp_pose.PoseLandmark.RIGHT_ELBOW); l_elb = get_lm(mp_pose.PoseLandmark.LEFT_ELBOW)
        r_wri = get_lm(mp_pose.PoseLandmark.RIGHT_WRIST); l_wri = get_lm(mp_pose.PoseLandmark.LEFT_WRIST)
        l_eye = get_lm(mp_pose.PoseLandmark.LEFT_EYE); r_eye = get_lm(mp_pose.PoseLandmark.RIGHT_EYE)

        # 速度計算
        if r_wri:
            curr_Rx, curr_Ry = int(r_wri.x * w), int(r_wri.y * h)
            if prev_wrist_x_R != 0: wrist_speed_R = math.hypot(curr_Rx - prev_wrist_x_R, curr_Ry - prev_wrist_y_R)
            prev_wrist_x_R, prev_wrist_y_R = curr_Rx, curr_Ry
        if l_wri:
            curr_Lx, curr_Ly = int(l_wri.x * w), int(l_wri.y * h)
            if prev_wrist_x_L != 0: wrist_speed_L = math.hypot(curr_Lx - prev_wrist_x_L, curr_Ly - prev_wrist_y_L)
            prev_wrist_x_L, prev_wrist_y_L = curr_Lx, curr_Ly

        # --- 1. 腕組み ---
        if all([r_sho, l_sho, r_elb, l_elb, r_wri, l_wri]):
            sw = get_distance(r_sho, l_sho, w, h)
            if get_distance(r_wri, l_elb, w, h) < sw * 0.55 and get_distance(l_wri, r_elb, w, h) < sw * 0.55:
                current_action_status = "Crossing Arms (COLD)"
                crossing_arms_start_time = time.time()
        
        if crossing_arms_start_time and time.time() - crossing_arms_start_time < crossing_arms_duration:
            current_action_status = "Crossing Arms (COLD)"
        else: crossing_arms_start_time = None

        # --- 2. 手を温める ---
        if current_action_status == "---" and all([nose, r_wri, l_wri]):
            dist_R = get_distance(nose, r_wri, w, h); dist_L = get_distance(nose, l_wri, w, h)
            if dist_R < 300 and dist_L < 300 and (r_wri.z < nose.z) and (l_wri.z < nose.z):
                current_action_status = "Warming Hands (COLD)"
                warming_hands_start_time = time.time()

        if warming_hands_start_time and time.time() - warming_hands_start_time < warming_hands_duration:
            current_action_status = "Warming Hands (COLD)"
        else: warming_hands_start_time = None

        # --- 3. 汗を拭う ---
        if current_action_status == "---":
            is_wipe = False
            if all([r_wri, l_eye, r_eye]):
                y_chk = r_wri.y * h < (l_eye.y + r_eye.y)/2 * h
                x_chk = min(l_eye.x, r_eye.x) * w < r_wri.x * w < max(l_eye.x, r_eye.x) * w
                if y_chk and x_chk and wrist_speed_R > 10: is_wipe = True
            if all([l_wri, l_eye, r_eye]) and not is_wipe:
                y_chk = l_wri.y * h < (l_eye.y + r_eye.y)/2 * h
                x_chk = min(l_eye.x, r_eye.x) * w < l_wri.x * w < max(l_eye.x, r_eye.x) * w
                if y_chk and x_chk and wrist_speed_L > 10: is_wipe = True
            
            if is_wipe:
                current_action_status = "Wiping Sweat (HOT)"
                wiping_sweat_start_time = time.time()
        
        if wiping_sweat_start_time and time.time() - wiping_sweat_start_time < wiping_sweat_duration:
            current_action_status = "Wiping Sweat (HOT)"
        else: wiping_sweat_start_time = None

        # --- 4. あおぐ ---
        if current_action_status == "---":
            is_fan = False
            if all([nose, r_sho, r_elb, r_wri]):
                if calculate_angle(r_sho, r_elb, r_wri, w, h) < 140 and wrist_speed_R > 35 and get_distance(nose, r_wri, w, h) < 850: is_fan = True
            if all([nose, l_sho, l_elb, l_wri]) and not is_fan:
                if calculate_angle(l_sho, l_elb, l_wri, w, h) < 140 and wrist_speed_L > 35 and get_distance(nose, l_wri, w, h) < 850: is_fan = True

            if is_fan:
                current_action_status = "Fanning (HOT)"
                fanning_start_time = time.time()

        if fanning_start_time and time.time() - fanning_start_time < fanning_duration:
            current_action_status = "Fanning (HOT)"
        else: fanning_start_time = None

        # 他のタイマーリセット
        if current_action_status != "Fanning (HOT)": fanning_start_time = None
        if current_action_status != "Warming Hands (COLD)": warming_hands_start_time = None
        if current_action_status != "Crossing Arms (COLD)": crossing_arms_start_time = None
        if current_action_status != "Wiping Sweat (HOT)": wiping_sweat_start_time = None

        action_status_display = current_action_status
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # --- ECO Advisor 判定 ---
    if room_temp > ROOM_TEMP_HOT and skin_temp > SKIN_TEMP_HOT:
        if action_status_display in ["Fanning (HOT)", "Wiping Sweat (HOT)"]:
            advisor_status = "ALERT: Turn down AC (High Confidence)"
        elif action_status_display in ["Crossing Arms (COLD)", "Warming Hands (COLD)"]:
            advisor_status = "WARNING: Contradiction!"
        else:
            advisor_status = "User seems hot (Low Confidence)"
    elif room_temp < ROOM_TEMP_COLD and skin_temp < SKIN_TEMP_COLD:
        if action_status_display in ["Crossing Arms (COLD)", "Warming Hands (COLD)"]:
            advisor_status = "ALERT: Turn on Heater! (High Confidence)"
        elif action_status_display in ["Fanning (HOT)", "Wiping Sweat (HOT)"]:
             advisor_status = "WARNING: Contradiction!"
        else:
            advisor_status = "User seems cold (Low Confidence)"
    else:
        if action_status_display in ["Fanning (HOT)", "Wiping Sweat (HOT)"]:
            advisor_status = "User seems hot (Low Confidence)"
        elif action_status_display in ["Crossing Arms (COLD)", "Warming Hands (COLD)"]:
            advisor_status = "User seems cold (Low Confidence)"
        else:
            advisor_status = "OK: Comfortable"

    # 表示
    cv2.putText(image, f'Room Temp: {room_temp:.1f} C', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(image, f'Skin Temp (Net): {skin_temp:.1f} C', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(image, f'Action: {action_status_display}', (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image, f'Advisor: {advisor_status}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('ECO Advisor System', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
pose.close()
