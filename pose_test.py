import cv2
import mediapipe as mp
import math 
import time
import requests
import serial 

# --- 設定エリア ---
THERMAL_SERVER_URL = "http://pigeon02.local:7878"
ARDUINO_PORT = "/dev/tty.usbmodem1101" 
BAUD_RATE = 9600 

# --- セットアップ ---
print("--- ECO Advisor Personalization Setup ---")
try:
    print("Default: Room=24.0, Skin=33.5")
    user_comfortable_room_temp = float(input("Enter comfortable room temp: ") or 24.0)
    user_baseline_skin_temp = float(input("Enter baseline skin temp: ") or 33.5) 
except:
    user_comfortable_room_temp = 24.0
    user_baseline_skin_temp = 33.5

# --- 変数初期化 ---
room_temp = user_comfortable_room_temp 
room_humidity = 50.0
skin_temp = user_baseline_skin_temp 
arduino_connected = False

# --- 計算ロジック ---
def calculate_ptc(temp, hum):
    try:
        di = 0.81 * temp + 0.01 * hum * (0.99 * temp - 14.3) + 46.3
        score = (di - 75.0) / 5.0 
        return max(-3.0, min(3.0, score)) 
    except: return 0.0

def calculate_itc(current_skin, base_skin):
    diff = current_skin - base_skin
    score = diff * 1.5 
    return max(-3.0, min(3.0, score))

# --- 描画ヘルパー関数 (縁取り付きテキスト) ---
def draw_text(img, text, x, y, color=(255, 255, 255), scale=0.7):
    # 黒い縁取り (太さ4)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    # メインの色 (太さ2)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)

# --- デバイス設定 ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, smooth_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(1) 
if not cap.isOpened(): exit()

ser = None
try:
    ser = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=0.1)
    arduino_connected = True
except: print("Arduino not found.")

# --- センサー関数 ---
def get_thermal_data():
    try:
        response = requests.get(THERMAL_SERVER_URL, timeout=0.5)
        if response.status_code == 200:
            rows = response.text.strip().split('\n')
            max_temp = -999
            for row in rows:
                for val in row.split(','):
                    try:
                        t = int(val)
                        if 0 <= t <= 6000 and t > max_temp: max_temp = t
                    except: continue
            if max_temp > -999: return float(max_temp) / 100.0
    except: pass
    return None

def read_arduino():
    global room_temp, room_humidity, arduino_connected
    if ser and ser.in_waiting > 0:
        try:
            while ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if "Temp =" in line: room_temp = float(line.split()[2]); arduino_connected = True
                elif "Humidity =" in line: room_humidity = float(line.split()[2])
        except: pass

def get_dist(p1, p2, w, h):
    if not (p1 and p2): return 9999
    return math.hypot(p1.x*w - p2.x*w, p1.y*h - p2.y*h)

def get_angle(a, b, c, w, h):
    if not (a and b and c): return 180
    deg = abs(math.degrees(math.atan2(c.y*h-b.y*h, c.x*w-b.x*w) - math.atan2(a.y*h-b.y*h, a.x*w-b.x*w)))
    return 360-deg if deg>180 else deg

# --- ループ変数 ---
prev_R, prev_L = (0,0), (0,0)
spd_R, spd_L = 0, 0
timers = {"fan":None, "arm":None, "warm":None, "wipe":None}
durs = {"fan":1.0, "arm":2.0, "warm":2.0, "wipe":1.5}
last_therm = 0

while cap.isOpened():
    success, img = cap.read()
    if not success: break
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    
    # センサー更新
    read_arduino()
    if time.time() - last_therm > 1.0:
        val = get_thermal_data()
        if val: skin_temp = val
        last_therm = time.time()

    # スコア計算
    ptc = calculate_ptc(room_temp, room_humidity)
    itc = calculate_itc(skin_temp, user_baseline_skin_temp)

    # 動作検出
    act_status = "---"
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        def get(e): return lm[e.value] if lm[e.value].visibility > 0.6 else None
        
        nose = get(mp_pose.PoseLandmark.NOSE)
        rs, ls = get(mp_pose.PoseLandmark.RIGHT_SHOULDER), get(mp_pose.PoseLandmark.LEFT_SHOULDER)
        re, le = get(mp_pose.PoseLandmark.RIGHT_ELBOW), get(mp_pose.PoseLandmark.LEFT_ELBOW)
        rw, lw = get(mp_pose.PoseLandmark.RIGHT_WRIST), get(mp_pose.PoseLandmark.LEFT_WRIST)
        leye, reye = get(mp_pose.PoseLandmark.LEFT_EYE), get(mp_pose.PoseLandmark.RIGHT_EYE)

        # 速度
        if rw:
            curr_R = (int(rw.x*w), int(rw.y*h))
            if prev_R != (0,0): spd_R = math.hypot(curr_R[0]-prev_R[0], curr_R[1]-prev_R[1])
            prev_R = curr_R
        if lw:
            curr_L = (int(lw.x*w), int(lw.y*h))
            if prev_L != (0,0): spd_L = math.hypot(curr_L[0]-prev_L[0], curr_L[1]-prev_L[1])
            prev_L = curr_L

        now = time.time()
        # 1. 腕組み
        if all([rs, ls, re, le, rw, lw]):
            sw = get_dist(rs, ls, w, h)
            if get_dist(rw, le, w, h) < sw*0.55 and get_dist(lw, re, w, h) < sw*0.55:
                act_status = "Crossing Arms (COLD)"; timers["arm"] = now
        # 2. 手を温める
        if act_status == "---" and all([nose, rw, lw]):
            if get_dist(nose, rw, w, h) < 300 and get_dist(nose, lw, w, h) < 300 and rw.z < nose.z and lw.z < nose.z:
                act_status = "Warming Hands (COLD)"; timers["warm"] = now
        # 3. 汗拭き
        if act_status == "---":
            is_wipe = False
            if all([rw, leye, reye]) and rw.y < (leye.y+reye.y)/2 and spd_R > 10: is_wipe = True
            if all([lw, leye, reye]) and lw.y < (leye.y+reye.y)/2 and spd_L > 10: is_wipe = True
            if is_wipe: act_status = "Wiping Sweat (HOT)"; timers["wipe"] = now
        # 4. あおぐ
        if act_status == "---":
            is_fan = False
            if all([nose, rs, re, rw]) and get_angle(rs, re, rw, w, h)<140 and spd_R>35 and get_dist(nose, rw, w, h)<850: is_fan = True
            if all([nose, ls, le, lw]) and get_angle(ls, le, lw, w, h)<140 and spd_L>35 and get_dist(nose, lw, w, h)<850: is_fan = True
            if is_fan: act_status = "Fanning (HOT)"; timers["fan"] = now

        # タイマー維持
        for k, v in timers.items():
            if v and now - v < durs[k]:
                if k=="fan": act_status="Fanning (HOT)"
                elif k=="wipe": act_status="Wiping Sweat (HOT)"
                elif k=="arm": act_status="Crossing Arms (COLD)"
                elif k=="warm": act_status="Warming Hands (COLD)"
                break
            else: timers[k] = None

        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # --- 判定 ---
    is_act_hot = 1 if act_status in ["Fanning (HOT)", "Wiping Sweat (HOT)"] else 0
    is_itc_hot = 1 if itc > 1.0 else 0
    is_ptc_hot = 1 if ptc > 1.5 else 0

    is_act_cold = 1 if act_status in ["Crossing Arms (COLD)", "Warming Hands (COLD)"] else 0
    is_itc_cold = 1 if itc < -1.0 else 0
    is_ptc_cold = 1 if ptc < -1.5 else 0

    hot_score = is_act_hot + is_itc_hot + is_ptc_hot
    cold_score = is_act_cold + is_itc_cold + is_ptc_cold

    msg = "Feeling Comfortable"
    conf = ""
    bg_col = (0, 200, 0) # 緑

    if hot_score >= 2:
        msg = "Feeling Hot"
        conf = "(High Confidence)"
        bg_col = (0, 0, 255) # 赤
    elif hot_score == 1:
        msg = "Feeling Hot"
        if is_act_hot: conf = "(Med: Action)"
        elif is_itc_hot: conf = "(Med: Skin)"
        else: conf = "(Low: Room)"
        bg_col = (0, 165, 255) # オレンジ
    elif cold_score >= 2:
        msg = "Feeling Cold"
        conf = "(High Confidence)"
        bg_col = (255, 0, 0) # 青
    elif cold_score == 1:
        msg = "Feeling Cold"
        if is_act_cold: conf = "(Med: Action)"
        elif is_itc_cold: conf = "(Med: Skin)"
        else: conf = "(Low: Room)"
        bg_col = (255, 255, 0) # 水色
    if hot_score > 0 and cold_score > 0:
        msg = "Contradiction!"
        conf = ""
        bg_col = (128, 0, 128) # 紫

    # --- 描画 (見やすく改良) ---
    
    # 色の決定 (暑い=赤, 寒い=青, 普通=白)
    col_ptc = (0, 0, 255) if ptc > 0.5 else (255, 0, 0) if ptc < -0.5 else (255, 255, 255)
    col_itc = (0, 0, 255) if itc > 0.5 else (255, 0, 0) if itc < -0.5 else (255, 255, 255)
    col_act = (0, 0, 255) if "HOT" in act_status else (255, 0, 0) if "COLD" in act_status else (255, 255, 255)

    ard_stat = "" if arduino_connected else "(Disc.)"
    
    # 情報表示 (黒縁取り付きで表示)
    draw_text(img, f'Room: {room_temp:.1f}C / {room_humidity:.0f}% {ard_stat}', 20, 40)
    draw_text(img, f'Skin: {skin_temp:.1f} C', 20, 80)
    
    draw_text(img, f'PTC (Env): {ptc:+.1f}', 20, 130, col_ptc)
    draw_text(img, f'ITC (Body): {itc:+.1f}', 20, 170, col_itc)
    draw_text(img, f'Action: {act_status}', 20, 220, col_act)

    # 判定バー
    cv2.rectangle(img, (0, h-80), (w, h), bg_col, -1)
    cv2.putText(img, msg, (20, h-40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)
    cv2.putText(img, conf, (20, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)

    cv2.imshow('ECO Advisor', img)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
pose.close()
