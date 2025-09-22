import cv2
import numpy as np
import mediapipe as mp
import time
import os
from PIL import ImageFont, ImageDraw, Image

#字型路徑（微軟正黑體，僅限 Windows）
font_path = "C:/Windows/Fonts/msjh.ttc"
font = ImageFont.truetype(font_path, 28)

#MediaPipe 初始化
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

#建立截圖資料夾
if not os.path.exists("incorrect_postures"):
    os.makedirs("incorrect_postures")

#工具函數：計算三點夾角
def calc_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

#抓 landmark
def get_landmark(landmarks, name):
    lm = mp_pose.PoseLandmark[name]
    return [landmarks[lm.value].x, landmarks[lm.value].y, landmarks[lm.value].z]

#右肩角度
def get_right_shoulder_angle(landmarks):
    shoulder = get_landmark(landmarks, "RIGHT_SHOULDER")
    elbow = get_landmark(landmarks, "RIGHT_ELBOW")
    wrist = get_landmark(landmarks, "RIGHT_WRIST")
    return calc_angle(shoulder, elbow, wrist)

#腰部扭轉
def get_right_waist_twist(landmarks):
    r_shoulder = get_landmark(landmarks, "RIGHT_SHOULDER")
    l_hip = get_landmark(landmarks, "LEFT_HIP")
    r_hip = get_landmark(landmarks, "RIGHT_HIP")
    upper = np.array([r_shoulder[0] - l_hip[0], r_shoulder[1] - l_hip[1]])
    lower = np.array([r_hip[0] - l_hip[0], r_hip[1] - l_hip[1]])
    return calc_angle(lower, [0, 0], upper)

#攝影機初始化
video = cv2.VideoCapture(0)
hold_start_time = None
correct_hold_duration = 2
motion_in_progress = False
posture_score = 0
snapshot_count = 0

cv2.namedWindow("排球攻擊偵測", cv2.WINDOW_NORMAL)

while True:
    ret, frame = video.read()
    if not ret:
        break

    frame_height, frame_width = frame.shape[:2]
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    annotated = frame.copy()
    pil_image = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    results = pose.process(image_rgb)
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # 擷取角度與差值
        shoulder_angle = get_right_shoulder_angle(lm)
        waist_twist = get_right_waist_twist(lm)
        shoulder_y = get_landmark(lm, "RIGHT_SHOULDER")[1]
        elbow_y = get_landmark(lm, "RIGHT_ELBOW")[1]
        elbow_diff = abs(shoulder_y - elbow_y)

        # 姿勢判斷邏輯
        is_correct = False
        if shoulder_angle > 170:
            feedback = "❌ 肩膀抬太高，注意肩夾傷害"
            feedback_color = (255, 0, 0)
        elif 70 <= shoulder_angle <= 110 and elbow_diff < 0.08 and waist_twist > 30:
            is_correct = True
            feedback = "✅ 拉弓姿勢正確，準備攻擊！"
            feedback_color = (0, 255, 0)
        else:
            feedback = "❌ 姿勢錯誤！請拉起手肘並扭轉腰部"
            feedback_color = (0, 0, 255)

        # 分數與截圖
        if is_correct:
            if not motion_in_progress:
                hold_start_time = time.time()
                motion_in_progress = True
            if time.time() - hold_start_time >= correct_hold_duration:
                posture_score += 1
                motion_in_progress = False
        else:
            if motion_in_progress:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"incorrect_postures/bad_posture_{timestamp}_{snapshot_count}.jpg", frame)
                snapshot_count += 1
                motion_in_progress = False

        # 畫骨架
        mp_drawing.draw_landmarks(
            annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=feedback_color, thickness=2),
            mp_drawing.DrawingSpec(color=feedback_color, thickness=2)
        )

        #輔助線工具函數（pixel 座標）
        def p(name):
            x, y = get_landmark(lm, name)[:2]
            return int(x * frame_width), int(y * frame_height)

        #畫輔助線：右臂、肩膀、腰部、軀幹
        cv2.line(annotated, p("RIGHT_SHOULDER"), p("RIGHT_ELBOW"), (0, 255, 0), 2)
        cv2.line(annotated, p("RIGHT_ELBOW"), p("RIGHT_WRIST"), (0, 255, 0), 2)
        cv2.line(annotated, p("LEFT_SHOULDER"), p("RIGHT_SHOULDER"), (255, 0, 0), 2)
        cv2.line(annotated, p("LEFT_HIP"), p("RIGHT_HIP"), (255, 0, 0), 2)
        cv2.line(annotated, p("RIGHT_SHOULDER"), p("RIGHT_HIP"), (255, 255, 0), 2)

        #理想拉弓區塊
        shoulder_px = int(get_landmark(lm, "RIGHT_SHOULDER")[0] * frame_width)
        shoulder_py = int(get_landmark(lm, "RIGHT_SHOULDER")[1] * frame_height)
        elbow_box = {
            "x1": shoulder_px - 100,
            "x2": shoulder_px - 40,
            "y1": shoulder_py - 30,
            "y2": shoulder_py + 30
        }
        cv2.rectangle(annotated,
                      (elbow_box["x1"], elbow_box["y1"]),
                      (elbow_box["x2"], elbow_box["y2"]),
                      (0, 255, 0), 2)
        cv2.putText(annotated, "理想拉弓區", (elbow_box["x1"], elbow_box["y1"] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        #中文提示與資訊
        draw.text((10, 10), f"右肩膀角度: {int(shoulder_angle)}", font=font, fill=(255, 255, 255))
        draw.text((10, 50), f"腰部扭轉角度: {int(waist_twist)}", font=font, fill=(255, 255, 255))
        draw.text((10, 90), f"得分: {posture_score}", font=font, fill=(255, 255, 0))
        draw.text((10, 140), feedback, font=font, fill=feedback_color)

        annotated = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # 顯示畫面
    cv2.imshow("排球攻擊偵測", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
