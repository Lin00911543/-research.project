import cv2
import mediapipe as mp
import math

# 初始化 MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_angle(a, b, c):
    """ 計算三點之間的夾角 """
    ang = math.degrees(
        math.acos(
            ( (a[0]-b[0])*(c[0]-b[0]) + (a[1]-b[1])*(c[1]-b[1]) ) /
            ( math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2) *
              math.sqrt((c[0]-b[0])**2 + (c[1]-b[1])**2) )
        )
    )
    return ang

# 開啟攝影機
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 轉換 BGR → RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # 取得右肩、右肘、右手腕座標
        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        
        # 計算肩角
        shoulder_angle = calculate_angle(shoulder, elbow, wrist)

        # 顯示肩角
        cv2.putText(frame, f"Shoulder Angle: {int(shoulder_angle)} deg",
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        # 畫骨架
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("AI Pose Detection", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
