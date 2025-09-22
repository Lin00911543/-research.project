import cv2
import mediapipe as mp

# 初始化 MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# 視訊擷取
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # BGR 轉 RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    
    # 如果偵測到骨架
    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # 範例：擷取右肩與右肘座標計算肩角
        landmarks = results.pose_landmarks.landmark
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        
        # 簡單判斷肩角是否小於120度 → 顯示警告
        if right_elbow.y < right_shoulder.y:
            cv2.putText(frame, "Elbow OK", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        else:
            cv2.putText(frame, "Raise your elbow!", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    
    cv2.imshow("AI Training Feedback", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC 離開
        break

cap.release()
cv2.destroyAllWindows()
