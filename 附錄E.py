import cv2
import dlib
import numpy as np
import pickle

# 初始化檢測與識別器
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# 載入已註冊學生資料（特徵向量 + UUID）
with open("students_face_db.pkl", "rb") as f:
    student_db = pickle.load(f)  # 格式: {uuid: {"name": "學生A", "vector": np.array([...])}}

# 歐氏距離計算
def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

# 攝影機啟動
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector(rgb_frame)
    
    for face in faces:
        shape = sp(rgb_frame, face)
        face_descriptor = np.array(facerec.compute_face_descriptor(rgb_frame, shape))
        
        # 身份比對
        matched_uuid = None
        min_distance = 0.6
        for uuid, info in student_db.items():
            dist = euclidean_distance(face_descriptor, info["vector"])
            if dist < min_distance:
                min_distance = dist
                matched_uuid = uuid
        
        # 顯示識別結果
        (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
        if matched_uuid:
            name = student_db[matched_uuid]["name"]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} - ID:{matched_uuid}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 範例：將姿勢數據與 UUID 綁定
            # save_pose_data(matched_uuid, pose_parameters)
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, "Unknown", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
