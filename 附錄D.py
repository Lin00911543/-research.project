import cv2
import pyttsx3

# 初始化語音引擎
engine = pyttsx3.init()

def give_feedback(frame, angle):
    """根據角度給回饋"""
    if 120 <= angle <= 170:
        color = (0, 255, 0)  # 綠框
        text = "Good posture!"
        engine.say("Good posture")
    else:
        color = (0, 0, 255)  # 紅框
        text = "Raise your elbow!"
        engine.say("Raise your elbow")

    # 畫框與文字
    h, w, _ = frame.shape
    cv2.rectangle(frame, (50, 50), (w-50, h-50), color, 3)
    cv2.putText(frame, f"Shoulder Angle: {int(angle)} deg", (60, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, text, (60, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    engine.runAndWait()
    return frame

# 假設我們已經取得一個測試影像與肩角
test_frame = cv2.imread("test.jpg")
feedback_frame = give_feedback(test_frame, 110)  # 模擬錯誤姿勢
cv2.imshow("Feedback", feedback_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
