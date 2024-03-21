import cv2

# 加载人脸识别分类器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 加载人眼识别分类器
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# 打开视频文件
cap = cv2.VideoCapture("Video/chen3.mp4")

while True:
    # 读取视频中的一帧
    ret, frame = cap.read()
    if not ret:
        break

    # 将图像转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # 在每张脸上标记眼睛
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # 眼睛检测
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            # 计算眼睛中心点坐标
            eye_center = (x + ex + int(ew / 2), y + ey + int(eh / 2))
            # 在眼睛中心点处画一个红色的点
            cv2.circle(frame, eye_center, 2, (0, 0, 255), -1)  # 使用 -1 填充圆点

    # 显示视频帧
    cv2.imshow('Video', frame)

    # 按 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放视频流
cap.release()
cv2.destroyAllWindows()
