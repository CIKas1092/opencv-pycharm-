import cv2
import mediapipe as mp
import time



class FaceDetector():
    def __init__(self, minDetectionCon=0.5):
        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxs = []

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                    int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])
                if draw:
                    img = self.fancyDraw(img, bbox)
                    cv2.putText(img, f'{int(detection.score[0] * 100)}%',
                                (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                                3, (255, 0, 0), 3)
        return img, bboxs

    def fancyDraw(self, img, bbox, l=20, t=5, rt=1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        cv2.rectangle(img, bbox, (0, 0, 255), rt)
        cv2.line(img, (x, y), (x + l, y), (0, 0, 255), t)
        cv2.line(img, (x, y), (x, y + l), (0, 0, 255), t)
        cv2.line(img, (x1, y), (x1 - l, y), (0, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y + l), (0, 0, 255), t)
        cv2.line(img, (x, y1), (x + l, y1), (0, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (0, 0, 255), t)
        cv2.line(img, (x1, y1), (x1 - l, y1), (0, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (0, 0, 255), t)
        return img


def main():
    cap = cv2.VideoCapture("Video/chen4.mp4")
    pTime = 0
    detector = FaceDetector()

    # 输出视频设置
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, (int(cap.get(3)), int(cap.get(4))))

    while True:
        success, img = cap.read()
        if not success:
            break

        # 人脸检测
        img, _ = detector.findFaces(img)

        # 人眼检测
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                eye_center = (x + ex + int(ew / 2), y + ey + int(eh / 2))
                cv2.circle(img, eye_center, 2, (0, 0, 255), -1)

        # 显示视频帧
        cv2.imshow('Combined Video', img)
        out.write(img)

        # 按 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()

#     cap =cv2.VideoCapture("Video/chen4.mp4")
#     pTime = 0
#     detector = FaceDetector()
#         cTime = time.time()
#         fps=1/(cTime-pTime)
#         pTime = cTime
#         cv2.putText(img,f'FPS: {int(fps)}',(20,70),cv2.FONT_HERSHEY_PLAIN,3,(255,255,0),2)
# import cv2



if __name__ == "__main__":
    main()
