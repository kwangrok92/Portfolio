import cv2
import dlib

# cap = cv2.VideoCapture(0)  # Cam
cap = cv2.VideoCapture("Zuckerburg.mp4")  # Input the Video
# cap.set(cv2.CAP_PROP_FPS, 100)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

        landmarks = predictor(gray, face)

        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.putText(frame, str(n), (x, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3, (0, 255, 0))  # 랜드마크 숫자로 찍기
            # cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)  # 랜드마크 점으로 찍기

    cv2.imshow("Face Landmark Detector", frame)  # 원본
    # frame_rs = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    # cv2.imshow("Face Landmark Detector", frame_rs)  # 1.5배 확대

    # 아무 키를 누르면 창 종료 (키를 누르지 않으면 -1 이 반환되기 때문에)
    if cv2.waitKey(10) > -1:
        break

cap.release()
cv2.destroyAllWindows()
