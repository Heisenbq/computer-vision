import cv2

videoThread = cv2.VideoCapture("assets/videos/sample_1.mp4", cv2.CAP_ANY)

while (True):
    ret, frame = videoThread.read()
    if not(ret):
        break
    cv2.imshow('frame', frame)
    if cv2.waitKey(35) & 0xFF == 27:
        break

videoThread = cv2.VideoCapture("assets/videos/sample_2.gif", cv2.CAP_ANY)

while (True):
    ret, frame = videoThread.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if not(ret):
        break
    cv2.imshow('frame', gray)
    if cv2.waitKey(60) & 0xFF == 27:
        break