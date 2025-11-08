import cv2

cap = cv2.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 480)

while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)

    if not(ret):
        break
    if cv2.waitKey(16) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

