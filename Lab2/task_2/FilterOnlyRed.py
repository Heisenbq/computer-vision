import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = (170, 120, 70)
    upper_red = (180, 255, 255)

    red = cv2.inRange(hsv, lower_red, upper_red)

    cv2.imshow('Original', frame)
    cv2.imshow('Red Only', red)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
