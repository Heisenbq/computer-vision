import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]
    x, y = width // 2, height // 2

    b, g, r = frame[y, x]

    if r >= g and r >= b:
        color = (0, 0, 255)
    elif g >= r and g >= b:
        color = (0, 255, 0)
    else:
        color = (255, 0, 0)

    cv2.rectangle(frame, (x - 50, y - 200), (x + 50, y + 200), color, -1)
    cv2.rectangle(frame, (x - 200, y - 50), (x + 200, y + 50), color, -1)

    cv2.imshow("Filled Cross", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
