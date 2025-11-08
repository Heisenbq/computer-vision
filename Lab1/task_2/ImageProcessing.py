import cv2

img1 = cv2.imread("assets/images/img1.png", cv2.IMREAD_COLOR)

cv2.namedWindow("Display window", cv2.WINDOW_NORMAL)
cv2.imshow("Display window", img1)
cv2.waitKey(0)
cv2.destroyAllWindows()

img2 = cv2.imread("assets/images/img2.jpg", cv2.IMREAD_REDUCED_GRAYSCALE_8)

cv2.namedWindow("Display window", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Display window", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

img3 = cv2.imread("assets/images/img3.bmp", cv2.IMREAD_GRAYSCALE)

cv2.namedWindow("Display window", cv2.WINDOW_FREERATIO)
cv2.imshow("Display window", img3)
cv2.waitKey(0)
cv2.destroyAllWindows()


