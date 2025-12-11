import cv2
import numpy as np

def process_image(path, img_size):
    # Считываем изображение
    img = cv2.resize(cv2.imread(path), (img_size, img_size))
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)  # производная по X
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)  # производная по Y

    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    angle = np.arctan2(grad_y, grad_x)

    print("Matrix  len of gradients:\n", magnitude)
    print("\nMatrix  angle of gradients:\n", angle)


    # Вывод окон
    cv2.imshow("Черно-белое", gray)


    cv2.waitKey(0)
    cv2.destroyAllWindows()

process_image("assets/images/img1.png",400)
