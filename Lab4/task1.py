import cv2

def process_image(path, img_size):
    # Считываем изображение
    img = cv2.resize(cv2.imread(path), (img_size, img_size))
   
    
    # Перевод в черно-белый формат
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Вывод чёрно-белого изображения
    cv2.imshow("Черно-белое изображение", gray)
    
    # Применение размытия по Гауссу
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # так тоже можно
    # res3 = apply_gaussian_filter(img_small, size=11, sigma=3.0)

    # Вывод размытые картинки
    cv2.imshow("Размытие по Гауссу", blurred)

    # Ожидание нажатия клавиши
    cv2.waitKey(0)
    cv2.destroyAllWindows()


process_image('assets/images/img1.png',400)