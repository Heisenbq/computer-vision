import cv2
import numpy as np
import matplotlib.pyplot as plt

def gaussian_kernel(size, sigma):
    center = size // 2
    kernel = np.zeros((size, size), dtype=np.float32)

    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    kernel /= np.sum(kernel)

    return kernel

def apply_gaussian_filter(image, size, sigma):
    kernel = gaussian_kernel(size, sigma)
    area = size // 2
    h, w = image.shape
    new_img = np.zeros_like(image, dtype=np.float32)

    for i in range(area, h - area):
        for j in range(area, w - area):
            region = image[i - area:i + area + 1, j - area:j + area + 1]
            new_img[i, j] = np.sum(region * kernel)
    
    return np.uint8(new_img)


img = cv2.imread('assets/images/img2.jpg', cv2.IMREAD_GRAYSCALE)
img_small = cv2.resize(img, (200, 200))  


result = apply_gaussian_filter(img_small, size=11, sigma=3.0)

plt.subplot(1, 2, 1)
plt.imshow(img_small, cmap='gray')
plt.title('Исходное')

plt.subplot(1, 2, 2)
plt.imshow(result, cmap='gray')
plt.title('После фильтра Гаусса')

plt.show()
