import numpy as np

def gaussian_kernel(size, sigma,is_norm):
    center = size // 2  
    kernel = np.zeros((size, size), dtype=np.float32)

    
    a, b = 0, 0  
    sum = 0;

    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            kernel[i, j] = (1 / (2 * np.pi * sigma**2)) * np.exp(-((x - a)**2 + (y - b)**2) / (2 * sigma**2))
            sum += kernel[i,j]

    if is_norm: kernel = kernel / sum

    return kernel



kernel_3x3 = gaussian_kernel(3, 1, False)
print("Yadro Svertki 3x3 (sigma=1) bez normirovki:")
print(kernel_3x3)

print("\nSumma elem yadra:", np.sum(kernel_3x3))

kernel_3x3 = gaussian_kernel(3, 1, True)
print("Yadro Svertki 3x3 (sigma=1) s normirovkoi:")
print(kernel_3x3)

print("\nSumma elem yadra:", np.sum(kernel_3x3))