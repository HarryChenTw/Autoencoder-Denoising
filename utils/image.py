import numpy as np

def gaussian_noise(image, proportion:float=0.3, mean:float=0, std:float=1):
    row,col = image.shape
    gauss = np.random.normal(mean,std,(row,col))
    gauss = gauss.reshape(row,col)
    noisy_image = image + gauss*proportion
    return np.clip(noisy_image,0,1)