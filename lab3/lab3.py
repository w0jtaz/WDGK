from lab2.lab2 import BaseImage, ColorModel
import numpy as np
from matplotlib.image import imread

class GrayScaleTransform(BaseImage):
    def __init__(self, path: str) -> None:
        self.data = imread(path)
        pass

    def to_gray(self) -> BaseImage:
        R = self.data[:, :, 0].astype('float32')
        G = self.data[:, :, 1].astype('float32')
        B = self.data[:, :, 2].astype('float32')
        I = (R + G + B) / 3
        GRAY = np.stack((I, I, I), axis=2).astype('uint8')
        self.color_model = 4
        self.data = GRAY
        return GRAY
        #metoda zwracajaca obraz w skali szarosci jako obiekt klasy BaseImage
        pass
    def to_gray2(self) -> BaseImage:
        R = self.data[:, :, 0].astype('float32')
        G = self.data[:, :, 1].astype('float32')
        B = self.data[:, :, 2].astype('float32')
        I = (R + G + B) / 3
        GRAY = np.stack((I*0.299, I*0.587, I*0.114), axis=2).astype('uint8')
        self.color_model = 4
        self.data = GRAY
        return GRAY
    def to_sepia(self, alpha_beta: tuple = (None, None), w: int = None) -> BaseImage:
        R = self.data[:, :, 0].astype('float32')
        G = self.data[:, :, 1].astype('float32')
        B = self.data[:, :, 2].astype('float32')
        I = (R + G + B) / 3
        if w is not None:
            L0 = I + 2 * w
            L1 = I + w
            L2 = I
        else:
            L0 = I * alpha_beta[0]
            L1 = I
            L2 = I * alpha_beta[1]
        for i in range(0, L0.shape[0]):
            for j in range(0, L0.shape[1]):
                if(L0[i][j]>255):
                    L0[i][j] = 255
                elif(L0[i][j]<0):
                    L0[i][j] = 0
                if (L1[i][j] > 255):
                    L1[i][j] = 255
                elif (L1[i][j] < 0):
                    L1[i][j] = 0
                if(L2[i][j]>255):
                    L2[i][j] = 255
                elif(L2[i][j]<0):
                    L2[i][j] = 0
        SEPIA = np.stack((L0, L1, L2), axis=2).astype('uint8')
        self.data = SEPIA
        return  SEPIA
        #metoda zwracajaca obraz w sepii jako obiekt klasy BaseImage
        #sepia tworzona metoda 1 w przypadku przekazania argumentu alpha_beta
        #lub metoda 2 w przypadku przekazania argumentu w
        pass

class Image(GrayScaleTransform):
    def __init__(self, path: str):
        super().__init__(path)
    pass
