from lab2.lab2 import BaseImage
from lab3.lab3 import Image, GrayScaleTransform
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
from matplotlib.image import imread


class Histogram:

    #klasa reprezentujaca histogram danego obrazu

    values: np.ndarray  # atrybut przechowujacy wartosci histogramu danego obrazu

    def __init__(self, values: np.ndarray) -> None:
        self.values = values
        pass

    def plot(self) -> None:
        R = self.values[:, :, 0].ravel()
        G = self.values[:, :, 1].ravel()
        B = self.values[:, :, 2].ravel()
        f = plt.figure(figsize=(10, 3))
        ax1 = f.add_subplot(131)
        ax2 = f.add_subplot(132)
        ax3 = f.add_subplot(133)
        ax1.set_ylim([-200, 3800])
        ax2.set_ylim([-200, 2600])
        ax3.set_ylim([-200, 4500])
        f.tight_layout(pad=0.5)
        ax1.hist(B, bins=256, range=[0, 256], color='red', histtype='step')
        ax2.hist(G, bins=256, range=[0, 256], color='green', histtype='step')
        ax3.hist(R, bins=256, range=[0, 256], color='blue', histtype='step')
        plt.show()
        #metoda wyswietlajaca histogram na podstawie atrybutu values
        pass

class ImageDiffMethod(Enum):
    mse = 0
    rmse = 1
class ImageComparison(BaseImage):
    def __init__(self, path:str):
        super().__init__(path)
    #Klasa reprezentujaca obraz, jego histogram oraz metody porównania
    def histogram(self) -> Histogram:
        X = self.data
        if(self.color_model != 4):
            R = X[:, :, 0].ravel()
            G = X[:, :, 1].ravel()
            B = X[:, :, 2].ravel()
            print('G(shape) = '+str(len(G)))
            W1:np.ndarray = np.zeros(256)
            W2: np.ndarray = np.zeros(256)
            W3: np.ndarray = np.zeros(256)
            tmp = 0
            tmp2 = 0
            for i in range(0, R.shape[0]):
                W1[R[i].astype('uint8')] += 1
                tmp += 1
            for i in range(0, G.shape[0]):
                W2[G[i].astype('uint8')] += 1
                tmp += 1
            for i in range(0, B.shape[0]):
                W3[B[i].astype('uint8')] += 1
                tmp += 1
            W = np.stack((W1, W2, W3), axis=1)
            f = plt.figure(figsize=(10, 3))
            ax1 = f.add_subplot(131)
            ax2 = f.add_subplot(132)
            ax3 = f.add_subplot(133)
            f.tight_layout()
            ax1.plot(W[:, 2], color='red')
            ax2.plot(W[:, 1], color='green')
            ax3.plot(W[:, 0], color='blue')
            plt.show()
            return W
        else:
            GRAY = X.ravel()
            W:np.ndarray = np.zeros(256)
            for i in range(0, GRAY.shape):
                W[GRAY[i].astype('uint8')] += 1
            return W
        #metoda zwracajaca obiekt zawierajacy histogram biezacego obrazu (1- lub wielowarstwowy)
        pass


def compare_to(self, other: Image, method: ImageDiffMethod) -> float:
    X1 = self.data
    X1A = X1[:, :, 0]
    X1B = X1[:, :, 1]
    X1C = X1[:, :, 2]
    I1 = ((X1A + X1B + X1C) / 3).ravel()
    X2 = other.data
    X2A = X2[:, :, 0]
    X2B = X2[:, :, 1]
    X2C = X2[:, :, 2]
    I2 = ((X2A + X2B + X2C) / 3).ravel()
    X1: np.ndarray = np.zeros(256)
    X2: np.ndarray = np.zeros(256)
    for i in range(0, I1.shape[0]):
        X1[I1[i].astype('int64')] += 1
    for i in range(0, I2.shape[0]):
        X2[I2[i].astype('int64')] += 1
    # MSE
    if (method == 0):
        MSE: float = 0
        for i in range(0, X1.shape[0]):
            MSE += (X1[i] - X2[i]) * (X1[i] - X2[i])
        MSE = MSE / 256
        return MSE
    # RMSE
    elif (method == 1):
        RMSE: float = 0
        for i in range(0, X1.shape[0]):
            RMSE += (X1[i] - X2[i]) * (X1[i] - X2[i])
        RMSE = np.sqrt(RMSE / 256)
        return RMSE
    # metoda zwracajaca mse lub rmse dla dwoch obrazow
    pass

x = BaseImage('lena.jpg')
y = GrayScaleTransform('lena.jpg')
print(x.color_model)
y.to_sepia(w=40)
y.to_gray2()
y.show_img()
x.to_hsv()
x.to_rgb()
x.show_img()
x.to_hsi()
x.show_img()
img = imread('lena.jpg')
print(img)
z = Histogram(imread('lena.jpg'))
z.plot()

y = GrayScaleTransform('lena.jpg')
y.to_gray()
print(y)
y.show_img()

z = ImageComparison('lena.jpg')
y = z.histogram()
print(y.shape)
print(y)
y.to_gray()
y.show_img()
z2 = Image('lenaplus1.jpg')
print(z.compare_to(z2, 0))