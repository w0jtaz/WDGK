import numpy as np
from numpy import*
from enum import Enum
import matplotlib as plt
from matplotlib.image import imread
from matplotlib.pyplot import *
from matplotlib.image import imsave
from typing import Any



class ColorModel(Enum):
    rgb = 0
    hsv = 1
    hsi = 2
    hsl = 3
    gray = 4

class BaseImage:
    data: np.ndarray  # tensor przechowujacy piksele obrazu
    color_model: ColorModel  # atrybut przechowujacy biezacy model barw obrazu

    def __init__(self, data: Any, color_model: ColorModel) -> None:
        if data is None:
            self.data = None
        elif isinstance(data, str):
            self.data = imread(data)
        else:
            self.data = data
        self.color_model = color_model

    def save_img(self, path: str) -> None:
        """
        metoda zapisujaca obraz znajdujacy sie w atrybucie data do pliku
        """
        imsave('image.jpg', self.data)

    def show_img(self) -> None:
        """
        metoda wyswietlajaca obraz znajdujacy sie w atrybucie data
        """
        imshow(self.data)
        plt.show()

    def get_layer(self, layer_id: int) -> 'Image':
        """
        metoda zwracajaca warstwe o wskazanym indeksie
        """
        layer = squeeze(dsplit(self.data, self.data.shape[-1]))[layer_id]
        return layer

    def get_img_layers(self) -> []:
        return np.squeeze(np.dsplit(self.data, self.data.shape[-1]))

    def to_hsv(self) -> 'Image':
        """
        metoda dokonujaca konwersji obrazu w atrybucie data do modelu hsv
        metoda zwraca nowy obiekt klasy image zawierajacy obraz w docelowym modelu barw
        """
        red, green, blue = self.get_img_layers() / 255
        M = np.max([red, green, blue], axis=0)
        m = np.min([red, green, blue], axis=0)
        V = M / 255
        S = np.where(M > 0, 1 - m / M, 0)
        additionMinusSubtraction = np.power(red, 2) + np.power(green, 2) + np.power(blue,
                                                                                    2) - red * green - red * blue - green * blue
        H = np.where(green >= blue,
                     np.cos((red - green / 2 - blue / 2) / np.sqrt(additionMinusSubtraction)) ** (-1),
                     360 - np.cos((red - green / 2 - blue / 2) / np.sqrt(additionMinusSubtraction)) ** (-1))

        return BaseImage(np.dstack((H, S, V)), ColorModel.hsv)

    def to_hsi(self) -> 'Image':
        """
        metoda dokonujaca konwersji obrazu w atrybucie data do modelu hsi
        metoda zwraca nowy obiekt klasy image zawierajacy obraz w docelowym modelu barw
        """

        red, green, blue = self.get_img_layers() / 255
        M = np.max([red, green, blue], axis=0)
        m = np.min([red, green, blue], axis=0)
        I = (red + green + blue) / 3
        S = np.where(M > 0, 1 - m / M, 0)
        additionMinusSubtraction = red ** 2 + green ** 2 + blue ** 2 - red * green - red * blue - green * blue
        H = np.where(green >= blue,
                     np.cos((red - green / 2 - blue / 2) / np.sqrt(additionMinusSubtraction)) ** (-1),
                    360 - np.cos((red - green / 2 - blue / 2) / np.sqrt(additionMinusSubtraction)) ** (-1))

        return BaseImage(np.dstack((H, S, I)), ColorModel.hsi)

    def to_hsl(self) -> 'Image':
        """
        metoda dokonujaca konwersji obrazu w atrybucie data do modelu hsl
        metoda zwraca nowy obiekt klasy image zawierajacy obraz w docelowym modelu barw
        """
        red, green, blue = self.get_img_layers() / 255
        M = np.max([red, green, blue], axis=0)
        m = np.min([red, green, blue], axis=0)
        L = (0.5 * (M + m)) / 255
        d = (M - m) / 255
        S = np.where(L > 0, d / (1 - np.fabs(2 * L - 1)), 0)
        additionMinusSubtraction = np.power(red, 2) + np.power(green, 2) + np.power(blue,
                                                                                    2) - red * green - red * blue - green * blue
        H = np.where(green >= blue,
                     np.cos((red - green / 2 - blue / 2) / np.sqrt(additionMinusSubtraction)) ** (-1),
                     360 - np.cos((red - green / 2 - blue / 2) / np.sqrt(additionMinusSubtraction)) ** (-1))

        return BaseImage(np.dstack((H, S, L)), ColorModel.hsl)

    def hsv_to_rgb(self) -> 'BaseImage':
        if self.color_model != ColorModel.hsv:
            raise Exception("color_model must be hsv to use this method!")
        H, S, V = self.get_img_layers()
        M = 255 * V
        m = M * (1 - S)
        z = (M - m) * (1 - np.fabs(((H / 60) % 2) - 1))
        r = np.where(H >= 300, M, np.where(H >= 240, z + m, np.where(H >= 120, m, np.where(H >= 60, z + m, M))))
        g = np.where(H >= 300, z + m, np.where(H >= 240, m, np.where(H >= 120, M, np.where(H >= 60, M, z + m))))
        b = np.where(H >= 300, m, np.where(H >= 240, M, np.where(H >= 120, z + m, m)))

        return BaseImage(np.dstack((r, g, b)), ColorModel.rgb)

    def hsi_to_rgb(self) -> 'BaseImage':
        if self.color_model != ColorModel.hsi:
            raise Exception("color_model must be hsi to use this method!")
        H, S, I = self.get_img_layers()
        IS = I * S
        r = np.where(H > 240, I + IS * (1 - np.cos(H - 240) / np.cos(300 - H)),
            np.where(H >= 120, I - IS, np.where(H > 0, I + IS * np.cos(H) / np.cos(60 - H), I + 2 * IS)))
        g = np.where(H >= 240, I - IS, np.where(H > 120, I + IS * np.cos(H - 120) / np.cos(180 - H), np.where(H == 120, I + 2 * IS,
            np.where(H > 0, I + IS * (1 - np.cos(H) / np.cos(60 - H)),I - IS))))
        b = np.where(H > 240, I + IS * np.cos(H - 240) / np.cos(300 - H), np.where(H == 240, I + 2 * IS,
            np.where(H > 120, I + IS * (1 - np.cos(H - 120) / np.cos(180 - H)),I - IS)))

        return BaseImage(np.dstack((r, g, b)), ColorModel.rgb)

    def hsl_to_rgb(self) -> 'BaseImage':
        if self.color_model != ColorModel.hsl:
            raise Exception("to use this method, color model of img must be hsl!")
        H, S, L = self.get_img_layers()
        d = S * (1 - np.fabs(2 * L - 1))
        m = 255 * (L - 0.5 * d)
        x = d * (1 - np.fabs(((H / 60) % 2) - 1))
        r = np.where(H >= 300, 255 * d + m, np.where(H >= 240, 255 * x + m, np.where(H >= 120, m,
            np.where(H >= 60, 255 * x + m, 255 * d + m))))
        g = np.where(H >= 240, m, np.where(H >= 180, 255 * x + m, np.where(H >= 60, 255 * d + m, 255 * x + m)))
        b = np.where(H >= 300, 255 * x + m, np.where(H >= 240, 255 * d + m, np.where(H >= 180, 255 * d + m,
            np.where(H >= 120, 255 * x + m, m))))

        return BaseImage(np.dstack((r, g, b)), ColorModel.rgb)

    def to_rgb(self) -> 'Image':
        """
        metoda dokonujaca konwersji obrazu w atrybucie data do modelu rgb
        metoda zwraca nowy obiekt klasy image zawierajacy obraz w docelowym modelu barw
        """
        if self.color_model == ColorModel.hsv:
            return self.hsv_to_rgb()
        if self.color_model == ColorModel.hsi:
            return self.hsi_to_rgb()
        if self.color_model == ColorModel.hsl:
            return self.hsl_to_rgb()