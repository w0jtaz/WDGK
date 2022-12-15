from lab2.lab2 import BaseImage
from lab3.lab3 import GrayScaleTransform
from lab4.lab4 import ImageComparison
import numpy as np
import matplotlib.pyplot as plt


class Histogram:

    def __init__(self, values: np.ndarray) -> None:
        if values.shape[-1] == 3:
            r = np.histogram(values[:, :, 0], bins=np.linspace(0, 255, 256))
            g = np.histogram(values[:, :, 1], bins=np.linspace(0, 255, 256))
            b = np.histogram(values[:, :, 2], bins=np.linspace(0, 255, 256))
            self.values = list(r + g + b)
        else:
            self.values = list(np.histogram(values, bins=np.linspace(0, 255, 256)))
    def to_cumulated(self) -> 'Histogram':
        if len(self.values) == 6:
            self.values[0] = list(np.cumsum(self.values[0]))
            self.values[2] = list(np.cumsum(self.values[2]))
            self.values[4] = list(np.cumsum(self.values[4]))
        else:
            self.values[0] = list(np.cumsum(self.values[0]))
        return self

        pass

    def plot(self) -> None:
        if len(self.values) == 6:
            r, g, b = self.values[0], self.values[2], self.values[4]
            x_r, x_g, x_b = self.values[1], self.values[3], self.values[5]
            r, g, b = np.append(r, [0]), np.append(g, [0]), np.append(b, [0])
            plt.plot(x_r, r, color='red', linewidth=0.5)
            plt.plot(x_g, g, color='green', linewidth=0.5)
            plt.plot(x_b, b, color='blue', linewidth=0.5)
        else:
            x = np.append(self.values[0], [0])
            plt.plot(self.values[1], x, color='black', linewidth=0.5)
        plt.xlim(0, 255)
        plt.title("Histogram")
        plt.show()
class ImageAligning(BaseImage):

    def __init__(self, path) -> None:
        """
        inicjalizator ...
        """
        super().__init__(path)

    def align_image(self, tail_elimination: bool = True) -> 'BaseImage':




class Image(GrayScaleTransform, ImageComparison, ImageAligning):
    def __init__(self, path: str):
        super().__init__(path)

    pass



x = Image("lena.jpg")
x.to_gray()
histogram = Histogram(x.data)
histogram.to_cumulated().plot()


y = Image("lena.jpg")
y= Histogram(y.data)
y.to_cumulated().plot()

z= Image("lena.jpg")
z.to_gray()
z = Histogram(z.data)
z.plot()