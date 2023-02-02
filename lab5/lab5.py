from lab2.lab2 import BaseImage
from lab3.lab3 import GrayScaleTransform
# from lab4.lab4 import ImageComparison
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

    def align_channel(self, channel: np.ndarray, tail_elimination: bool = False) -> np.ndarray:
        """
        metoda wyrównująca histogram danego kanału
        """
        channel = channel.astype(np.float64)
        rows, cols = self.data.shape[:2]
        max_value = np.max(channel)
        min_value = np.min(channel)
        if tail_elimination == True:
            max_value = np.percentile(channel, 95)
            min_value = np.percentile(channel, 5)
        for i in range(rows):
            for j in range(cols):
                try:
                    channel[i, j] = ((channel[i, j] - min_value) / (max_value - min_value)) * 255
                except ZeroDivisionError:
                    print("ZeroDivisionError")
        channel[channel > 255] = 255
        channel[channel < 0] = 0
        return channel.astype('uint8')

    def align_image(self, tail_elimination: bool = False) -> 'BaseImage':
        """
        metoda wyrównująca histogram obrazu
        """
        if self.data.shape[-1] == 3:
            r = self.align_channel(self.data[:, :, 0], tail_elimination)
            g = self.align_channel(self.data[:, :, 1], tail_elimination)
            b = self.align_channel(self.data[:, :, 2], tail_elimination)
            self.data = np.dstack((r, g, b))
        else:
            self.data = self.align_channel(self.data, tail_elimination)
        return self




class Image(GrayScaleTransform, ImageAligning):
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

q = Image("lena.jpg")
q.align_image(True)
q = Histogram(q.data)
q.plot()