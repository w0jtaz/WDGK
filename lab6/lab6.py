from typing import Optional
from lab2.lab2 import BaseImage
from lab3.lab3 import GrayScaleTransform
from lab4.lab4 import ImageComparison, Histogram
from lab5.lab5 import ImageAligning
import numpy as np



class ImageFiltration:
    def conv_2d(self, image: BaseImage, kernel: np.ndarray, prefix: Optional[float] = None) -> BaseImage:
        """
        kernel: filtr w postaci tablicy numpy
        prefix: przedrostek filtra, o ile istnieje; Optional - forma poprawna obiektowo, lub domyslna wartosc = 1 - optymalne arytmetycznie
        metoda zwroci obraz po procesie filtrowania
        """
        pass


class Image(GrayScaleTransform, ImageComparison, ImageAligning, ImageFiltration):
    """
    interfejs glowny biblioteki c.d.
    """
    pass