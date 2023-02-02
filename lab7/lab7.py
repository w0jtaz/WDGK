from lab2.lab2 import BaseImage
from lab3.lab3 import GrayScaleTransform
from lab4.lab4 import ImageComparison
from lab5.lab5 import ImageAligning
from lab6.lab6 import ImageFiltration

class Thresholding(BaseImage):
    def threshold(self, value: int) -> BaseImage:
        """
        metoda dokonujaca operacji segmentacji za pomoca binaryzacji
        """
        pass
class Image(GrayScaleTransform, ImageComparison, ImageAligning, ImageFiltration, Thresholding):
    """
    interfejs glowny biblioteki c.d.
    """
    pass