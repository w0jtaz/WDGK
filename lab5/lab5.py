from lab2.lab2_x import BaseImage
from lab3.lab3 import GrayScaleTransform
from lab4.lab4 import ImageComparison


class Histogram:
    """
    kontunuacja implementacji klasy
    """

    def to_cumulated(self) -> 'Histogram':
        """
        metoda zwracajaca histogram skumulowany na podstawie stanu wewnetrznego obiektu
        """
        pass
class ImageAligning(BaseImage):
    """
    klasa odpowiadająca za wyrównywanie hostogramu
    """
    def __init__(self) -> None:
        """
        inicjalizator ...
        """
        pass

    def align_image(self, tail_elimination: bool = True) -> 'BaseImage':
        """
        metoda zwracajaca poprawiony obraz metoda wyrownywania histogramow
        """
class Image(GrayScaleTransform, ImageComparison, ImageAligning):
    """
    interfejs glownej klasy biblioteki c.d.
    """
    pass