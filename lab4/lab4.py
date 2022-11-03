from enum import Enum
from lab2.lab2 import BaseImage
from lab3.lab3 import Image, GrayScaleTransform
import numpy as np


class Histogram:
    """
    klasa reprezentujaca histogram danego obrazu
    """
    values: np.ndarray  # atrybut przechowujacy wartosci histogramu danego obrazu

    def __init__(self, values: np.ndarray) -> None:
        pass

    def plot(self) -> None:
        """
        metoda wyswietlajaca histogram na podstawie atrybutu values
        """
        pass


class ImageDiffMethod(Enum):
    mse = 0
    rmse = 1
class ImageComparison(BaseImage):
    """
    Klasa reprezentujaca obraz, jego histogram oraz metody porównania
    """

    def histogram(self) -> Histogram:
        """
        metoda zwracajaca obiekt zawierajacy histogram biezacego obrazu (1- lub wielowarstwowy)
        """
        pass

    def compare_to(self, other: Image, method: ImageDiffMethod) -> float:
        """
        metoda zwracajaca mse lub rmse dla dwoch obrazow
        """
        pass
class Image(GrayScaleTransform, ImageComparison):
    pass