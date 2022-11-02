from lab2.lab2 import BaseImage
import numpy as np
from matplotlib.image import imread

class GrayScaleTransform(BaseImage):
    data: np.ndarray
    def __init__(self, path: str) -> None:
        self.data=imread(path)


    def to_gray(self) -> BaseImage:
        """
        metoda zwracajaca obraz w skali szarosci jako obiekt klasy BaseImage
        """
        return np.dot(self[...,3],[0.299, 0.587, 0.114])

    def to_sepia(self, alpha_beta: tuple = (None, None), w: int = None) -> BaseImage:
        """
        metoda zwracajaca obraz w sepii jako obiekt klasy BaseImage
        sepia tworzona metoda 1 w przypadku przekazania argumentu alpha_beta
        lub metoda 2 w przypadku przekazania argumentu w
        """
        pass
class Image(GrayScaleTransform):
    """
    klasa stanowiaca glowny interfejs biblioteki
    w pozniejszym czasie bedzie dziedziczyla po kolejnych klasach
    realizujacych kolejne metody przetwarzania obrazow
    """
    def __init__(self, path:str) -> None:
        pass