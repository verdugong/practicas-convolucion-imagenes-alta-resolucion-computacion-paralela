"""Convoluciones explicitas float32. Preparacion separada del calculo medido."""
from abc import ABC, abstractmethod
import numpy as np


class FiltroCPU(ABC):
    k_permitidas = (3,)

    def __init__(self, k=3):
        if k not in self.k_permitidas:
            raise ValueError("Mascara invalida para el filtro.")
        self.k = k

    @staticmethod
    def _validar(entrada):
        if (entrada.ndim != 2 or entrada.dtype != np.float32
                or not entrada.flags.c_contiguous or not entrada.size):
            raise ValueError("Entrada requerida: matriz float32 contigua no vacia.")

    def preparar(self, entrada):
        self._validar(entrada)
        return np.empty_like(entrada)

    def ejecutar(self, entrada):
        """Conveniencia para pruebas; el benchmark usa preparar + aplicar."""
        salida = self.preparar(entrada)
        self.aplicar(entrada, salida)
        return salida

    @abstractmethod
    def aplicar(self, entrada, salida):
        """Solo convolucion; el llamador valida y reserva antes de medir."""


class FiltroMediaCPU(FiltroCPU):
    k_permitidas = (3, 9, 21)

    def __init__(self, k=3):
        super().__init__(k)
        self.radio = k // 2
        self.coeficiente = np.float32(1) / np.float32(k * k)

    def aplicar(self, entrada, salida):
        alto, ancho = entrada.shape
        radio, coeficiente = self.radio, self.coeficiente
        for fila in range(alto):
            for columna in range(ancho):
                acumulado = np.float32(0)
                for dy in range(-radio, radio + 1):
                    for dx in range(-radio, radio + 1):
                        y, x = fila + dy, columna + dx
                        if 0 <= y < alto and 0 <= x < ancho:
                            acumulado += entrada[y, x] * coeficiente
                salida[fila, columna] = acumulado


class FiltroMascaraCPU(FiltroCPU):
    coeficientes = ()

    def __init__(self, k=3):
        super().__init__(k)
        self.mascara = np.array(self.coeficientes, dtype=np.float32)

    def aplicar(self, entrada, salida):
        alto, ancho = entrada.shape
        mascara = self.mascara
        for fila in range(alto):
            for columna in range(ancho):
                suma = np.float32(0)
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        y, x = fila + dy, columna + dx
                        if 0 <= y < alto and 0 <= x < ancho:
                            suma += entrada[y, x] * mascara[(dy + 1) * 3 + dx + 1]
                salida[fila, columna] = suma


class FiltroNitidezCPU(FiltroMascaraCPU):
    coeficientes = (0, -1, 0, -1, 5, -1, 0, -1, 0)


class FiltroLaplacianoCPU(FiltroMascaraCPU):
    coeficientes = (0, 1, 0, 1, -4, 1, 0, 1, 0)


def crear_filtro_cpu(filtro, k):
    clases = {"media": FiltroMediaCPU, "nitidez": FiltroNitidezCPU,
              "laplaciano": FiltroLaplacianoCPU}
    if filtro not in clases:
        raise ValueError("Filtro permitido: media, nitidez o laplaciano.")
    return clases[filtro](k)
