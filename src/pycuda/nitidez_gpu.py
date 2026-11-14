"""Filtro de nitidez ejecutado mediante PyCUDA."""

from .base import FiltroPyCUDA

CODIGO = r"""
__global__ void nitidezKernel(const float* entrada, float* salida, int ancho, int alto)
{
    const size_t indice = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (indice >= static_cast<size_t>(ancho) * alto) return;
    const int fila = static_cast<int>(indice / ancho);
    const int columna = static_cast<int>(indice % ancho);
    const float mascara[9] = {0,-1,0,-1,5,-1,0,-1,0};
    float suma = 0.0f;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            const int y = fila + dy;
            const int x = columna + dx;
            if (y >= 0 && y < alto && x >= 0 && x < ancho)
                suma += entrada[static_cast<size_t>(y) * ancho + x]
                      * mascara[(dy + 1) * 3 + dx + 1];
        }
    }
    salida[indice] = suma;
}
"""


class FiltroNitidezPyCUDA(FiltroPyCUDA):
    """Reserva y ejecuta el kernel de nitidez."""

    nombre_kernel = "nitidezKernel"
