"""Filtro de media ejecutado mediante PyCUDA."""

from .base import FiltroPyCUDA

CODIGO = r"""
__global__ void mediaKernel(const float* entrada, float* salida,
                            int ancho, int alto, int k, float coeficiente)
{
    const size_t indice = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (indice >= static_cast<size_t>(ancho) * alto) return;
    const int fila = static_cast<int>(indice / ancho);
    const int columna = static_cast<int>(indice % ancho);
    const int radio = k / 2;
    float acumulado = 0.0f;
    for (int dy = -radio; dy <= radio; ++dy) {
        for (int dx = -radio; dx <= radio; ++dx) {
            const int y = fila + dy;
            const int x = columna + dx;
            if (y >= 0 && y < alto && x >= 0 && x < ancho)
                acumulado += entrada[static_cast<size_t>(y) * ancho + x] * coeficiente;
        }
    }
    salida[indice] = acumulado;
}
"""


class FiltroMediaPyCUDA(FiltroPyCUDA):
    """Reserva y ejecuta el kernel de media."""

    nombre_kernel = "mediaKernel"
    k_permitidas = (3, 9, 21)
