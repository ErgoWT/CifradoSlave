"""
Este código será experimental para implementar
el sistema Maestro-Esclavo en un mismo código
"""
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from PIL import Image

# PARÁMETROS DE ROSSLER
ROSSLER_PARAMS = {
    'a': 0.2,
    'b': 0.2,
    'c': 5.7
}
H = 0.01
TIME_SINC = 2500
KEYSTREAM = 20000
Y0_MAESTRO = [0.1, 0.1, 0.1]
Y0_ESCLAVO = [1.0, 1.0, 1.0]
K = 2.0

LOGISTIC_PARAMS = {
    'aLog': 3.99,
    'x0Log': 0.5
}

# ========== RUTAS Y ARCHIVOS ==========
CARPETA_SALIDA = Path("Resultados_Local")
CARPETA_SALIDA.mkdir(exist_ok=True)
IMAGEN_ENTRADA = Path("Prueba2.jpg")
RUTA_IMAGEN_CIFRADA = CARPETA_SALIDA / "imagen_cifrada.png"
RUTA_IMAGEN_DESCIFRADA = CARPETA_SALIDA / "imagen_descifrada.png"
RUTA_DISPERSION = CARPETA_SALIDA / "dispersion_original_vs_cifrada.png"
RUTA_HAMMING = CARPETA_SALIDA / "hamming.png"
RUTA_SERIE_Y = CARPETA_SALIDA / "serie_temporal_y.png"
RUTA_SERIE_Y_ACOPLE = CARPETA_SALIDA / "serie_temporal_y_acople.png"
RUTA_ERROR_Y = CARPETA_SALIDA / "error_y_vs_tiempo.png"
RUTA_DISPERSION_Y = CARPETA_SALIDA / "dispersion_y_maestro_vs_esclavo.png"
RUTA_DISPERSION_PIXELES = CARPETA_SALIDA / "dispersion_pixeles.png"
RUTA_HISTOGRAMAS = CARPETA_SALIDA / "histogramas.png"
RUTA_TIMINGS = CARPETA_SALIDA / "tiempos_procesos.csv"

# ========== DINÁMICA COMBINADA MAESTRO–ESCLAVO ==========
def rossler_maestro_esclavo(t, state, a, b, c, k):
    """
    Estado: [x_m, y_m, z_m, x_s, y_s, z_s]
    """
    x_m, y_m, z_m, x_s, y_s, z_s = state

    # Maestro
    dxm = -y_m - z_m
    dym = x_m + a * y_m
    dzm = b + z_m * (x_m - c)

    # Esclavo (acoplado en y)
    dxs = -y_s - z_s
    dys = x_s + a * y_s + k * (y_m - y_s)
    dzs = b + z_s * (x_s - c)

    return [dxm, dym, dzm, dxs, dys, dzs]

def integrar_sistema(rossler_params, h, time_sinc, keystream, y0_maestro, y0_esclavo, k):
    iteraciones = time_sinc + keystream
    t_span = (0, iteraciones * h)
    t_eval = np.arange(0, iteraciones * h, h)

    y0 = [*y0_maestro, *y0_esclavo]

    sol = solve_ivp(
        fun=rossler_maestro_esclavo,
        t_span=t_span,
        y0=y0,
        args=(
            rossler_params['a'],
            rossler_params['b'],
            rossler_params['c'],
            k
        ),
        t_eval=t_eval,
        method='RK45',
        rtol=1e-5,
        atol=1e-5
    )

    x_m, y_m, z_m, x_s, y_s, z_s = sol.y
    return {
        "t": sol.t,
        "master": {"x": x_m, "y": y_m, "z": z_m},
        "slave": {"x": x_s, "y": y_s, "z": z_s},
    }

def cargar_imagen(ruta_imagen):
    imagen = Image.open(ruta_imagen).convert()