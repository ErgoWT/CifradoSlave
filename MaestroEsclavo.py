"""
Este código será experimental para implementar
el sistema Maestro-Esclavo en un mismo código
"""
from pathlib import Path
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from PIL import Image

# PARÁMETROS DE ROSSLER
ROSSLER_PARAMS = {
    'a': 0.2,
    'b': 0.2,
    'c': 5.7
}

A = 0.3
B = 0.201
C = 6.2

H = 0.01
TIEMPO_SINC = 3500
KEYSTREAM = 40000
Y0_MAESTRO = [0.1, 0.1, 0.1]
Y0_ESCLAVO = [1.0, 1.0, 1.0]
K = 2.0

LOGISTIC_PARAMS = {
    'aLog': 3.99,
    'x0_log': 0.4
}

# ========== RUTAS Y ARCHIVOS ==========
CARPETA_RESULTADOS = Path("Resultados_Local")
CARPETA_RESULTADOS.mkdir(exist_ok=True)
CARPETA_HAMMING_LOGISTIC = CARPETA_RESULTADOS / "Hamming_Logistic"
CARPETA_HAMMING_LOGISTIC.mkdir(parents=True, exist_ok=True)
IMAGEN_ORIGINAL = Path("Prueba2.jpg")
IMAGEN_DESCIFRADA = CARPETA_RESULTADOS / "ImagenDescifrada_Local.png"

# SERIES TEMPORALES
RUTA_SERIE_X = CARPETA_RESULTADOS / "x_maestro_esclavo.png"
RUTA_SERIE_Y = CARPETA_RESULTADOS / "y_maestro_esclavo.png"
RUTA_SERIE_Z = CARPETA_RESULTADOS / "z_maestro_esclavo.png"
# ERROR
RUTA_ERRORES = CARPETA_RESULTADOS / "errores_sincronizacion.csv"
RUTA_ERRORES_MEDIOS = CARPETA_RESULTADOS / "errores_medios_sincronizacion.csv"
RUTA_ERROR_X = CARPETA_RESULTADOS / "error_x.png"
RUTA_ERROR_Y = CARPETA_RESULTADOS / "error_y.png"
RUTA_ERROR_Z = CARPETA_RESULTADOS / "error_z.png"
# DISPERSION
RUTA_DISPERSION_X = CARPETA_RESULTADOS / "dispersion_x.png"
RUTA_DISPERSION_Y = CARPETA_RESULTADOS / "dispersion_y.png"
RUTA_DISPERSION_Z = CARPETA_RESULTADOS / "dispersion_z.png"

# SALIDAS CIFRADO
RUTA_IMAGEN_DIFUSION = CARPETA_RESULTADOS / "ImagenDifusion_Local.png"
RUTA_IMAGEN_CONFUSION = CARPETA_RESULTADOS / "ImagenConfusion_Local.png"
RUTA_IMAGEN_CIFRADA = CARPETA_RESULTADOS / "ImagenCifrada_Local.png"
RUTA_TIMINGS = CARPETA_RESULTADOS / "tiempos_procesos.csv"
RUTA_DISPERSION = CARPETA_RESULTADOS / "diagrama_dispersion.png"
RUTA_HAMMING = CARPETA_RESULTADOS / "hamming.csv"
RUTA_SERIES_VECTORES = CARPETA_RESULTADOS / "series_difusion_logistico_rossler.png"
RUTA_VECTOR_CIFRADO_SERIE = CARPETA_RESULTADOS / "vector_cifrado.png"
RUTA_HISTOGRAMA_IMAGENES = CARPETA_RESULTADOS / "histogramas_imagenes.png"
RUTA_CORRELACION = CARPETA_RESULTADOS / "correlacion.csv"

PUNTOS_EVAL = 10000

RUTA_DISTANCIA_HAMMING_A = CARPETA_RESULTADOS / "hamming_vs_a.png"
RUTA_DISTANCIA_HAMMING_CSV_A = CARPETA_RESULTADOS / "hamming_vs_a.csv"
RUTA_DISTANCIA_HAMMING_B = CARPETA_RESULTADOS / "hamming_vs_b.png"
RUTA_DISTANCIA_HAMMING_CSV_B = CARPETA_RESULTADOS / "hamming_vs_b.csv"
RUTA_DISTANCIA_HAMMING_C = CARPETA_RESULTADOS / "hamming_vs_c.png"
RUTA_DISTANCIA_HAMMING_CSV_C = CARPETA_RESULTADOS / "hamming_vs_c.csv"

RUTA_DISTANCIA_HAMMING_A_LOG = CARPETA_RESULTADOS / "hamming_vs_aLog.png"
RUTA_DISTANCIA_HAMMING_CSV_A_LOG = CARPETA_RESULTADOS / "hamming_vs_aLog.csv"
RUTA_DISTANCIA_HAMMING_X0_LOG = CARPETA_RESULTADOS / "hamming_vs_x0_log.png"
RUTA_DISTANCIA_HAMMING_CSV_X0_LOG = CARPETA_RESULTADOS / "hamming_vs_x0_log.csv"

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

def mapa_logistico(logistic_params, nmax):
    a = logistic_params["aLog"]
    x0 = logistic_params["x0_log"]
    vector_logistico = np.zeros(nmax)
    for i in range(nmax):
        vector_logistico[i] = x0
        x0 = a * x0 * (1 - x0)

    return vector_logistico


def integrar_rossler(rossler_params, nmax):
    a = rossler_params['a']
    b = rossler_params['b']
    c = rossler_params['c']

    iteraciones = TIEMPO_SINC + KEYSTREAM
    t_span = (0, iteraciones * H)
    t_eval = np.arange(0, iteraciones * H, H)

    y0 = np.array(Y0_MAESTRO + Y0_ESCLAVO, dtype=float)

    sol = solve_ivp(
        fun=rossler_maestro_esclavo,
        t_span=t_span,
        y0=y0,
        args=(a, b, c, K),
        t_eval=t_eval,
        method='RK23', # Todos los metodos para solve_ivp son: 'RK23', 'RK45', 'DOP853', 'Radau', 'BDF', 'LSODA'
        rtol=1e-6,
        atol=1e-8,
    )
    """
    Anotaciones respecto a sol usando RK45 como base:
    RK23 si logró un error de sincronización menor, logrando una trayectoria continua de error = 0
    DOP853 logró resultados similares a RK45
    RADAU igual, similar a RK45
    BDF logra error menor, pero no continuo, con saltos en el error
    LSODA mismos casos con los demás

    En conclusión, actualmente RK23 es el que mejores resultados ha dado
    """
    x_maestro = sol.y[0]
    y_maestro = sol.y[1]
    z_maestro = sol.y[2]
    x_esclavo = sol.y[3]
    y_esclavo = sol.y[4]
    z_esclavo = sol.y[5]
    x_cif = sol.y[0][TIEMPO_SINC:iteraciones] # Se obtienen los datos después de TIEMPO_SINC
    x_descif = sol.y[3][TIEMPO_SINC:iteraciones]
    x_cif = np.resize(x_cif, nmax)
    x_descif = np.resize(x_descif, nmax)
    t = sol.t

    return t, x_maestro, y_maestro, z_maestro, x_esclavo, y_esclavo, z_esclavo, x_cif, x_descif

def aplicar_difusion(vector_inf, nmax):
    print("[DIFUSION] INICIANDO...")
    t_inicio_difusion = time.perf_counter()

    vector_logistico = mapa_logistico(LOGISTIC_PARAMS, nmax)

    vector_mezcla = np.floor(vector_logistico * nmax).astype(int)

    vector_temp = vector_inf.copy() 
    difusion = np.zeros(nmax) 
    contador = 0    
    for i in range(nmax):
        pos = vector_mezcla[i]
        if vector_temp[pos] != 260.0:
            difusion[contador] = vector_temp[pos]
            contador += 1
            vector_temp[pos] = 260.0

    for j in range(nmax):
        if contador >= nmax:
            break
        if vector_temp[j] != 260.0:
            difusion[contador] = vector_temp[j]
            contador += 1

    t_fin_difusion = time.perf_counter()
    tiempo_difusion = t_fin_difusion - t_inicio_difusion
    print(f"[DIFUSION] Tiempo de difusión: {tiempo_difusion:.4f} segundos")
    print("[DIFUSION] DIFUSIÓN COMPLETADA")
    
    return difusion, vector_logistico, tiempo_difusion

def aplicar_confusion(difusion, vector_logistico, nmax, rossler_params):    
    print("[CONFUSION] APLICANDO CONFUSIÓN...")
    t_inicio_confusion = time.perf_counter()

    # Integrar Rössler
    t_inicio_rossler = time.perf_counter()
    t, x_maestro, y_maestro, z_maestro, x_esclavo, y_esclavo, z_esclavo, x_cif, x_descif = integrar_rossler(rossler_params, nmax)  
    t_fin_rossler = time.perf_counter()
    tiempo_rossler = t_fin_rossler - t_inicio_rossler
    
    # 4. Aplicar confusión (solo después del tiempo de sincronización)
    vector_cifrado = np.zeros(nmax)
    vector_cifrado = difusion + vector_logistico + x_cif 
    print("[CONFUSION] Confusión aplicada correctamente")

    t_fin_confusion = time.perf_counter()
    tiempo_confusion = t_fin_confusion - t_inicio_confusion

    print(f"[CONFUSION] Tiempo de integración de Rössler: {tiempo_rossler:.4f} segundos")
    print(f"[CONFUSION] Tiempo total de confusión: {tiempo_confusion:.4f} segundos")

    return vector_cifrado, tiempo_rossler, tiempo_confusion, t, x_maestro, y_maestro, z_maestro, x_esclavo, y_esclavo, z_esclavo, x_cif, x_descif

def revertir_confusion(vector_cifrado, vector_logistico, x_descif):

    difusion = vector_cifrado - vector_logistico - x_descif

    return difusion

def revertir_difusion(difusion, vector_logistico, nmax, ancho, alto):
    # 1. Regenerar vector de mezcla (mismo que el maestro)
    vector_mezcla = np.floor(vector_logistico * nmax).astype(int)

    # 2. Inicializar estructuras para revertir
    vector_temp = np.full(nmax, 260.0)
    # vector_original = np.zeros(nmax)
    contador = 0

    # 3. Primera pasada: asignar a posiciones originales
    for i in range(nmax):
        pos = vector_mezcla[i]
        if vector_temp[pos] == 260.0:
            vector_temp[pos] = difusion[contador]
            contador += 1

    # 4. Segunda apsada: asignar posiciones restantes en orden
    for j in range(nmax):
        if contador >= nmax:
            break
        if vector_temp[j] == 260.0:
            vector_temp[j] = difusion[contador]
            contador += 1

    # 5. Reconstruir la imagen
    img_array = np.round(vector_temp * 255.0).clip(0, 255).astype(np.uint8)
    img_array = img_array.reshape((alto, ancho, 3))
    print("[DEBUG] img_array shape:", img_array.shape)
    return Image.fromarray(img_array)

def graficar_histogramas():
    img_original = Image.open(IMAGEN_ORIGINAL).convert("L")
    img_descifrada = Image.open(IMAGEN_DESCIFRADA).convert("L")

    pix_original = np.array(img_original).ravel()
    pix_descifrada = np.array(img_descifrada).ravel()

    bins = np.arange(257)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    axes[0].hist(pix_original, bins=bins, density=True, alpha=0.8, color="C0")
    axes[0].set_title("Imagen original")
    axes[0].set_xlabel("Intensidad (0 - 255)")
    axes[0].set_ylabel("Densidad de probabilidad")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(pix_descifrada, bins=bins, density=True, alpha=0.8, color="C1")
    axes[1].set_title("Imagen descifrada")
    axes[1].set_xlabel("Intensidad (0 - 255)")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Histogramas de intensidades: original vs descifrada (escala de grises)", y=1.02)
    plt.tight_layout()
    plt.savefig(RUTA_HISTOGRAMA_IMAGENES, dpi=300)

def cargar_imagen():
    imagen = Image.open(IMAGEN_ORIGINAL)
    vector_inf = np.array(imagen)
    alto, ancho, canales = vector_inf.shape
    vector_inf = vector_inf.flatten().astype(np.float64)/255.0
    nmax = vector_inf.size
    print("[CARGA] Imagen cargada y vectorizada correctamente")
    return imagen, vector_inf, ancho, alto, nmax

def difusion_confusion_imagenes(imagen, difusion, vector_cifrado, ancho, alto):
    """
    Genera y guarda la figura comparativa:
    original, después de difusión y después de confusión.
    """
    plt.figure(figsize=(15, 5))

    # Original
    plt.subplot(1, 3, 1)
    plt.imshow(imagen)
    plt.title("Original")
    plt.axis("off")

    # Después de difusión
    difusion_img = np.clip(difusion, 0.0, 1.0)
    difusion_img = np.round(difusion_img*255.0).astype(np.uint8).reshape((alto, ancho, 3))
    plt.subplot(1, 3, 2)
    plt.imshow(difusion_img)
    plt.title("Después de Difusión")
    plt.axis("off")

    # Después de confusión (pseudo-imagen)
    cifrado_norm = (vector_cifrado - np.min(vector_cifrado)) / (
        np.max(vector_cifrado) - np.min(vector_cifrado) + 1e-12
    )
    cifrado_img = np.clip(cifrado_norm, 0.0, 1.0)
    cifrado_img = np.round(cifrado_img*255.0).astype(np.uint8).reshape((alto, ancho, 3))
    plt.subplot(1, 3, 3)
    plt.imshow(cifrado_img)
    plt.title("Después de Confusión")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(RUTA_IMAGEN_CIFRADA)
    print(f"Resultados del proceso completo guardados en {RUTA_IMAGEN_CIFRADA}")

def graficar_series_vectores(difusion, vector_logistico, x_conf):
    plt.figure(figsize=(12, 8))

    # 1. Vector de difusión
    plt.subplot(3, 1, 1)
    plt.plot(difusion[:PUNTOS_EVAL])
    plt.ylabel("Difusión")
    plt.title("Vector de difusión")
    plt.grid(alpha=0.3)

    # 2. Vector logístico
    plt.subplot(3, 1, 2)
    plt.plot(vector_logistico[:PUNTOS_EVAL])
    plt.ylabel("Logístico")
    plt.title("Vector logístico")
    plt.grid(alpha=0.3)

    # 3. Serie x de Rössler
    plt.subplot(3, 1, 3)
    plt.plot(x_conf[:PUNTOS_EVAL])
    plt.xlabel("Índice")
    plt.ylabel("x (Rössler)")
    plt.title("Serie x de Rössler (confusión)")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(RUTA_SERIES_VECTORES)
    print(f"[GRAFICA] Series difusión/logístico/Rössler guardadas en {RUTA_SERIES_VECTORES}")

def graficar_vector_cifrado(vector_cifrado):
    plt.figure(figsize=(10, 4))
    plt.plot(vector_cifrado[:PUNTOS_EVAL])
    plt.xlabel("Índice")
    plt.ylabel("Valor")
    plt.title("Vector cifrado")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RUTA_VECTOR_CIFRADO_SERIE)
    print(f"[GRAFICA] Vector cifrado guardado en {RUTA_VECTOR_CIFRADO_SERIE}")

def graficar_dispersion(imagen, vector_cifrado):
    img_original = np.array(imagen).flatten().astype(np.float32)/255.0
    
    cifrado_normalizado = (vector_cifrado - np.min(vector_cifrado)) / (
        np.max(vector_cifrado) - np.min(vector_cifrado) + 1e-12
    )
    plt.figure(figsize=(6, 6))
    plt.scatter(img_original, cifrado_normalizado, s=1, alpha=0.3)
    plt.xlabel("Original (normalizada)")
    plt.ylabel("Cifrada (normalizada)")
    plt.title("Diagrama de dispersión: imagen original vs cifrada")
    plt.tight_layout()
    plt.savefig(str(RUTA_DISPERSION))
    print(f"[GRAFICA] Dispersión guardada en {RUTA_DISPERSION}")

def distancia_hamming(imagen_original, imagen_descifrada):
    img_original = np.array(imagen_original.convert("L"), dtype=np.uint8)
    img_descifrada = np.array(imagen_descifrada.convert("L"), dtype=np.uint8)
    
    img_original = img_original.flatten()
    img_descifrada = img_descifrada.flatten()
    
    n_bytes = min(img_original.size, img_descifrada.size)
    img_original = img_original[:n_bytes]
    img_descifrada = img_descifrada[:n_bytes]
    
    img_original = np.unpackbits(img_original)
    img_descifrada = np.unpackbits(img_descifrada)
    
    n_bits = min(img_original.size, img_descifrada.size)
    img_original = img_original[:n_bits]
    img_descifrada = img_descifrada[:n_bits]
    
    diff_bits = img_original ^ img_descifrada
    hamming_abs = int(diff_bits.sum())
    hamming_norm = hamming_abs / n_bits if n_bits > 0 else 0.0
    
    print(f"[HAMMING] Distancia de Hamming absoluta: {hamming_abs}")
    print(f"[HAMMING] Distancia de Hamming normalizada: {hamming_norm:.6f}")

    return hamming_abs, hamming_norm
    

def registrar_tiempos(tiempo_difusion, tiempo_rossler, tiempo_confusion, tiempo_programa, tiempo_descifrado):
    """
    Se registran las métricas de tiempo para cada proceso en un archivo CSV
    """
    registro = {
        "timestamp": time.strftime("%m-%d %H:%M:%S"),
        "tiempo_difusion_segundos": tiempo_difusion,
        "tiempo_rossler_segundos": tiempo_rossler,
        "tiempo_confusion_segundos": tiempo_confusion,
        "tiempo_programa_segundos": tiempo_programa,
        "tiempo_descifrado": tiempo_descifrado,
    }

    df = pd.DataFrame([registro])
    archivo = RUTA_TIMINGS.exists()
    df.to_csv(RUTA_TIMINGS, mode='a', index = False, header = not archivo)
    print(f"[TIEMPOS] Tiempos registrados en {RUTA_TIMINGS}")

def graficar_series(t, x_maestro, y_maestro, z_maestro, x_esclavo, y_esclavo, z_esclavo):
    # x(t)
    plt.figure(figsize=(10,6))
    plt.plot(t, x_maestro, label="Maestro", color='black', linewidth=1.3)
    plt.plot(t, x_esclavo, label="Esclavo", color="orange", linewidth=0.8, linestyle="dotted", alpha = 0.8)
    plt.xlabel("Tiempo")
    plt.ylabel("x(t)")
    plt.title("Serie temporal de x(t) - Maestro vs Esclavo")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RUTA_SERIE_X, dpi = 400)
    print(f"[GRAFICA] Series x(t) guardadas en: {RUTA_SERIE_X}")

    # y(t)
    plt.figure(figsize=(10,6))
    plt.plot(t, y_maestro, label="Maestro", color='black', linewidth=1.3)
    plt.plot(t, y_esclavo, label="Esclavo", color="orange", linewidth=0.8, linestyle="dotted", alpha = 0.8)
    plt.xlabel("Tiempo")
    plt.ylabel("y(t)")
    plt.title("Serie temporal de y(t) - Maestro vs Esclavo")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RUTA_SERIE_Y, dpi = 400)
    print(f"[GRAFICA] Series y(t) guardadas en: {RUTA_SERIE_Y}")

    # z(t)
    plt.figure(figsize=(10,6))
    plt.plot(t, z_maestro, label="Maestro", color='black', linewidth=1.3)
    plt.plot(t, z_esclavo, label="Esclavo", color="orange", linewidth=0.8, linestyle="dotted", alpha = 0.8)
    plt.xlabel("Tiempo")
    plt.ylabel("z(t)")
    plt.title("Serie temporal de z(t) - Maestro vs Esclavo")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RUTA_SERIE_Z, dpi = 400)
    print(f"[GRAFICA] Series z(t) guardadas en: {RUTA_SERIE_Z}")

def graficar_errores(t, error_x, error_y, error_z):
    # Error x
    plt.figure(figsize=(10,6))
    plt.plot(t, error_x, color='purple', linewidth=1)
    plt.xlabel("Tiempo")
    plt.ylabel("Error en x(t)")
    plt.title("Error de sincronización en x(t)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RUTA_ERROR_X, dpi = 400)
    print(f"[GRAFICA] Error en x(t) guardado en: {RUTA_ERROR_X}")

    # Error y
    plt.figure(figsize=(10,6))
    plt.plot(t, error_y, color='purple', linewidth=1)
    plt.xlabel("Tiempo")
    plt.ylabel("Error en y(t)")
    plt.title("Error de sincronización en y(t)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RUTA_ERROR_Y, dpi = 400)
    print(f"[GRAFICA] Error en y(t) guardado en: {RUTA_ERROR_Y}")

    # Error z
    plt.figure(figsize=(10,6))
    plt.plot(t, error_z, color='purple', linewidth=1)
    plt.xlabel("Tiempo")
    plt.ylabel("Error en z(t)")
    plt.title("Error de sincronización en z(t)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RUTA_ERROR_Z, dpi = 400)
    print(f"[GRAFICA] Error en z(t) guardado en: {RUTA_ERROR_Z}")

def graficar_dispersion(x_maestro, x_esclavo, y_maestro, y_esclavo, z_maestro, z_esclavo):
    # Dispersión x
    plt.figure(figsize=(6,6))
    plt.scatter(x_maestro, x_esclavo, color='green', s=1, alpha=0.5)
    plt.xlabel("x Maestro")
    plt.ylabel("x Esclavo")
    plt.title("Diagrama de dispersión x Maesto vs x Esclavo")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RUTA_DISPERSION_X, dpi = 400)
    print(f"[GRAFICA] Diagrama de dispersión x guardado en: {RUTA_DISPERSION_X}")

    # Dispersión y
    plt.figure(figsize=(6,6))
    plt.scatter(y_maestro, y_esclavo, color='green', s=1, alpha=0.5)
    plt.xlabel("y Maestro")
    plt.ylabel("y Esclavo")
    plt.title("Diagrama de dispersión y Maesto vs y Esclavo")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RUTA_DISPERSION_Y, dpi = 400)
    print(f"[GRAFICA] Diagrama de dispersión y guardado en: {RUTA_DISPERSION_Y}")

    # Dispersión z
    plt.figure(figsize=(6,6))
    plt.scatter(z_maestro, z_esclavo, color='green', s=1, alpha=0.5)
    plt.xlabel("z Maestro")
    plt.ylabel("z Esclavo")
    plt.title("Diagrama de dispersión z Maestro vs z Esclavo")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RUTA_DISPERSION_Z, dpi = 400)
    print(f"[GRAFICA] Diagrama de dispersión z guardado en: {RUTA_DISPERSION_Z}")

def guardar_errores_csv(t, error_x, error_y, error_z):
    df_errores = pd.DataFrame({
        "Tiempo": t,
        "Error_x": error_x,
        "Error_y": error_y,
        "Error_z": error_z
    })

    df_errores.to_csv(RUTA_ERRORES, index=False)
    print(f"[CSV] Errores de sincronizacion guardados en: {RUTA_ERRORES}")

def coeficiente_correlacion_imagenes(imagen_original, imagen_descifrada):
    # Convertir a escala de grises y a float
    img_orig = np.array(imagen_original.convert("L"), dtype=np.float64)
    img_decr = np.array(imagen_descifrada.convert("L"), dtype=np.float64)

    # Aplanar
    orig_flat = img_orig.flatten()
    decr_flat = img_decr.flatten()

    # Emparejar longitudes
    n = min(orig_flat.size, decr_flat.size)
    orig_flat = orig_flat[:n]
    decr_flat = decr_flat[:n]

    # Evitar problemas de desviación estándar cero o muy pocos datos
    if n < 2 or np.std(orig_flat) == 0 or np.std(decr_flat) == 0:
        correlacion = 0.0
    else:
        corr_matrix = np.corrcoef(orig_flat, decr_flat)
        correlacion = float(corr_matrix[0, 1])

    print(f"[CORRELACION] Coeficiente de correlación: {correlacion:.6f}")
    return correlacion

def experimento_hamming_vs_a(
    rossler_params,
    nmax,
    vector_cifrado,
    ancho,
    alto,
    vector_logistico
):
    img_original = Image.open(IMAGEN_ORIGINAL)
    valores_a = np.arange(0.0, 1.0 + 1e-6, 0.01)
    resultados = []
    
    print("[HAMMING] Iniciando experimento de distancia de Hamming vs 'a' de Rössler...")
    
    for a_val in valores_a:
        print(f"[HAMMING] Evaluando a = {a_val:.2f}...")
        params_test = dict(rossler_params)
        params_test['a'] = float(a_val)
        
        # Sincronizar con el nuevo a
        _, _, _, _, _, _, _, _, x_descif = integrar_rossler(params_test, nmax)
        # Descifrar
        difusion = revertir_confusion(vector_cifrado, vector_logistico, x_descif)
        imagen_descifrada = revertir_difusion(difusion, vector_logistico, nmax, ancho, alto)

        # Calcular la distancia hamming
        hamming_abs, hamming_norm = distancia_hamming(img_original, imagen_descifrada)
        resultados.append({
            "a": a_val,
            "hamming_abs": hamming_abs,
            "hamming_norm": hamming_norm
        })
        
        df_resultados = pd.DataFrame(resultados)
        df_resultados.to_csv(RUTA_DISTANCIA_HAMMING_CSV_A, index=False)
        print(f"[HAMMING] Resultados guardados en {RUTA_DISTANCIA_HAMMING_CSV_A}")

        plt.figure(figsize=(10, 6))
        plt.plot(df_resultados['a'], df_resultados["hamming_abs"], marker = "o", linewidth = 0.8)
        plt.xlabel("a")
        plt.ylabel("Distancia de Hamming")
        plt.title("Distancia de hamming imagen original vs descifrada")
        plt.grid(True, alpha = 0.3)
        plt.tight_layout()
        plt.savefig(RUTA_DISTANCIA_HAMMING_A, dpi=300)
        plt.close()
        print(f"[HAMMING] Gráfico guardado en {RUTA_DISTANCIA_HAMMING_A}")

def experimento_hamming_vs_logistico(
    LOGISTIC_PARAMS,
    vector_cifrado,
    ancho,
    alto,
    nmax,
    x_sinc,
    vector_logistico_original,
    img_original
):
    aLog_original = float(LOGISTIC_PARAMS["aLog"])

    base_aLog = np.arange(3.97, 4.0 + 1e-6, 0.001)
    valores_aLog = list(base_aLog)

    if not any(np.isclose(aLog_original, v, atol=1e-15) for v in valores_aLog):
        valores_aLog.append(aLog_original)

    valores_aLog = sorted(valores_aLog)
    resultados_aLog = []

    print("[HAMMING-LOG] Iniciando experimento Hamming vs aLog del mapa logístico...")

    for a_val in valores_aLog:
        es_original = np.isclose(a_val, aLog_original, atol=1e-15)
        if es_original:
            print(f"[HAMMING-LOG] Evaluando aLog = {a_val:.15f} (valor ORIGINAL del maestro, reutilizando vector_logistico_original)...")
            vector_logistico_test = vector_logistico_original
        else:
            print(f"[HAMMING-LOG] Evaluando aLog = {a_val:.15f}...")

            params_test = dict(LOGISTIC_PARAMS)
            params_test['aLog'] = float(a_val)

            vector_logistico_test = mapa_logistico(params_test, nmax)

        difusion = revertir_confusion(vector_cifrado, vector_logistico_test, x_sinc)
        imagen_descifrada = revertir_difusion(difusion, vector_logistico_test, nmax, ancho, alto)

        nombre_img = f"hamming_aLog_{a_val:.3f}".replace(".", "_") + ".png"
        ruta_img = CARPETA_HAMMING_LOGISTIC / nombre_img
        imagen_descifrada.save(ruta_img)
        print(f"[HAMMING-LOG] Imagen descifrada guardada en {ruta_img}")

        hamming_abs, hamming_norm = distancia_hamming(img_original, imagen_descifrada)

        resultados_aLog.append({
            "aLog": a_val,
            "aLog_es_original": int(es_original),
            "hamming_abs": hamming_abs,
            "hamming_norm": hamming_norm
        })

    df_aLog = pd.DataFrame(resultados_aLog)
    df_aLog.to_csv(RUTA_DISTANCIA_HAMMING_CSV_A_LOG, index=False)
    print(f"[HAMMING-LOG] Resultados Hamming vs aLog guardados en {RUTA_DISTANCIA_HAMMING_CSV_A_LOG}")

    plt.figure(figsize=(10, 6))
    plt.plot(df_aLog["aLog"], df_aLog["hamming_abs"], marker="o", linewidth=0.8)
    plt.xlabel("aLog")
    plt.ylabel("Distancia de Hamming absoluta")
    plt.title("Distancia de Hamming imagen original vs descifrada\n(barrido en aLog del mapa logístico)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RUTA_DISTANCIA_HAMMING_A_LOG, dpi=300)
    plt.close()
    print(f"[HAMMING-LOG] Gráfico Hamming vs aLog guardado en {RUTA_DISTANCIA_HAMMING_A_LOG}")

    x0_original = float(LOGISTIC_PARAMS["x0_log"])
    base_x0 = np.arange(0.3, 0.5 + 1e-6, 0.01)
    valores_x0 = list(base_x0)
    if not any(np.isclose(x0_original, v, atol=1e-15) for v in valores_x0):
        valores_x0.append(x0_original)
    valores_x0 = sorted(valores_x0)

    resultados_x0 = []

    print("[HAMMING-LOG] Iniciando experimento Hamming vs x0_log del mapa logístico...")

    for x0_val in valores_x0:
        es_original_x0 = np.isclose(x0_val, x0_original, atol=1e-15)
        if es_original_x0:
            print(f"[HAMMING-LOG] Evaluando x0_log = {x0_val:.15f} (valor ORIGINAL del maestro)...")
        else:
            print(f"[HAMMING-LOG] Evaluando x0_log = {x0_val:.15f}...")

        params_test = dict(LOGISTIC_PARAMS)
        params_test["x0_log"] = float(x0_val)

        vector_logistico_test = mapa_logistico(params_test, nmax)
        difusion = revertir_confusion(vector_cifrado, vector_logistico_test, x_sinc)
        imagen_descifrada = revertir_difusion(difusion, vector_logistico_test, nmax, ancho, alto)

        nombre_img = f"hamming_x0_log_{x0_val:.3f}".replace(".", "_") + ".png"
        ruta_img = CARPETA_HAMMING_LOGISTIC / nombre_img
        imagen_descifrada.save(ruta_img)
        print(f"[HAMMING-LOG] Imagen descifrada guardada en {ruta_img}")

        hamming_abs, hamming_norm = distancia_hamming(img_original, imagen_descifrada)

        resultados_x0.append({
            "x0_log": x0_val,
            "x0_log_es_original": int(es_original_x0),
            "hamming_abs": hamming_abs,
            "hamming_norm": hamming_norm
        })

    df_x0 = pd.DataFrame(resultados_x0)
    df_x0.to_csv(RUTA_DISTANCIA_HAMMING_CSV_X0_LOG, index=False)
    print(f"[HAMMING-LOG] Resultados Hamming vs x0_log guardados en {RUTA_DISTANCIA_HAMMING_CSV_X0_LOG}")

    plt.figure(figsize=(10, 6))
    plt.plot(df_x0["x0_log"], df_x0["hamming_abs"], marker="o", linewidth=0.8)
    plt.xlabel("x0_log")
    plt.ylabel("Distancia de Hamming absoluta")
    plt.title("Distancia de Hamming imagen original vs descifrada\n(barrido en x0_log del mapa logístico)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RUTA_DISTANCIA_HAMMING_X0_LOG, dpi=300)
    plt.close()
    print(f"[HAMMING-LOG] Gráfico Hamming vs x0_log guardado en {RUTA_DISTANCIA_HAMMING_X0_LOG}")


def main():
    # =============================== PROCESO DE CIFRADO ===============================

    inicio_programa = time.perf_counter()
    imagen, vector_inf, ancho, alto, nmax = cargar_imagen()
    print(f"[CARGA] Medidas de la imagen: Ancho={ancho}, Alto={alto}, Canales=3, Total píxeles={nmax//3}, Total valores={nmax}")

    # 2. Aplicar difusión
    difusion, vector_logistico, tiempo_difusion = aplicar_difusion(vector_inf, nmax)

    # 3. Aplicar confusión
    vector_cifrado, tiempo_rossler, tiempo_confusion, t, x_maestro, y_maestro, z_maestro, x_esclavo, y_esclavo, z_esclavo, x_cif, x_descif = aplicar_confusion(difusion, vector_logistico, nmax, ROSSLER_PARAMS)
    
    
    # =============================== PROCESO DE DESCIFRADO ===============================
    print("[DESCIFRADO] INICIANDO PROCESO DE DESCIFRADO...")
    t_inicio_descifrado = time.perf_counter()
    # Revertir confusion
    difusion = revertir_confusion(vector_cifrado, vector_logistico, x_descif)

    # Revertir difusion
    imagen_descifrada = revertir_difusion(difusion, vector_logistico, nmax, ancho, alto)
    imagen_descifrada.save(IMAGEN_DESCIFRADA)
    t_fin_descifrado = time.perf_counter()
    tiempo_descifrado = t_fin_descifrado - t_inicio_descifrado
    fin_programa = time.perf_counter()
    tiempo_programa = fin_programa - inicio_programa
    print(f"[PROGRAMA] Tiempo total del programa: {tiempo_programa:.4f} segundos")
    registrar_tiempos(
        tiempo_difusion,
        tiempo_rossler,
        tiempo_confusion,
        tiempo_programa,
        tiempo_descifrado
    )

    # Calcular errores de sincronizacion
    error_x = np.abs(x_maestro - x_esclavo)
    error_y = np.abs(y_maestro - y_esclavo)
    error_z = np.abs(z_maestro - z_esclavo)

    img_original = Image.open(IMAGEN_ORIGINAL)
    img_descifrada = Image.open(IMAGEN_DESCIFRADA)

    # Graficar difusion y confusion como imagenes
    difusion_confusion_imagenes(imagen, difusion, vector_cifrado, ancho, alto)

    # Graficar series de difusion, logistico y rossler
    graficar_series_vectores(difusion, vector_logistico, x_cif)

    # Graficar vector cifrado
    graficar_vector_cifrado(vector_cifrado)

    # Graficar distancia Hamming
    distancia_hamming(img_original, img_descifrada)

    # Graficar series temporales
    graficar_series(
        t, x_maestro, y_maestro, z_maestro, x_esclavo, y_esclavo, z_esclavo
    )
    # Graficar errores
    graficar_errores(t, error_x, error_y, error_z)

    # Graficar diagramas de dispersion
    graficar_dispersion(x_maestro, x_esclavo, y_maestro, y_esclavo, z_maestro, z_esclavo)

    # Guardar errores en CSV
    guardar_errores_csv(t, error_x, error_y, error_z)

    # Calcular errores medios despues de TIEMPO_SINC
    errores_medios = pd.DataFrame([{
        "Error_medio_x": np.mean(error_x[TIEMPO_SINC:]),
        "Error_medio_y": np.mean(error_y[TIEMPO_SINC:]),
        "Error_medio_z": np.mean(error_z[TIEMPO_SINC:])
    }])

    # Guardar errores medios en CSV
    errores_medios.to_csv(RUTA_ERRORES_MEDIOS, index=False)
    print(f"[CSV] Errores medios de sincronizacion guardados en: {RUTA_ERRORES_MEDIOS}")

    # Distancia hamming entre imagen original y descifrada
    
    distancia_hamming(img_original, img_descifrada)
    # Coeficiente de correlacion entre imagen original y descifrada
    correlacion = coeficiente_correlacion_imagenes(img_original, img_descifrada)

    # Guardar correlación en CSV
    df_corr_base = pd.DataFrame([{
        "descripcion": "correlacion_original_vs_descifrada",
        "correlacion": correlacion
    }])
    df_corr_base.to_csv(
        RUTA_CORRELACION,
        index=False,
        mode="a",
        header=not RUTA_CORRELACION.exists()
    )
    print(f"[CORR] Correlación base guardada en {RUTA_CORRELACION}")

    # Experimento distancia Hamming vs 'a' de Rössler
    # experimento_hamming_vs_a(
    #     ROSSLER_PARAMS,
    #     nmax,
    #     vector_cifrado,
    #     ancho,
    #     alto,
    #     vector_logistico
    # )
    # Experimento distancia Hamming vs 'aLog' y 'x0_log' del mapa logístico
    # experimento_hamming_vs_logistico(
    #     LOGISTIC_PARAMS,
    #     vector_cifrado,
    #     ancho,
    #     alto,
    #     nmax,
    #     x_descif,
    #     vector_logistico,
    #     img_original
    # )


if __name__ == "__main__":
    main()
    print("[PROGRAMA] EJECUCIÓN COMPLETADA")