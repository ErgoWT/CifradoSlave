import json 
import time 
import ssl
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import paho.mqtt.client as mqtt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from PIL import Image



# ========== CONFIGURACION MQTT ==========
BROKER = "raspberrypiJED.local"
PORT = 8883
USERNAME = "usuario1"
PASSWORD = "qwerty123"
TOPIC_KEYS = "chaoskeystream/keys"
TOPIC_DATA = "chaoskeystream/data"
CA_CERT_PATH = "/home/tunchi/CifradoSlave/certs/ca.crt"
QOS = 1
IMG_SCALE = 100
# ========== PARAMETROS GLOBALES PARA ALMACENAMIENTO ==========
RECEIVED_KEYS = None
RECEIVED_DATA = None

# ========== PARAMETROS DE ROSSLER ==========
H = 0.01
K = 2.0
X0 = [1.0, 1.0, 1.0]
# Parámetros característicos de prueba, estos son distintos a los del maestro
A = 0.3
B = 0.201
C = 6.2
ALOG = 3.9899
X0_LOG = 0.4000001

# ========== RUTAS Y ARCHIVOS ==========
CARPETA_RESULTADOS = Path("Resultados_TLS_KEYSTREAM")
CARPETA_HAMMING_LOGISTIC = CARPETA_RESULTADOS / "Hamming_Logistic"
CARPETA_HAMMING_LOGISTIC.mkdir(parents=True, exist_ok=True)

RUTA_SINCRONIZACION_X = CARPETA_RESULTADOS / "sincronizacion_x_resultados.png"
RUTA_SINCRONIZACION_Y = CARPETA_RESULTADOS / "sincronizacion_y_resultados.png"
RUTA_SINCRONIZACION_Z = CARPETA_RESULTADOS / "sincronizacion_z_resultados.png"

RUTA_ERROR_X_GRAFICA = CARPETA_RESULTADOS / "error_sincronizacion_x.png"
RUTA_ERROR_Y_GRAFICA = CARPETA_RESULTADOS / "error_sincronizacion_y.png"
RUTA_ERROR_Z_GRAFICA = CARPETA_RESULTADOS / "error_sincronizacion_z.png"

RUTA_DISPERSION_X = CARPETA_RESULTADOS / "dispersion_error_x.png"
RUTA_DISPERSION_Y = CARPETA_RESULTADOS / "dispersion_error_y.png"
RUTA_DISPERSION_Z = CARPETA_RESULTADOS / "dispersion_error_z.png"
RUTA_DISPERSION_PIXELES = CARPETA_RESULTADOS / "dispersion_pixeles.png"

RUTA_ERRORES = CARPETA_RESULTADOS / "errores_sincronizacion.csv"

RUTA_IMAGEN_ORIGINAL = Path("Prueba2.jpg")
RUTA_HISTOGRAMA_IMAGENES = CARPETA_RESULTADOS / "histogramas.png"

RUTA_IMAGEN_DESCIFRADA = CARPETA_RESULTADOS / "imagen_descifrada.png"
RUTA_TIMINGS = CARPETA_RESULTADOS / "tiempo_procesos_esclavo.csv"

RUTA_CORRELACION_BASE = CARPETA_RESULTADOS / "correlacion_original_vs_descifrada.csv"

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

PUNTOS_EVAL = 40000

def on_connect(client, userdata, flags, rc):
    print(f"[MQTT-ESCLAVO] Conectado al broker {BROKER}: {PORT} con TLS (rc={rc})")
    if rc == 0:
        client.subscribe(TOPIC_KEYS, qos=QOS)
        client.subscribe(TOPIC_DATA, qos=QOS)
        print("[MQTT-ESCLAVO] Suscrito a los topics para recibir datos y keys")
    else:
        print(f"[MQTT-ESCLAVO] Error al conectar al broker MQTT (rc={rc})")

def on_message_data(client, userdata, msg):
    global RECEIVED_DATA
    RECEIVED_DATA = json.loads(msg.payload.decode())
    print(f"[MQTT-ESCLAVO] Datos recibidos en {msg.topic}")

def on_message_keys(client, userdata, msg):
    global RECEIVED_KEYS
    RECEIVED_KEYS = json.loads(msg.payload.decode())
    print(f"[MQTT-ESCLAVO] Keys recibidas en {msg.topic}")

# ========== FUNCION DE ROSSLER ESCLAVO ==========
def rossler_esclavo(t, state, y_master_interp, a, b, c, k):
    x_s, y_s, z_s = state
    y_m = y_master_interp(t)
    dxdt = -y_s - z_s
    dydt = x_s + a * y_s + k * (y_m - y_s)
    dzdt = b + z_s * (x_s - c)
    return [dxdt, dydt, dzdt]

# ========== FUNCION DEL MAPA LOGISTICO ==========
def mapa_logistico(LOGISTIC_PARAMS, nmax):

    a_log = LOGISTIC_PARAMS['aLog']
    x0_log = LOGISTIC_PARAMS['x0_log']

    vector_logistico = np.zeros(nmax)
    x = x0_log
    for i in range(nmax):
        x = a_log * x * (1 - x)
        vector_logistico[i] = x

    return vector_logistico

# ========== FUNCION PARA REVERTIR LA CONFUSIÓN ==========
def sincronizacion(y_maestro, times, ROSSLER_PARAMS, time_sinc, keystream, nmax):

    iteraciones = time_sinc + keystream
    t_span = (0, iteraciones*H)
    t_eval = np.linspace(0, iteraciones*H, iteraciones)

    y_maestro_interp = interp1d(
        times,
        y_maestro,
        kind='cubic',
        fill_value="extrapolate"
    )    

    sol_esclavo = solve_ivp(
        fun = rossler_esclavo,
        t_span = t_span,
        y0 = X0,
        t_eval = t_eval,
        args = (y_maestro_interp, 
                ROSSLER_PARAMS['a'], 
                ROSSLER_PARAMS['b'], 
                ROSSLER_PARAMS['c'], 
                K),
        rtol = 1e-6,
        atol = 1e-8,
        method='RK23'
    )
    
    x_esclavo = sol_esclavo.y[0]
    y_esclavo = sol_esclavo.y[1]
    z_esclavo = sol_esclavo.y[2]
    t_esclavo = sol_esclavo.t
    x_sinc = sol_esclavo.y[0][time_sinc:]
    x_sinc = np.resize(x_sinc, nmax)
    return x_esclavo, y_esclavo, z_esclavo, t_esclavo, x_sinc

def revertir_confusion(vector_cifrado, vector_logistico, x_sinc):

    difusion = vector_cifrado - vector_logistico - x_sinc

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
    return Image.fromarray(img_array)

def extraer_parametros(receivedKeys, receivedData):
    # Llaves del maestro
    ROSSLER_PARAMS = receivedKeys['ROSSLER_PARAMS']
    LOGISTIC_PARAMS = receivedKeys['LOGISTIC_PARAMS']

    vector_cifrado = np.array(receivedData['vector_cifrado'])
    x_maestro = np.array(receivedData['x_maestro'])
    y_maestro = np.array(receivedData['y_maestro'])
    z_maestro = np.array(receivedData['z_maestro'])
    t_maestro = np.array(receivedData['t_maestro'])
    ancho = int(receivedData['ancho'])
    alto = int(receivedData['alto'])
    nmax = int(receivedData['nmax'])
    tiempo_sinc = int(receivedData['tiempo_sinc'])
    keystream = int(receivedData['KEYSTREAM'])

    return (
        ROSSLER_PARAMS,
        LOGISTIC_PARAMS,
        vector_cifrado,
        x_maestro,
        y_maestro,
        z_maestro,
        t_maestro,
        ancho,
        alto,
        nmax,
        tiempo_sinc,
        keystream
    )
    

def grafica_serie_temporal(t_maestro, x_maestro, y_maestro, z_maestro, t_esclavo, x_esclavo, y_esclavo, z_esclavo):
    # Series temporales en X
    plt.figure(figsize=(10, 6))
    plt.plot(t_maestro, x_maestro, label='Maestro x(t)', linewidth = 1.4, color='red') # Señal del maestro
    plt.plot(t_esclavo, x_esclavo, label='Esclavo x(t)', linewidth = 1.4, linestyle='--', alpha=0.9)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("x(t)")
    plt.title("Serie Temporal de x(t): Maestro vs Esclavo")
    plt.legend(loc="upper right", frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RUTA_SINCRONIZACION_X, dpi=300)
    
    # Imprimir longitud total de times y t_esclavo
    print(f"[GRAFICAS] Longitud times (maestro): {len(t_maestro)}, Longitud t_esclavo (esclavo): {len(t_esclavo)}")

    # Series temporales en Y
    plt.figure(figsize=(10, 6))
    plt.plot(t_maestro, y_maestro, label='Maestro y(t)', linewidth = 1.4, color='red') # Señal del maestro
    plt.plot(t_esclavo, y_esclavo, label='Esclavo y(t)', linewidth = 1.4, linestyle='--', alpha=0.9)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("y(t)")
    plt.title("Serie Temporal de y(t): Maestro vs Esclavo")
    plt.legend(loc="upper right", frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RUTA_SINCRONIZACION_Y, dpi=300)

    # Series temporales en Z
    plt.figure(figsize=(10, 6))
    plt.plot(t_maestro, z_maestro, label='Maestro z(t)', linewidth = 1.4, color='red') # Señal del maestro
    plt.plot(t_esclavo, z_esclavo, label='Esclavo z(t)', linewidth = 1.4, linestyle='--', alpha=0.9)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("z(t)")
    plt.title("Serie Temporal de z(t): Maestro vs Esclavo")
    plt.legend(loc="upper right", frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RUTA_SINCRONIZACION_Z, dpi=300)

def graficar_errores_dispersion(t_maestro, x_maestro, y_maestro, z_maestro, t_esclavo, x_esclavo, y_esclavo, z_esclavo, error_x, error_y, error_z):

    # ==================== GRÁFICA ERROR VS TIEMPO X ====================
    plt.figure(figsize=(10, 6))
    plt.plot(t_maestro, error_x, linewidth=1.4)
    plt.xlabel("Tiempo")
    plt.ylabel("|x_m(t) - x_s(t)|")
    plt.title("Error en la serie x(t) entre maestro y esclavo")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RUTA_ERROR_X_GRAFICA, dpi=300)
    print(f"[GRAFICAS] Error |x_m - x_s| vs tiempo guardado en {RUTA_ERROR_X_GRAFICA}")
    # ==================== GRÁFICA ERROR VS TIEMPO Y ====================
    plt.figure(figsize=(10, 6))
    plt.plot(t_maestro, error_y, linewidth=1.4)
    plt.xlabel("Tiempo")
    plt.ylabel("|y_m(t) - y_s(t)|")
    plt.title("Error en la serie y(t) entre maestro y esclavo")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RUTA_ERROR_Y_GRAFICA, dpi=300)
    print(f"[GRAFICAS] Error |y_m - y_s| vs tiempo guardado en {RUTA_ERROR_Y_GRAFICA}")
    # ==================== GRÁFICA ERROR VS TIEMPO Z ====================
    plt.figure(figsize=(10, 6))
    # Aquí se gráfican los datos desde 2000 hasta 40000
    plt.plot(t_maestro, error_z, linewidth=1.4)
    plt.xlabel("Tiempo")
    plt.ylabel("|z_m(t) - z_s(t)|")
    plt.title("Error en la serie z(t) entre maestro y esclavo")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RUTA_ERROR_Z_GRAFICA, dpi=300)
    print(f"[GRAFICAS] Error |z_m - z_s| vs tiempo guardado en {RUTA_ERROR_Z_GRAFICA}")

    
    # ==================== DIAGRAMA DE DISPERSIÓN x_m vs x_s ====================
    plt.figure(figsize=(10, 6))
    plt.scatter(x_maestro, x_esclavo, s=8, alpha=0.6,)
    min_val = min(x_maestro.min(), x_esclavo.min())
    max_val = max(x_maestro.max(), x_esclavo.max())
    plt.plot([min_val, max_val], [min_val, max_val], linewidth=1.0, linestyle="--")
    plt.xlabel("x_m(t)  (maestro)")
    plt.ylabel("x_s(t)  (esclavo)")
    plt.title("Diagrama de dispersión x_maestro vs x_esclavo")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RUTA_DISPERSION_X, dpi=300)
    # ==================== DIAGRAMA DE DISPERSIÓN y_m vs y_s ====================
    plt.figure(figsize=(10, 6))
    plt.scatter(y_maestro, y_esclavo, s=8, alpha=0.6,)
    min_val = min(y_maestro.min(), y_esclavo.min())
    max_val = max(y_maestro.max(), y_esclavo.max())
    plt.plot([min_val, max_val], [min_val, max_val], linewidth=1.0, linestyle="--")
    plt.xlabel("y_m(t)  (maestro)")
    plt.ylabel("y_s(t)  (esclavo)")
    plt.title("Diagrama de dispersión y_maestro vs y_esclavo")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RUTA_DISPERSION_Y, dpi=300)
    # ==================== DIAGRAMA DE DISPERSIÓN z_m vs z_s ====================
    plt.figure(figsize=(10, 6))
    plt.scatter(z_maestro, z_esclavo, s=8, alpha=0.6,)
    min_val = min(z_maestro.min(), z_esclavo.min())
    max_val = max(z_maestro.max(), z_esclavo.max())
    plt.plot([min_val, max_val], [min_val, max_val], linewidth=1.0, linestyle="--")
    plt.xlabel("z_m(t)  (maestro)")
    plt.ylabel("z_s(t)  (esclavo)")
    plt.title("Diagrama de dispersión z_maestro vs z_esclavo")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RUTA_DISPERSION_Z, dpi=300)

def graficar_histogramas():
    img_original = Image.open(RUTA_IMAGEN_ORIGINAL).convert("L")
    img_descifrada = Image.open(RUTA_IMAGEN_DESCIFRADA).convert("L")

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

def graficar_dispersion_pixeles():
    img_original = Image.open(RUTA_IMAGEN_ORIGINAL).convert("L")
    img_descifrada = Image.open(RUTA_IMAGEN_DESCIFRADA).convert("L")

    plt.figure(figsize=(10, 6))
    plt.scatter(img_original, img_descifrada, s=5, alpha=0.4)
    min_val = 0
    max_val = 255
    plt.plot([min_val, max_val], [min_val, max_val], linewidth=1.0, linestyle="--")
    plt.xlabel("Valor de píxel imagen original")
    plt.ylabel("Valor de píxel imagen descifrada")
    plt.title("Dispersión de valores de píxel: original vs descifrada")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RUTA_DISPERSION_PIXELES, dpi=300)

def coeficiente_correlacion_imagenes(imagen_original, imagen_descifrada):
    """
    Calcula el coeficiente de correlación de Pearson entre la imagen original
    y la imagen descifrada (trabajando en escala de grises).
    """
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
        corr = 0.0
    else:
        corr_matrix = np.corrcoef(orig_flat, decr_flat)
        corr = float(corr_matrix[0, 1])

    print(f"[CORR] Coeficiente de correlación: {corr:.6f}")
    return corr

def guardar_errores(t_esclavo, error_x, error_y, error_z):
    df_error = pd.DataFrame({
        'Tiempo': t_esclavo,
        'Error_x': error_x,
        'Error_y': error_y,
        'Error_z': error_z
    })
    df_error.to_csv(RUTA_ERRORES, index=False)
    print(f"[SINCRONIZACIÓN] Error de sincronización guardado en {RUTA_ERRORES}")

def tiempos_procesos(tiempo_mqtt, tiempo_sincronizacion, tiempo_descifrado, tiempo_total):
    df_tiempos = pd.DataFrame([{
        "tiempo_mqtt": tiempo_mqtt,
        "tiempo_sincronizacion": tiempo_sincronizacion,
        "tiempo_descifrado": tiempo_descifrado,
        "tiempo_total": tiempo_total
    }])

    df_tiempos.to_csv(RUTA_TIMINGS, index=False, mode="a", header=not RUTA_TIMINGS.exists())

    print(f"[TIEMPOS] Tiempos de proceso guardados en {RUTA_TIMINGS}")

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

def experimento_hamming_vs_a(
    ROSSLER_PARAMS,
    LOGISTIC_PARAMS,
    y_sinc,
    times,
    time_sinc,
    nmax,
    keystream,
    vector_cifrado,
    ancho,
    alto,
    vector_logistico
):
    img_original = Image.open(RUTA_IMAGEN_ORIGINAL)
    valores_a = np.arange(0.0, 1.0 + 1e-6, 0.02)
    resultados = []
    
    print("[HAMMING] Iniciando experimento de distancia de Hamming vs 'a' de Rössler...")
    
    for a_val in valores_a:
        print(f"[HAMMING] Evaluando a = {a_val:.2f}...")
        params_test = dict(ROSSLER_PARAMS)
        params_test['a'] = float(a_val)
        # Se añade la siguiente instruccion para asegurar utilizar el valor real de rosslerparams
        # cuando valores_a contenga 0.2 (valor original)
        if abs(a_val - ROSSLER_PARAMS['a']) < 1e-8:
            params_test['a'] = ROSSLER_PARAMS['a']
            
        
        # Sincronizar con el nuevo a
        _, _, y_slave, t_slave, x_sinc = sincronizacion(
            y_sinc, times, params_test, time_sinc, keystream, nmax
        )
        # Descifrar
        difusion = revertir_confusion(vector_cifrado, vector_logistico, x_sinc)
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

def experimento_hamming_vs_b(
    ROSSLER_PARAMS,
    LOGISTIC_PARAMS,
    y_sinc,
    times,
    time_sinc,
    nmax,
    keystream,
    vector_cifrado,
    ancho,
    alto,
    vector_logistico
):
    img_original = Image.open(RUTA_IMAGEN_ORIGINAL)
    valores_b = np.arange(0.0, 1.0 + 1e-6, 0.02)
    resultados = []
    
    print("[HAMMING] Iniciando experimento de distancia de Hamming vs 'b' de Rössler...")
    
    for b_val in valores_b:
        print(f"[HAMMING] Evaluando b = {b_val:.2f}...")
        params_test = dict(ROSSLER_PARAMS)
        params_test['b'] = float(b_val)
        
        # Sincronizar con el nuevo b
        _, _, y_slave, t_slave, x_sinc = sincronizacion(
            y_sinc, times, params_test, time_sinc, keystream, nmax
        )
        # Descifrar
        difusion = revertir_confusion(vector_cifrado, vector_logistico, x_sinc, nmax)
        imagen_descifrada = revertir_difusion(difusion, vector_logistico, nmax, ancho, alto)

        # Calcular la distancia hamming
        hamming_abs, hamming_norm = distancia_hamming(img_original, imagen_descifrada)
        resultados.append({
            "b": b_val,
            "hamming_abs": hamming_abs,
            "hamming_norm": hamming_norm
        })
        
        df_resultados = pd.DataFrame(resultados)
        df_resultados.to_csv(RUTA_DISTANCIA_HAMMING_CSV_B, index=False)
        print(f"[HAMMING] Resultados guardados en {RUTA_DISTANCIA_HAMMING_CSV_B}")

        plt.figure(figsize=(10, 6))
        plt.plot(df_resultados['b'], df_resultados["hamming_abs"], marker = "o", linewidth = 0.8)
        plt.xlabel("b")
        plt.ylabel("Distancia de Hamming")
        plt.title("Distancia de hamming imagen original vs descifrada")
        plt.grid(True, alpha = 0.3)
        plt.tight_layout()
        plt.savefig(RUTA_DISTANCIA_HAMMING_B, dpi=300)
        plt.close()
        print(f"[HAMMING] Gráfico guardado en {RUTA_DISTANCIA_HAMMING_B}")

def experimento_hamming_vs_c(
    ROSSLER_PARAMS,
    LOGISTIC_PARAMS,
    y_sinc,
    times,
    time_sinc,
    nmax,
    keystream,
    vector_cifrado,
    ancho,
    alto,
    vector_logistico
):
    img_original = Image.open(RUTA_IMAGEN_ORIGINAL)
    valores_c = np.arange(5.0, 6.0 + 1e-6, 0.02)
    resultados = []
    
    print("[HAMMING] Iniciando experimento de distancia de Hamming vs 'c' de Rössler...")
    
    for c_val in valores_c:
        print(f"[HAMMING] Evaluando c = {c_val:.2f}...")
        params_test = dict(ROSSLER_PARAMS)
        params_test['c'] = float(c_val)
        
        # Sincronizar con el nuevo c
        _, _, y_slave, t_slave, x_sinc = sincronizacion(
            y_sinc, times, params_test, time_sinc, keystream, nmax
        )
        # Descifrar
        difusion = revertir_confusion(vector_cifrado, vector_logistico, x_sinc, nmax)
        imagen_descifrada = revertir_difusion(difusion, vector_logistico, nmax, ancho, alto)

        # Calcular la distancia hamming
        hamming_abs, hamming_norm = distancia_hamming(img_original, imagen_descifrada)
        resultados.append({
            "c": c_val,
            "hamming_abs": hamming_abs,
            "hamming_norm": hamming_norm
        })
        
        df_resultados = pd.DataFrame(resultados)
        df_resultados.to_csv(RUTA_DISTANCIA_HAMMING_CSV_C, index=False)
        print(f"[HAMMING] Resultados guardados en {RUTA_DISTANCIA_HAMMING_CSV_C}")

        plt.figure(figsize=(10, 6))
        plt.plot(df_resultados['c'], df_resultados["hamming_abs"], marker = "o", linewidth = 0.8)
        plt.xlabel("c")
        plt.ylabel("Distancia de Hamming")
        plt.title("Distancia de hamming imagen original vs descifrada")
        plt.grid(True, alpha = 0.3)
        plt.tight_layout()
        plt.savefig(RUTA_DISTANCIA_HAMMING_C, dpi=300)
        plt.close()
        print(f"[HAMMING] Gráfico guardado en {RUTA_DISTANCIA_HAMMING_C}")

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

        difusion = revertir_confusion(vector_cifrado, vector_logistico_test, x_sinc, nmax)
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
        difusion = revertir_confusion(vector_cifrado, vector_logistico_test, x_sinc, nmax)
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
    global RECEIVED_KEYS, RECEIVED_DATA
    t_inicio_programa = time.perf_counter()

    # ========== CONEXIÓN A MQTT ==========
    t_inicio_mqtt = time.perf_counter()

    client = mqtt.Client()
    client.username_pw_set(USERNAME, PASSWORD)
    client.tls_set(ca_certs=CA_CERT_PATH, tls_version=ssl.PROTOCOL_TLS_CLIENT)
    client.tls_insecure_set(False)
    client.on_connect = on_connect
    client.message_callback_add(TOPIC_DATA, on_message_data)
    client.message_callback_add(TOPIC_KEYS, on_message_keys)
    client.connect(BROKER, PORT, 60)
    client.loop_start()

    print("[MQTT-ESCLAVO] Esperando datos del maestro...")
    while RECEIVED_KEYS is None or RECEIVED_DATA is None:
        time.sleep(0.2)
    t_fin_mqtt = time.perf_counter()
    client.loop_stop()
    client.disconnect()

    print("[MQTT-ESCLAVO] Datos y keys recibidos correctamente")
    # ========== EXTRACCIÓN DE PARÁMETROS ==========
    (
        ROSSLER_PARAMS,
        LOGISTIC_PARAMS,
        vector_cifrado,
        x_maestro,
        y_maestro,
        z_maestro,
        t_maestro,
        ancho,
        alto,
        nmax,
        tiempo_sinc,
        keystream
    ) = extraer_parametros(RECEIVED_KEYS, RECEIVED_DATA)
    # Imprimir longitud de todos los datos recibidos
    print(f"[EXTRACCIÓN] Longitud vector_cifrado: {len(vector_cifrado)}, times: {len(t_maestro)}, nmax: {nmax}, ancho: {ancho}, alto: {alto}, tiempo_sinc: {tiempo_sinc}, keystream: {keystream}")

    # ========== PROCESO DE SINCRONIZACIÓN ==========
    print("[SINCRONIZACIÓN] Sincronizando sistema esclavo...")
    t_inicio_sincronizacion = time.perf_counter()
    x_esclavo, y_esclavo, z_esclavo, t_esclavo, x_sinc = sincronizacion(
        y_maestro, t_maestro, ROSSLER_PARAMS, tiempo_sinc, keystream, nmax
    )
    t_fin_sincronizacion = time.perf_counter()

    print("[SINCRONIZACIÓN] Proceso de sincronización completado.")

    # ========== PROCESO DE DESCIFRADO ==========
    print("[DESCIFRADO] Iniciando proceso de descifrado...")
    t_inicio_descifrado = time.perf_counter()

    vector_logistico = mapa_logistico(LOGISTIC_PARAMS, nmax)
    print("[DESCIFRADO] Vector logístico generado.")

    # Revertir la confusión
    difusion_x = revertir_confusion(vector_cifrado, vector_logistico, x_sinc)
    print("[DESCIFRADO] Confusión revertida.")
    difusion = difusion_x * IMG_SCALE

    # Revertir la difusión
    imagen_descifrada = revertir_difusion(
        difusion, vector_logistico, nmax, ancho, alto
    )
    print("[DESCIFRADO] Difusión revertida. Imagen descifrada generada.")
    t_fin_descifrado = time.perf_counter()
    t_fin_programa = time.perf_counter()

    # Guardar imagen descifrada
    imagen_descifrada.save(RUTA_IMAGEN_DESCIFRADA)
    print(f"[DESCIFRADO] Imagen descifrada guardada en {RUTA_IMAGEN_DESCIFRADA}")
    # ========== GUARDADO DE RESULTADOS ==========
    error_x = np.abs(x_maestro - x_esclavo)
    error_y = np.abs(y_maestro - y_esclavo)
    error_z = np.abs(z_maestro - z_esclavo)
    grafica_serie_temporal(t_maestro, x_maestro, y_maestro, z_maestro, t_esclavo, x_esclavo, y_esclavo, z_esclavo)
    graficar_errores_dispersion(t_maestro, x_maestro, y_maestro, z_maestro, t_esclavo, x_esclavo, y_esclavo, z_esclavo, error_x, error_y, error_z)
    graficar_histogramas()
    graficar_dispersion_pixeles()
    guardar_errores(t_esclavo, error_x, error_y, error_z)
    distancia_hamming(Image.open(RUTA_IMAGEN_ORIGINAL), imagen_descifrada)

    img_original_base = Image.open(RUTA_IMAGEN_ORIGINAL)
    coef_corr_base = coeficiente_correlacion_imagenes(img_original_base, imagen_descifrada)

    # Guardar correlación en CSV
    df_corr_base = pd.DataFrame([{
        "descripcion": "correlacion_original_vs_descifrada",
        "coef_corr": coef_corr_base
    }])
    df_corr_base.to_csv(
        RUTA_CORRELACION_BASE,
        index=False,
        mode="a",
        header=not RUTA_CORRELACION_BASE.exists()
    )
    print(f"[CORR] Correlación base guardada en {RUTA_CORRELACION_BASE}")
    # ========== REGISTRO DE TIEMPOS ==========
    tiempo_mqtt = t_fin_mqtt - t_inicio_mqtt
    tiempo_sincronizacion = t_fin_sincronizacion - t_inicio_sincronizacion
    tiempo_descifrado = t_fin_descifrado - t_inicio_descifrado
    tiempo_total = t_fin_programa - t_inicio_programa

    print(f"[TIEMPOS] Tiempo MQTT: {tiempo_mqtt:.4f} segundos")
    print(f"[TIEMPOS] Tiempo sincronización: {tiempo_sincronizacion:.4f} segundos")
    print(f"[TIEMPOS] Tiempo descifrado: {tiempo_descifrado:.4f} segundos")
    print(f"[TIEMPOS] Tiempo total: {tiempo_total:.4f} segundos")
    tiempos_procesos(tiempo_mqtt, tiempo_sincronizacion, tiempo_descifrado, tiempo_total)

    # experimento_hamming_vs_a(
    #     ROSSLER_PARAMS,
    #     LOGISTIC_PARAMS,
    #     y_maestro,
    #     t_maestro,
    #     tiempo_sinc,
    #     nmax,
    #     keystream,
    #     vector_cifrado,
    #     ancho,
    #     alto,
    #     vector_logistico
    # )
    # experimento_hamming_vs_b(
    #     ROSSLER_PARAMS,
    #     LOGISTIC_PARAMS,
    #     y_maestro,
    #     t_maestro,
    #     time_sinc,
    #     nmax,
    #     keystream,
    #     vector_cifrado,
    #     ancho,
    #     alto,
    #     vector_logistico
    # )
    # experimento_hamming_vs_c(
    #     ROSSLER_PARAMS,
    #     LOGISTIC_PARAMS,
    #     y_maestro,
    #     t_maestro,
    #     time_sinc,
    #     nmax,
    #     keystream,
    #     vector_cifrado,
    #     ancho,
    #     alto,
    #     vector_logistico
    # )
    # experimento_hamming_vs_logistico(
    #     LOGISTIC_PARAMS,
    #     vector_cifrado,
    #     ancho,
    #     alto,
    #     nmax,
    #     x_sinc,
    #     vector_logistico,
    #     img_original_base
    # )

if __name__ == "__main__":
    main()
    print("[ESCLAVO] Programa finalizado.")
