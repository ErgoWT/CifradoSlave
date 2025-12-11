"""
Docstring for Esclavo_TLS KEYSTREAM NoInterp
Experimento para implementar un sistema de sincronización sin utilizar interpolación
Se implementan todos los análisis y gráficas correspondientes
El canal de comunicación es seguro mediante TLS
Se utiliza hamming para el análisis de sensibilidad a parámetros
"""

import json 
import time 
import ssl
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import paho.mqtt.client as mqtt
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

# ========== PARAMETROS GLOBALES PARA ALMACENAMIENTO ==========
RECEIVED_KEYS = None
RECEIVED_DATA = None

# ========== PARAMETROS DE ROSSLER ==========
H = 0.01
K = 2.0
X0 = [1.0, 1.0, 1.0]

# ========== RUTAS Y ARCHIVOS ==========
CARPETA_RESULTADOS = Path("No_Interpolar")

# RUTAS SERIES TEMPORALES
RUTA_SINCRONIZACION_X = CARPETA_RESULTADOS / "sincronizacion_x_resultados.png"
RUTA_SINCRONIZACION_Y1 = CARPETA_RESULTADOS / "sincronizacion_y_resultados.png"
RUTA_SINCRONIZACION_Y2 = CARPETA_RESULTADOS / "sincronizacion_y_resultados2.png"
RUTA_SINCRONIZACION_Z = CARPETA_RESULTADOS / "sincronizacion_z_resultados.png"
# RUTAS ERRORES Y DISPERSIONES
RUTA_ERROR_X_GRAFICA = CARPETA_RESULTADOS / "error_sincronizacion_x.png"
RUTA_ERROR_Y_GRAFICA = CARPETA_RESULTADOS / "error_sincronizacion_y.png"
RUTA_ERROR_Z_GRAFICA = CARPETA_RESULTADOS / "error_sincronizacion_z.png"
RUTA_DISPERSION_X = CARPETA_RESULTADOS / "dispersion_error_x.png"
RUTA_DISPERSION_Y = CARPETA_RESULTADOS / "dispersion_error_y.png"
RUTA_DISPERSION_Z = CARPETA_RESULTADOS / "dispersion_error_z.png"
RUTA_DISPERSION_PIXELES = CARPETA_RESULTADOS / "dispersion_pixeles.png"
#RUTAS DE CSV
RUTA_ERRORES = CARPETA_RESULTADOS / "errores_sincronizacion.csv"
# RUTAS DE IMAGENES E HISTOGRAMAS
RUTA_IMAGEN_ORIGINAL = Path("Prueba2.jpg")
RUTA_HISTOGRAMA_IMAGENES = CARPETA_RESULTADOS / "histogramas.png"
RUTA_IMAGEN_DESCIFRADA = CARPETA_RESULTADOS / "imagen_descifrada.png"
RUTA_TIMINGS = CARPETA_RESULTADOS / "tiempo_procesos_esclavo.csv"

# RUTAS PARA DISTANCIA HAMMING
RUTA_DISTANCIA_HAMMING_A = CARPETA_RESULTADOS / "hamming_vs_a.png"
RUTA_DISTANCIA_HAMMING_CSV_A = CARPETA_RESULTADOS / "hamming_vs_a.csv"
RUTA_DISTANCIA_HAMMING_B = CARPETA_RESULTADOS / "hamming_vs_b.png"
RUTA_DISTANCIA_HAMMING_CSV_B = CARPETA_RESULTADOS / "hamming_vs_b.csv"
RUTA_DISTANCIA_HAMMING_C = CARPETA_RESULTADOS / "hamming_vs_c.png"
RUTA_DISTANCIA_HAMMING_CSV_C = CARPETA_RESULTADOS / "hamming_vs_c.csv"
    
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
def rossler_slave(state, y_m, a, b, c, k):
    x_s, y_s, z_s = state
    dxdt = -y_s - z_s
    dydt = x_s + a * y_s + k * (y_m - y_s)
    dzdt = b + z_s * (x_s - c)
    return np.array([dxdt, dydt, dzdt], dtype=float)

def integrar_slave(y_maestro, h, a, b, c, k, x0):
    n = len(y_maestro)
    t = np.linspace(0, (n - 1) * h, n, dtype=np.float64)

    x = np.empty(n, dtype=np.float64)
    y = np.empty(n, dtype=np.float64)
    z = np.empty(n, dtype=np.float64)
    x[0], y[0], z[0] = x0

    for i in range(n - 1):
        y_m = y_maestro[i]
        # k1
        dx1 = -y[i] - z[i]
        dy1 = x[i] + a * y[i] + k * (y_m - y[i])
        dz1 = b + z[i] * (x[i] - c)
        # k2
        x2 = x[i] + 0.5 * h * dx1
        y2 = y[i] + 0.5 * h * dy1
        z2 = z[i] + 0.5 * h * dz1
        dx2 = -y2 - z2
        dy2 = x2 + a * y2 + k * (y_m - y2)
        dz2 = b + z2 * (x2 - c)
        # k3
        x3 = x[i] + 0.5 * h * dx2
        y3 = y[i] + 0.5 * h * dy2
        z3 = z[i] + 0.5 * h * dz2
        dx3 = -y3 - z3
        dy3 = x3 + a * y3 + k * (y_m - y3)
        dz3 = b + z3 * (x3 - c)
        # k4 (mismo y_m para cero‑orden)
        x4 = x[i] + h * dx3
        y4 = y[i] + h * dy3
        z4 = z[i] + h * dz3
        dx4 = -y4 - z4
        dy4 = x4 + a * y4 + k * (y_m - y4)
        dz4 = b + z4 * (x4 - c)

        x[i+1] = x[i] + (h/6.0) * (dx1 + 2*dx2 + 2*dx3 + dx4)
        y[i+1] = y[i] + (h/6.0) * (dy1 + 2*dy2 + 2*dy3 + dy4)
        z[i+1] = z[i] + (h/6.0) * (dz1 + 2*dz2 + 2*dz3 + dz4)

    return t, x, y, z


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
def sincronizacion(y_sinc, times, ROSSLER_PARAMS, time_sinc, keystream, nmax):
    iteraciones = time_sinc + keystream

    if len(y_sinc) < iteraciones:
        raise ValueError("La señal y_sinc recibida es más corta que el número de iteraciones esperado.")

    y_maestro = y_sinc[:iteraciones]

    t_slave, x_slave, y_slave, z_slave = integrar_slave(
        y_maestro=y_maestro,
        h=H,
        a=ROSSLER_PARAMS['a'],
        b=ROSSLER_PARAMS['b'],
        c=ROSSLER_PARAMS['c'],
        k=K,
        x0=X0
    )

    if time_sinc + keystream > len(x_slave):
        raise ValueError("x_slave más corto que time_sinc+keystream")

    x_sinc = x_slave[time_sinc:time_sinc + keystream]
    x_sinc = np.resize(x_sinc, nmax)

    return x_slave, y_slave, z_slave, t_slave, x_sinc

def revertir_confusion(vector_cifrado, vector_logistico, x_sinc, nmax):

    difusion = vector_cifrado - vector_logistico - x_sinc

    return difusion

def revertir_difusion(difusion, vector_logistico, nmax, ancho, alto):
    # 1. Regenerar vector de mezcla (mismo que el maestro)
    vector_mezcla = np.floor(vector_logistico * nmax).astype(int)
    vector_mezcla = np.clip(vector_mezcla, 0, nmax - 1)

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
    img_array = (vector_temp * 255).reshape(alto, ancho, 3).astype(np.uint8)
    return Image.fromarray(img_array)

def extraer_parametros(receivedKeys, receivedData):
    # Llaves del maestro
    ROSSLER_PARAMS = receivedKeys['ROSSLER_PARAMS']
    LOGISTIC_PARAMS = receivedKeys['LOGISTIC_PARAMS']

    x_maestro = np.array(receivedData['x_maestro'])
    y_maestro = np.array(receivedData['y_maestro'])
    z_maestro = np.array(receivedData['z_maestro'])
    times = np.array(receivedData['times'])

    time_sinc = int(receivedData['time_sinc'])
    nmax = int(receivedData['nmax'])
    keystream = int(receivedData['KEYSTREAM'])

    vector_cifrado = np.array(receivedData['vector_cifrado'])
    ancho = int(receivedData['ancho'])
    alto = int(receivedData['alto'])

    return (
        ROSSLER_PARAMS,
        LOGISTIC_PARAMS,
        x_maestro,
        y_maestro,
        z_maestro,
        times,
        time_sinc,
        nmax,
        keystream,
        vector_cifrado,
        ancho,
        alto
    )
    

def grafica_serie_temporal(times, x_maestro, y_maestro, z_maestro, t_esclavo, x_esclavo, y_esclavo, z_esclavo):
    # Serie temporal en X
    plt.figure(figsize=(10, 6))
    plt.plot(times, x_maestro, label='Maestro x(t)', linewidth = 1.4, color='red') # Señal del maestro
    plt.plot(t_esclavo, x_esclavo, label='Esclavo x(t)', linewidth = 1.4, linestyle='--', alpha=0.9)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("x(t)")
    plt.title("Serie Temporal de x(t): Maestro vs Esclavo")
    plt.legend(loc="upper right", frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RUTA_SINCRONIZACION_X, dpi=300)

    # Serie temporal en Y
    plt.figure(figsize=(10, 6))
    plt.plot(times, y_maestro, label='Maestro y(t)', linewidth = 1.4, color='red') # Señal del maestro
    plt.plot(t_esclavo, y_esclavo, label='Esclavo y(t)', linewidth = 1.4, linestyle='--', alpha=0.9)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("y(t)")
    plt.title("Serie Temporal de y(t): Maestro vs Esclavo")
    plt.legend(loc="upper right", frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RUTA_SINCRONIZACION_Y1, dpi=300)

    # Serie temporal en Z
    plt.figure(figsize=(10, 6))
    plt.plot(times, z_maestro, label='Maestro z(t)', linewidth = 1.4, color='red')
    plt.plot(t_esclavo, z_esclavo, label='Esclavo z(t)', linewidth = 1.4, linestyle='--', alpha=0.9)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("z(t)")
    plt.title("Serie Temporal de z(t): Maestro vs Esclavo")
    plt.legend(loc="upper right", frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RUTA_SINCRONIZACION_Z, dpi=300)

    # Gráfica primeros datos
    n_maestro = max(1, int(0.15 * len(times)))
    n_esclavo = max(1, int(0.15 * len(t_esclavo)))

    times_25 = times[:n_maestro]
    y_maestro_25 = y_maestro[:n_maestro]
    t_esclavo_25 = t_esclavo[:n_esclavo]
    y_esclavo_25 = y_esclavo[:n_esclavo]

    plt.figure(figsize=(10, 6))
    plt.plot(times_25, y_maestro_25, label="Maestro  y(t)", linewidth=1.4)
    plt.plot(t_esclavo_25, y_esclavo_25, label="Esclavo  y(t)", linewidth=1.4, linestyle="--", alpha=0.9)
    plt.xlabel("Tiempo")
    plt.ylabel("y(t)")
    plt.title("Serie temporal de y(t) - Etapa de acoplamiento")
    plt.legend(loc="upper right", frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RUTA_SINCRONIZACION_Y2, dpi=300)

def graficar_error_dispersion(times, x_maestro, y_maestro, z_maestro, t_esclavo, x_esclavo, y_esclavo, z_esclavo, error_x, error_y, error_z):
    n_comun = min(len(times), len(t_esclavo))
    if n_comun <= 0:
        print("[GRAFICAS] No hay suficientes datos para graficar error y dispersión.")
        return

    t_comun = times[:n_comun]
    x_maestro_c = x_maestro[:n_comun]
    x_esclavo_c = x_esclavo[:n_comun]
    y_maestro_c = y_maestro[:n_comun]
    y_esclavo_c = y_esclavo[:n_comun]
    z_maestro_c = z_maestro[:n_comun]
    z_esclavo_c = z_esclavo[:n_comun]

    # ==================== GRÁFICA ERROR VS TIEMPO ====================
    # En x
    plt.figure(figsize=(10, 6))
    plt.plot(t_comun, error_x, linewidth=1.4)
    plt.xlabel("Tiempo")
    plt.ylabel("|x_m(t) - x_s(t)|")
    plt.title("Error en la serie x(t) entre maestro y esclavo")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RUTA_ERROR_X_GRAFICA, dpi=300)
    print(f"[GRAFICAS] Error |x_m - x_s| vs tiempo guardado en {RUTA_ERROR_X_GRAFICA}")

    # En y
    plt.figure(figsize=(10, 6))
    plt.plot(t_comun, error_y, linewidth=1.4)
    plt.xlabel("Tiempo")
    plt.ylabel("|y_m(t) - y_s(t)|")
    plt.title("Error en la serie y(t) entre maestro y esclavo")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RUTA_ERROR_Y_GRAFICA, dpi=300)
    print(f"[GRAFICAS] Error |y_m - y_s| vs tiempo guardado en {RUTA_ERROR_Y_GRAFICA}")

    # En z
    plt.figure(figsize=(10, 6))
    plt.plot(t_comun, error_z, linewidth=1.4)
    plt.xlabel("Tiempo")
    plt.ylabel("|z_m(t) - z_s(t)|")
    plt.title("Error en la serie z(t) entre maestro y esclavo")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RUTA_ERROR_Z_GRAFICA, dpi=300)
    print(f"[GRAFICAS] Error |z_m - z_s| vs tiempo guardado en {RUTA_ERROR_Z_GRAFICA}")

    # ==================== DIAGRAMA DE DISPERSIÓN ====================
    # En X
    plt.figure(figsize=(10, 6))
    plt.scatter(x_maestro_c, x_esclavo_c, s=8, alpha=0.6,)
    min_val = min(x_maestro_c.min(), x_esclavo_c.min())
    max_val = max(x_maestro_c.max(), x_esclavo_c.max())
    plt.plot([min_val, max_val], [min_val, max_val], linewidth=1.0, linestyle="--")
    plt.xlabel("x_m(t)  (maestro)")
    plt.ylabel("x_s(t)  (esclavo)")
    plt.title("Diagrama de dispersión x_maestro vs x_esclavo")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RUTA_DISPERSION_X, dpi=300)

    # En Y
    plt.figure(figsize=(10, 6))
    plt.scatter(y_maestro_c, y_esclavo_c, s=8, alpha=0.6,)
    min_val = min(y_maestro_c.min(), y_esclavo_c.min())
    max_val = max(y_maestro_c.max(), y_esclavo_c.max())
    plt.plot([min_val, max_val], [min_val, max_val], linewidth=1.0, linestyle="--")
    plt.xlabel("y_m(t)  (maestro)")
    plt.ylabel("y_s(t)  (esclavo)")
    plt.title("Diagrama de dispersión y_maestro vs y_esclavo")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RUTA_DISPERSION_Y, dpi=300)

    # En Z
    plt.figure(figsize=(10, 6))
    plt.scatter(z_maestro_c, z_esclavo_c, s=8, alpha=0.6,)
    min_val = min(z_maestro_c.min(), z_esclavo_c.min())
    max_val = max(z_maestro_c.max(), z_esclavo_c.max())
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

    plt.figure(figsize=(10, 6))
    bins = np.arange(257)
    plt.hist(pix_original, bins=bins, density=True, alpha=0.5, label="Imagen original")
    plt.hist(pix_descifrada, bins=bins, density=True, alpha=0.5, label="Imagen descifrada")
    plt.xlabel("Intensidad (0 - 255)")
    plt.ylabel("Densidad de probabilidad")
    plt.title("Histograma de intensidades: original vs descifrada (escala de grises)")
    plt.legend(loc="upper right", frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RUTA_HISTOGRAMA_IMAGENES, dpi=300)

def graficar_dispersion_pixeles(muestreo = 10000):
    img_original = Image.open(RUTA_IMAGEN_ORIGINAL).convert("L")
    img_descifrada = Image.open(RUTA_IMAGEN_DESCIFRADA).convert("L")

    pix_original = np.array(img_original).ravel()
    pix_descifrada = np.array(img_descifrada).ravel()

    n_comun = min(len(pix_original), len(pix_descifrada))

    pix_original = pix_original[:n_comun]
    pix_descifrada = pix_descifrada[:n_comun]

    # Muestreo opcional para no saturar
    if n_comun > muestreo:
        indices = np.random.choice(n_comun, size=muestreo, replace=False)
        pix_original = pix_original[indices]
        pix_descifrada = pix_descifrada[indices]

    plt.figure(figsize=(10, 6))
    plt.scatter(pix_original, pix_descifrada, s=5, alpha=0.4)
    min_val = 0
    max_val = 255
    plt.plot([min_val, max_val], [min_val, max_val], linewidth=1.0, linestyle="--")
    plt.xlabel("Valor de píxel imagen original")
    plt.ylabel("Valor de píxel imagen descifrada")
    plt.title("Dispersión de valores de píxel: original vs descifrada")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RUTA_DISPERSION_PIXELES, dpi=300)

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
        
        # Sincronizar con el nuevo a
        _, _, y_slave, t_slave, x_sinc = sincronizacion(
            y_sinc, times, params_test, time_sinc, keystream, nmax
        )
        # Descifrar
        difusion = revertir_confusion(vector_cifrado, vector_logistico, x_sinc, nmax)
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

def guardar_errores_csv(t_slave, error_x, error_y, error_z):
    df_error = pd.DataFrame({
        'Tiempo': t_slave,
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
        x_maestro,
        y_maestro,
        z_maestro,
        times,
        time_sinc,
        nmax,
        keystream,
        vector_cifrado,
        ancho,
        alto
    ) = extraer_parametros(RECEIVED_KEYS, RECEIVED_DATA)

    # ========== PROCESO DE SINCRONIZACIÓN ==========
    print("[SINCRONIZACIÓN] Sincronizando sistema esclavo...")
    t_inicio_sincronizacion = time.perf_counter()
    x_esclavo, y_esclavo, z_esclavo, t_slave, x_sinc = sincronizacion(
    y_maestro, times, ROSSLER_PARAMS, time_sinc, keystream, nmax
    )
    t_fin_sincronizacion = time.perf_counter()

    print("[SINCRONIZACIÓN] Proceso de sincronización completado.")

    # ========== PROCESO DE DESCIFRADO ==========
    print("[DESCIFRADO] Iniciando proceso de descifrado...")
    t_inicio_descifrado = time.perf_counter()

    vector_logistico = mapa_logistico(LOGISTIC_PARAMS, nmax)
    print("[DESCIFRADO] Vector logístico generado.")

    # Revertir la confusión
    difusion = revertir_confusion(vector_cifrado, vector_logistico, x_sinc, nmax)
    print("[DESCIFRADO] Confusión revertida.")

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
    n_sync = min(len(times), len(x_maestro), len(y_maestro), len(z_maestro),
                 len(t_slave), len(x_esclavo), len(y_esclavo), len(z_esclavo))
    error_x = np.abs(x_maestro[:n_sync] - x_esclavo[:n_sync])
    error_y = np.abs(y_maestro[:n_sync] - y_esclavo[:n_sync])
    error_z = np.abs(z_maestro[:n_sync] - z_esclavo[:n_sync])

    grafica_serie_temporal(
        times[:n_sync], x_maestro[:n_sync], y_maestro[:n_sync], z_maestro[:n_sync],
        t_slave[:n_sync], x_esclavo[:n_sync], y_esclavo[:n_sync], z_esclavo[:n_sync]
    )
    graficar_error_dispersion(
        times[:n_sync], x_maestro[:n_sync], y_maestro[:n_sync], z_maestro[:n_sync],
        t_slave[:n_sync], x_esclavo[:n_sync], y_esclavo[:n_sync], z_esclavo[:n_sync],
        error_x, error_y, error_z
    )
    graficar_histogramas()
    graficar_dispersion_pixeles()
    guardar_errores_csv(t_slave, error_x, error_y, error_z)
    experimento_hamming_vs_a(
        ROSSLER_PARAMS,
        LOGISTIC_PARAMS,
        y_maestro,
        times,
        time_sinc,
        nmax,
        keystream,
        vector_cifrado,
        ancho,
        alto,
        vector_logistico
    )
    experimento_hamming_vs_b(
        ROSSLER_PARAMS,
        LOGISTIC_PARAMS,
        y_maestro,
        times,
        time_sinc,
        nmax,
        keystream,
        vector_cifrado,
        ancho,
        alto,
        vector_logistico
    )
    experimento_hamming_vs_c(
        ROSSLER_PARAMS,
        LOGISTIC_PARAMS,
        y_maestro,
        times,
        time_sinc,
        nmax,
        keystream,
        vector_cifrado,
        ancho,
        alto,
        vector_logistico
    )

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

if __name__ == "__main__":
    main()
    print("[ESCLAVO] Programa finalizado.")
