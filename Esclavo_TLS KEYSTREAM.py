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
BROKER = "192.168.0.55"
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
CARPETA_RESULTADOS = Path("Resultados_TLS_KEYSTREAM")
RUTA_SINCRONIZACION_Y1 = CARPETA_RESULTADOS / "sincronizacion_y_resultados.png"
RUTA_SINCRONIZACION_Y2 = CARPETA_RESULTADOS / "sincronizacion_y_resultados2.png"

RUTA_ERROR_Y_GRAFICA = CARPETA_RESULTADOS / "error_sincronizacion_y.png"
RUTA_DISPERSION_Y = CARPETA_RESULTADOS / "dispersion_error_y.png"
RUTA_DISPERSION_PIXELES = CARPETA_RESULTADOS / "dispersion_pixeles.png"

RUTA_ERROR_Y = CARPETA_RESULTADOS / "error_sincronizacion_Y.csv"
RUTA_SERIETEMPORAL_Y = CARPETA_RESULTADOS / "serie_temporal_y.csv"

RUTA_IMAGEN_ORIGINAL = Path("Prueba.jpg")
RUTA_HISTOGRAMA_IMAGENES = CARPETA_RESULTADOS / "histogramas.png"

RUTA_IMAGEN_DESCIFRADA = CARPETA_RESULTADOS / "imagen_descifrada.png"
RUTA_TIMINGS = CARPETA_RESULTADOS / "tiempo_procesos_esclavo.csv"




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
def sincronizacion(y_sinc, times, ROSSLER_PARAMS, time_sinc, keystream, nmax):

    iteraciones = time_sinc + keystream
    t_span = (0, iteraciones*H)
    t_eval = np.linspace(0, iteraciones*H, iteraciones)

    y_master_interp = interp1d(
        times,
        y_sinc,
        kind='cubic',
        fill_value="extrapolate"
    )    

    sol_slave = solve_ivp(
        fun = rossler_esclavo,
        t_span = t_span,
        y0 = X0,
        t_eval = t_eval,
        args = (y_master_interp, 
                ROSSLER_PARAMS['a'], 
                ROSSLER_PARAMS['b'], 
                ROSSLER_PARAMS['c'], 
                K),
        rtol = 1e-8,
        atol = 1e-8,
        method='RK45'
    )
    
    y_slave = sol_slave.y[1]
    x_sinc = sol_slave.y[0][time_sinc:time_sinc + keystream]
    x_sinc = np.resize(x_sinc, nmax)
    return y_master_interp, t_eval, y_slave, sol_slave.t, x_sinc

def revertir_confusion(vector_cifrado, vector_logistico, x_sinc, nmax):

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
    img_array = (vector_temp * 255).reshape(alto, ancho, 3).astype(np.uint8)
    return Image.fromarray(img_array)

def extraer_parametros(receivedKeys, receivedData):
    # Llaves del maestro
    ROSSLER_PARAMS = receivedKeys['ROSSLER_PARAMS']
    LOGISTIC_PARAMS = receivedKeys['LOGISTIC_PARAMS']

    y_sinc = np.array(receivedData['y_sinc'])
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
        y_sinc,
        times,
        time_sinc,
        nmax,
        keystream,
        vector_cifrado,
        ancho,
        alto
    )
    

def grafica_serie_temporal(times, y_maestro, t_esclavo, y_esclavo):
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

def graficar_error_dispersion(times, y_maestro, t_esclavo, y_esclavo, error_y):
    n_comun = min(len(times), len(t_esclavo))
    if n_comun <= 0:
        print("[GRAFICAS] No hay suficientes datos para graficar error y dispersión.")
        return

    t_comun = times[:n_comun]
    y_maestro_c = y_maestro[:n_comun]
    y_esclavo_c = y_esclavo[:n_comun]

    # ==================== GRÁFICA ERROR VS TIEMPO ====================
    plt.figure(figsize=(10, 6))
    plt.plot(t_comun, error_y, linewidth=1.4)
    plt.xlabel("Tiempo")
    plt.ylabel("|y_m(t) - y_s(t)|")
    plt.title("Error en la serie y(t) entre maestro y esclavo")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RUTA_ERROR_Y_GRAFICA, dpi=300)
    print(f"[GRAFICAS] Error |y_m - y_s| vs tiempo guardado en {RUTA_ERROR_Y_GRAFICA}")

    # ==================== DIAGRAMA DE DISPERSIÓN y_m vs y_s ====================
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

def guardar_error_y_csv(t_slave, error_y):
    df_error = pd.DataFrame({
        'Tiempo': t_slave,
        'Error_y': error_y
    })
    df_error.to_csv(RUTA_ERROR_Y, index=False)
    print(f"[SINCRONIZACIÓN] Error de sincronización guardado en {RUTA_ERROR_Y}")


def guardar_serietemporal_y(t_slave, y_slave):
    df_y_slave = pd.DataFrame({
        'Tiempo': t_slave,
        'Y_esclavo': y_slave
    })
    df_y_slave.to_csv(RUTA_SERIETEMPORAL_Y, index=False)
    print(f"[SINCRONIZACIÓN] Serie temporal Y esclavo guardada en {RUTA_SERIETEMPORAL_Y}")

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
        y_sinc,
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
    y_master_interp, t_eval, y_slave, t_slave, x_sinc = sincronizacion(
        y_sinc, times, ROSSLER_PARAMS, time_sinc, keystream, nmax
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
    error_y = np.abs(y_sinc - y_slave)
    grafica_serie_temporal(times, y_sinc, t_slave, y_slave)
    graficar_error_dispersion(times, y_sinc, t_slave, y_slave, error_y)
    graficar_histogramas()
    graficar_dispersion_pixeles()

    guardar_serietemporal_y(t_slave, y_slave)
    guardar_error_y_csv(t_slave, error_y)

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
