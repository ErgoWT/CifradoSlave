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
TOPIC_KEYS = "chaos/keys"
TOPIC_DATA = "chaos/data"
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
CARPETA_RESULTADOS = Path("Resultados")
RUTA_SINCRONIZACION = CARPETA_RESULTADOS / "sincronizacion_resultados.png"
RUTA_ERROR_Y = CARPETA_RESULTADOS / "error_sincronizacion_Y.csv"
RUTA_SERIETEMPORAL_Y = CARPETA_RESULTADOS / "serie_temporal_y.csv"
RUTA_IMAGEN_DESCIFRADA = CARPETA_RESULTADOS / "imagen_descifrada.png"


def on_connect(client, userdata, flags, rc):
    print(f"Conectado al broker MQTT con código de resultado: {rc}")
    if rc == 0:
        client.subscribe(TOPIC_KEYS, qos=QOS)
        client.subscribe(TOPIC_DATA, qos=QOS)
        print("[MQTT] Suscrito a los topics para recibir datos y keys")
    else:
        print("[MQTT] Error al conectar al broker MQTT")

def on_message_data(client, userdata, msg):
    global RECEIVED_DATA
    RECEIVED_DATA = json.loads(msg.payload.decode())
    print("Datos recibidos")

def on_message_keys(client, userdata, msg):
    global RECEIVED_KEYS
    RECEIVED_KEYS = json.loads(msg.payload.decode())
    print("Keys recibidas")

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
def sincronizacion(y_sinc, times, ROSSLER_PARAMS, time_sinc, nmax):

    iteraciones = time_sinc + nmax
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
    x_sinc = sol_slave.y[0][time_sinc:time_sinc + nmax]

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
        vector_cifrado,
        ancho,
        alto
    )
    

def graficas():
    # # Graficar resultados
    # plt.figure(figsize=(15, 10))
    
    # # 1. Trayectorias de y
    # plt.subplot(2, 1, 1)
    # plt.plot(t_slave, y_sinc, 'b-', label='Maestro (y)')
    # plt.plot(t_slave, y_master_interp(t_slave), 'r--', label='Esclavo (y)')
    # plt.xlabel('Tiempo')
    # plt.ylabel('y(t)')
    # plt.title('Sincronización de Sistemas de Rössler')
    # plt.legend()
    # plt.grid(True)
    
    # # 2. Error de sincronización
    # plt.subplot(2, 1, 2)
    # plt.plot(t_slave, error_y, 'g-')
    # plt.xlabel('Tiempo')
    # plt.ylabel('Error |y_maestro - y_esclavo|')
    # plt.title('Error de Sincronización')
    # plt.grid(True)
    
    # plt.tight_layout()
    # plt.savefig('Resultados/sincronizacion_resultados.png')
    # print("Resultados de sincronización guardados en sincronizacion_resultados.png")
    return 0

def guardar_error_y_csv(t_slave, error_y):
    df_error = pd.DataFrame({
        'Tiempo': t_slave,
        'Error_y': error_y
    })
    df_error.to_csv(RUTA_ERROR_Y, index=False)
    print(f"Error de sincronización guardado en {RUTA_ERROR_Y}")


def guardar_serietemporal_y(t_slave, y_slave):
    df_y_slave = pd.DataFrame({
        'Tiempo': t_slave,
        'Y_esclavo': y_slave
    })
    df_y_slave.to_csv(RUTA_SERIETEMPORAL_Y, index=False)
    print(f"Serie temporal Y esclavo guardada en {RUTA_SERIETEMPORAL_Y}")

def main():
    global RECEIVED_KEYS, RECEIVED_DATA

    # ========== CONEXIÓN A MQTT ==========
    client = mqtt.Client()
    client.username_pw_set(USERNAME, PASSWORD)
    client.tls_set(ca_certs=CA_CERT_PATH, tls_version=ssl.PROTOCOL_TLS_CLIENT)
    client.tls_insecure_set(False)
    client.on_connect = on_connect
    client.message_callback_add(TOPIC_DATA, on_message_data)
    client.message_callback_add(TOPIC_KEYS, on_message_keys)
    client.connect(BROKER, PORT, 60)
    client.loop_start()

    print("[MQTT] Esperando datos del maestro...")
    while RECEIVED_KEYS is None or RECEIVED_DATA is None:
        time.sleep(0.2)
    client.loop_stop()
    client.disconnect()
    
    # ========== EXTRACCIÓN DE PARÁMETROS ==========
    (
        ROSSLER_PARAMS,
        LOGISTIC_PARAMS,
        y_sinc,
        times,
        time_sinc,
        nmax,
        vector_cifrado,
        ancho,
        alto
    ) = extraer_parametros(RECEIVED_KEYS, RECEIVED_DATA)

    # ========== PROCESO DE SINCRONIZACIÓN ==========
    print("Sincronizando sistema esclavo...")
    y_master_interp, t_eval, y_slave, t_slave, x_sinc = sincronizacion(
        y_sinc, times, ROSSLER_PARAMS, time_sinc, nmax
    )
    error_y = np.abs(y_master_interp(t_eval) - y_slave)
    
    graficas()
    guardar_serietemporal_y(t_slave, y_slave)
    guardar_error_y_csv(t_slave, error_y)

    print(f"Datos de error guardados en {RUTA_ERROR_Y}")
    print("[ROSSLER] Proceso de sincronización completado.")

    # ========== PROCESO DE DESCIFRADO ==========
    print("Iniciando proceso de descifrado...")

    vector_logistico = mapa_logistico(LOGISTIC_PARAMS, nmax)
    print("Vector logístico generado.")

    # Revertir la confusión
    difusion = revertir_confusion(vector_cifrado, vector_logistico, x_sinc, nmax)

    print("Confusión revertida.")

    # Revertir la difusión
    imagen_descifrada = revertir_difusion(
        difusion, vector_logistico, nmax, ancho, alto
    )
    print("Difusión revertida. Imagen descifrada generada.")
    # Guardar imagen descifrada
    imagen_descifrada.save(RUTA_IMAGEN_DESCIFRADA)
    print(f"Imagen descifrada guardada en {RUTA_IMAGEN_DESCIFRADA}")

if __name__ == "__main__":
    main()
    print("Programa finalizado.")
