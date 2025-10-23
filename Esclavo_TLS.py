import numpy as np
import json 
import time 
import paho.mqtt.client as mqtt
import ssl
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

# ========== CONFIGURACION MQTT ==========
broker = "192.168.0.55"
port = 1883
port_tls = 8883
username = "usuario1"
password = "qwerty123"
topicKeys = "chaos/keys"
topicData = "chaos/data"
ca_cert_path = "\etc\mosquitto\ca_certificates\ca.crt"


# ========== PARAMETROS GLOBALES PARA ALMACENAMIENTO ==========
receivedKeys = None
receivedData = None

def on_connect(client, userdata, flags, rc):
    print(f"Conectado al broker MQTT con código de resultado: {rc}")
    client.subscribe(topicData)  # <— SOLO datos aquí


def on_message_keys(client, userdata, msg):
    global receivedKeys
    receivedKeys = json.loads(msg.payload.decode())
    print("Llaves de cifrado recibidos")

def on_message_data(client, userdata, msg):
    global receivedData
    receivedData = json.loads(msg.payload.decode())
    print("Datos recibidos")


# ========== FUNCION DE ROSSLER ESCLAVO ==========
def rossler_esclavo(t, state, y_master_interp, a, b, c, k):
    x_s, y_s, z_s = state
    y_m = y_master_interp(t)
    dxdt = -y_s - z_s
    dydt = x_s + a * y_s + k * (y_m - y_s)
    dzdt = b + z_s * (x_s - c)
    return [dxdt, dydt, dzdt]

# vector_cifrado nmax logisticParams
# ========== FUNCION PARA REVERTIR LA CONFUSIÓN ==========
def sincronizacion(y_sinc, times, rosslerParams, time_sinc, keystream):
    h = 0.01
    k = 2.0
    x0_slave = [1.0, 1.0, 1.0]

    y_master_interp = interp1d(
        times,
        y_sinc,
        kind='cubic',
        fill_value="extrapolate"
    )

    iteraciones = time_sinc + keystream
    t_span = (0, iteraciones*h)
    t_eval = np.linspace(0, iteraciones*h, iteraciones)

    sol_slave = solve_ivp(
        fun = rossler_esclavo,
        t_span = t_span,
        y0 = x0_slave,
        t_eval = t_eval,
        args = (y_master_interp, rosslerParams['a'], rosslerParams['b'], rosslerParams['c'], k),
        rtol = 1e-8,
        atol = 1e-8,
        method='RK45'
    )
    
    return y_master_interp, t_eval, sol_slave.y[1], sol_slave.t, sol_slave.y[0][time_sinc:]

def revertir_confusion(vector_cifrado, vector_logistico, x_sinc, nmax):
    x_cif_esclavo = np.resize(x_sinc, nmax)

    difusion = vector_cifrado - vector_logistico - x_cif_esclavo

    return difusion

def revertir_difusion(difusion, vector_logistico, nmax, ancho, alto):
    # 1. Regenerar vector de mezcla (mismo que el maestro)
    vector_mezcla = np.floor(vector_logistico * nmax).astype(int)

    # 2. Inicializar estructuras para revertir
    vector_temp = np.full(nmax, 260.0)
    vector_original = np.zeros(nmax)
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

def main():
    global receivedKeys, receivedData

    # Cliente SIN TLS para topicData (1883) — se mantiene igual
    client = mqtt.Client()
    client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.message_callback_add(topicData, on_message_data)
    client.connect(broker, port, 60)
    client.loop_start()

    # NUEVO: Cliente TLS SOLO para topicKeys (8883)
    client_keys = mqtt.Client()
    client_keys.username_pw_set(username, password)
    client_keys.tls_set(ca_certs=ca_cert_path, tls_version=ssl.PROTOCOL_TLS_CLIENT)
    client_keys.tls_insecure_set(False)
    client_keys.connect(broker, port_tls, 60)
    client_keys.message_callback_add(topicKeys, on_message_keys)
    client_keys.subscribe(topicKeys)
    client_keys.loop_start()

    print("Esperando datos del maestro...")
    while receivedKeys is None or receivedData is None:
        time.sleep(0.2)

    # Parar loops y desconectar
    client.loop_stop()
    client.disconnect()
    client_keys.loop_stop()
    client_keys.disconnect()

    # Verificar existencia de 'times'
    if 'times' not in receivedData:
        print("\nERROR: 'times' no encontrado en los datos recibidos")
        print("Claves disponibles en receivedData:", receivedData.keys())
        return

    # Extraer los parámetros
    rosslerParams = receivedKeys['rosslerParams']
    y_sinc = np.array(receivedData['y_sinc'])
    times = np.array(receivedData['times'])
    time_sinc = receivedData['time_sinc']
    keystream = receivedData['keystream']

    print("Sincronizando sistema esclavo...")
    y_master_interp, t_eval, y_slave, t_slave, x_sinc = sincronizacion(
        y_sinc, times, rosslerParams, time_sinc, keystream
    )

    error_y = np.abs(y_master_interp(t_eval) - y_slave)
    # Graficar resultados
    plt.figure(figsize=(15, 10))
    
    # 1. Trayectorias de y
    plt.subplot(2, 1, 1)
    plt.plot(t_slave, y_sinc, 'b-', label='Maestro (y)')
    plt.plot(t_slave, y_master_interp(t_slave), 'r--', label='Esclavo (y)')
    plt.xlabel('Tiempo')
    plt.ylabel('y(t)')
    plt.title('Sincronización de Sistemas de Rössler')
    plt.legend()
    plt.grid(True)
    
    # 2. Error de sincronización
    plt.subplot(2, 1, 2)
    plt.plot(t_slave, error_y, 'g-')
    plt.xlabel('Tiempo')
    plt.ylabel('Error |y_maestro - y_esclavo|')
    plt.title('Error de Sincronización')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('sincronizacion_resultados.png')
    print("Resultados de sincronización guardados en sincronizacion_resultados.png")
    data_y = {
    'Tiempo': t_eval,
    'Error_y': error_y
    }
    trayectoria_y = {
    'Tiempo': t_slave,
    'Trayectoria_y': y_slave
    }
    df_trayectoria = pd.DataFrame(trayectoria_y)
    df_trayectoria.to_csv('trayectoria_y_esclavo.csv', index=False)
    df = pd.DataFrame(data_y)
    df.to_csv('error_sincronizacion_Y.csv', index=False)
    print("Datos de error guardados en error_sincronizacion_Y.csv")
    print("Proceso de sincronización completado.")

    # ========== PROCESO DE DESCIFRADO ==========
    print("Iniciando proceso de descifrado...")
    # Extraer datos de cifrado
    vector_cifrado = np.array(receivedData['vector_cifrado'])
    nmax = receivedData['nmax']
    ancho = receivedData['ancho']
    alto = receivedData['alto']

    # Regenerar vector logístico (mismo que el maestro)
    logisticParams = receivedKeys['logisticParams']
    a_log = logisticParams['aLog']
    x0_log = logisticParams['x0_log']

    vector_logistico = np.zeros(nmax)
    x = x0_log
    for i in range(nmax):
        x = a_log * x * (1 - x)
        vector_logistico[i] = x
    print("Vector logístico generado.")

    # Revertir la confusión
    difusion = revertir_confusion(
        vector_cifrado, vector_logistico, x_sinc, nmax
    )

    print("Confusión revertida.")

    # Revertir la difusión
    imagen_descifrada = revertir_difusion(
        difusion, vector_logistico, nmax, ancho, alto
    )
    print("Difusión revertida. Imagen descifrada generada.")
    # Guardar imagen descifrada
    imagen_descifrada.save('imagen_descifrada.png')
    print("Imagen descifrada guardada como imagen_descifrada.png")

if __name__ == "__main__":
    main()
    print("Programa finalizado.")
