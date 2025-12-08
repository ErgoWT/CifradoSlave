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

# ========== PARAMETROS GLOBALES PARA ALMACENAMIENTO ==========
RECEIVED_KEYS = None
RECEIVED_DATA = None

# ========== PARAMETROS DE ROSSLER ==========
H = 0.01
K = 2.0
X0 = [1.0, 1.0, 1.0]

# ========== RUTAS Y ARCHIVOS ==========
CARPETA_RESULTADOS = Path("Resultados_TLS_KEYSTREAM")
CARPETA_RESULTADOS_HAMMING = Path("Hamming_Resultados")

RUTA_IMAGEN_ORIGINAL = Path("Prueba2.jpg")
RUTA_TIMINGS = CARPETA_RESULTADOS / "tiempo_procesos_esclavo.csv"
RUTA_HISTOGRAMA_IMAGENES = CARPETA_RESULTADOS / "histogramas.png"
RUTA_IMAGEN_DESCIFRADA = CARPETA_RESULTADOS / "imagen_descifrada.png"
RUTA_DISTANCIA_HAMMING_A = CARPETA_RESULTADOS_HAMMING / "hamming_vs_a.png"
RUTA_DISTANCIA_HAMMING_CSV_A = CARPETA_RESULTADOS_HAMMING / "hamming_vs_a.csv"
RUTA_DISTANCIA_HAMMING_B = CARPETA_RESULTADOS_HAMMING / "hamming_vs_b.png"
RUTA_DISTANCIA_HAMMING_CSV_B = CARPETA_RESULTADOS_HAMMING / "hamming_vs_b.csv"
RUTA_DISTANCIA_HAMMING_C = CARPETA_RESULTADOS_HAMMING / "hamming_vs_c.png"
RUTA_DISTANCIA_HAMMING_CSV_C = CARPETA_RESULTADOS_HAMMING / "hamming_vs_c.csv"

RUTA_CORRELACION_BASE = CARPETA_RESULTADOS_HAMMING / "correlacion_original_vs_descifrada.csv"


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
        kind='nearest',
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
        rtol = 1e-5,
        atol = 1e-5,
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
    vector_temp = np.clip(vector_temp, 0.0, 1.0)
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
    graficar_histogramas()
    experimento_hamming_vs_a(
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
    )
    # experimento_hamming_vs_b(
    #     ROSSLER_PARAMS,
    #     LOGISTIC_PARAMS,
    #     y_sinc,
    #     times,
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
    #     y_sinc,
    #     times,
    #     time_sinc,
    #     nmax,
    #     keystream,
    #     vector_cifrado,
    #     ancho,
    #     alto,
    #     vector_logistico
    # )

        # ========== CORRELACIÓN IMAGEN ORIGINAL VS DESCIFRADA (CASO BASE) ==========
    img_original_base = Image.open(RUTA_IMAGEN_ORIGINAL)
    coef_corr_base = coeficiente_correlacion_imagenes(img_original_base, imagen_descifrada)

    # Guardar correlación en CSV (modo append)
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

if __name__ == "__main__":
    main()
    print("[ESCLAVO] Programa finalizado.")
