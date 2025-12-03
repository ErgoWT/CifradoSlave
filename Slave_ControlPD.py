import json
import time
import ssl
from pathlib import Path

import numpy as np
import paho.mqtt.client as mqtt
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Configuracion MQTT
BROKER = "raspberrypiJED.local"
PORT = 8883
USERNAME = "usuario1"
PASSWORD = "qwerty123"
TOPIC_KEYS = "chaosPD/keys"
TOPIC_DATA = "chaosPD/data"
CA_CERT_PATH = "/home/tunchi/CifradoSlave/certs/ca.crt"
QOS = 1

RECEIVED_KEYS = None
RECEIVED_DATA = None

# Parametros del sistema
H = 0.01
KEYSTREAM = 10000
KP = 2.0
KD = 0.0
X0 = [1.0, 1.0, 1.0]

# Rutas de archivos
CARPETA_RESULTADOS = Path("Resultados_SlavePD")
RUTA_SINCRONIZACION_X = CARPETA_RESULTADOS / "sincronizacion_X_SlavePD.png"
RUTA_SINCRONIZACION_Y = CARPETA_RESULTADOS / "sincronizacion_Y_SlavePD.png"
RUTA_SINCRONIZACION_Z = CARPETA_RESULTADOS / "sincronizacion_Z_SlavePD.png"
RUTA_DATOS = CARPETA_RESULTADOS / "datos_SlavePD.csv"
RUTA_ERROR_X = CARPETA_RESULTADOS / "error_X_SlavePD.png"

def on_connect(client, userdata, flags, rc):
    print("Conectado: ", rc)
    if rc == 0:
        client.subscribe(TOPIC_KEYS, qos=QOS)
        client.subscribe(TOPIC_DATA, qos=QOS)
    else:
        print("Error de conexion")

def on_message_keys(client, userdata, msg):
    global RECEIVED_KEYS
    if msg.topic == TOPIC_KEYS:
        RECEIVED_KEYS = json.loads(msg.payload.decode())
        print("Claves recibidas")

def on_message_data(client, userdata, msg):
    global RECEIVED_DATA
    if msg.topic == TOPIC_DATA:
        RECEIVED_DATA = json.loads(msg.payload.decode())
        print("Datos recibidos")

def rossler_slave(t, state, a, b, c, y_interp, dy_interp):
    x, y, z = state

    y_master = y_interp(t)
    dy_master = dy_interp(t)

    # Rossler sin control
    dxdt = -y - z
    dy_base = x + a * y
    dzdt = b + z * (x - c)

    # Derivada de y sin control
    dy_s = dy_base
    # Errores
    e = y_master - y # Error de salida
    de = dy_master - dy_s # Error de la derivada

    # Control PD
    u_p = KP * e
    u_d = KD *de
    u_pd = u_p + u_d

    # Se aplica el control en la ecuación de y
    dydt = dy_base + u_pd
    return [dxdt, dydt, dzdt]

def interpolar_datos(times, y_master):
    y_interp = interp1d(
        times, y_master, kind = "cubic", fill_value = "extrapolate"
    )

    # Derivada
    dy_master = np.gradient(y_master, times)
    # la funcion gradient devuelve un array con la derivada en cada punto
    dy_interp = interp1d(times, dy_master, kind = "cubic", fill_value = "extrapolate")
    return y_interp, dy_interp

def sincronizar_rossler(a, b, c, times, y_master):
    print("Iniciando sincronizacion...")
    y_interp, dy_interp = interpolar_datos(times, y_master)
    print("Interpolacion completada.")
    t_span = (0, 75.0)
    t_eval = np.arange(0, 75.0, H)

    sol = solve_ivp(
        lambda t, y: rossler_slave(t, y, a, b, c, y_interp, dy_interp),
        t_span = t_span,
        t_eval = t_eval,
        y0 = X0,
        method = "RK45",
        rtol = 1e-9,
        atol = 1e-9
    )

    x_slave = sol.y[0]
    y_slave = sol.y[1]
    z_slave = sol.y[2]
    t_slave = sol.t

    print("Sincronizacion completada.")
    return t_slave, x_slave, y_slave, z_slave

def recibir_datos(receivedKeys, receivedData):
    a = receivedKeys["a"]
    b = receivedKeys["b"]
    c = receivedKeys["c"]

    times = np.array(receivedData["t"])
    x_master = np.array(receivedData["x"])
    y_master = np.array(receivedData["y"])
    z_master = np.array(receivedData["z"])

    print(f"Parametros recibidos: a={a}, b={b}, c={c}")
    print(f"Datos recibidos: {len(times)} puntos de tiempo.")
    print(f"Tamaño de x_master: {len(x_master)}")
    print(f"Tamaño de y_master: {len(y_master)}")
    print(f"Tamaño de z_master: {len(z_master)}")

    return a, b, c, times, x_master, y_master, z_master

def graficar_x(t_slave, x_slave, x_master):
    plt.figure(figsize=(10, 6))
    plt.plot(t_slave, x_slave, label="x_slave (Slave PD)", color="blue", linewidth=1)
    plt.plot(t_slave, x_master, label="x_master (Master)", color="red", linewidth=1, alpha=0.7)
    plt.title("Sincronización del Sistema de Rössler con Control PD")
    plt.xlabel("Tiempo")
    plt.ylabel("Serie temporal x")
    plt.legend()
    plt.grid()
    plt.savefig(RUTA_SINCRONIZACION_X)
    plt.close()
    print(f"Gráfico guardado en {RUTA_SINCRONIZACION_X}")

def graficar_y(t_slave, y_slave, y_master):
    plt.figure(figsize=(10, 6))
    plt.plot(t_slave, y_slave, label="y_slave (Slave PD)", color="blue", linewidth=1)
    plt.plot(t_slave, y_master, label="y_master (Master)", color="red", linewidth=1, alpha=0.7)
    plt.title("Sincronización del Sistema de Rössler con Control PD")
    plt.xlabel("Tiempo")
    plt.ylabel("Serie temporal y")
    plt.legend()
    plt.grid()
    plt.savefig(RUTA_SINCRONIZACION_Y)
    plt.close()
    print(f"Gráfico guardado en {RUTA_SINCRONIZACION_Y}")

def graficar_z(t_slave, z_slave, z_master):
    plt.figure(figsize=(10, 6))
    plt.plot(t_slave, z_slave, label="z_slave (Slave PD)", color="blue", linewidth=1)
    plt.plot(t_slave, z_master, label="z_master (Master)", color="red", linewidth=1, alpha=0.7)
    plt.title("Sincronización del Sistema de Rössler con Control PD")
    plt.xlabel("Tiempo")
    plt.ylabel("Serie temporal z")
    plt.legend()
    plt.grid()
    plt.savefig(RUTA_SINCRONIZACION_Z)
    plt.close()
    print(f"Gráfico guardado en {RUTA_SINCRONIZACION_Z}")

def graficar_error_x(t_slave, error_x):
    plt.figure(figsize=(10, 6))
    plt.plot(t_slave, error_x, label="Error en x", color="green", linewidth=1)
    plt.title("Error de Sincronización en x")
    plt.xlabel("Tiempo")
    plt.ylabel("Error")
    plt.legend()
    plt.grid()
    plt.savefig(RUTA_ERROR_X)
    plt.close()
    print(f"Gráfico guardado en {RUTA_ERROR_X}")

# Función experimento_KD se encarga de hacer una gráfica del error
# conforme se varía el valor de KD manteniendo KP constante y guarda la grafica.
def experimento_KD(a, b, c, times, y_master):
    KD_values = np.arange(0.0, 2.0, 0.02)
    mae_values = []
    rmse_values = []

    print("Iniciando experimento de variacion de KD...")
    plt.figure(figsize=(10, 6))
    y_master = np.array(y_master)

    t_transitorio = 20.0

    for kd in KD_values:
        global KD
        KD = kd
        print(f"Probando KD = {KD:.2f}")
        t_slave, x_slave, y_slave, z_slave = sincronizar_rossler(a, b, c, times, y_master)
        # Aseguramos que trabajamos con el mismo largo
        n = min(len(y_slave), len(y_master))
        t_err = t_slave[:n]
        e_y = y_master[:n] - y_slave[:n]   # error en y

        mask = t_err >= t_transitorio
        t_ss = t_err[mask]
        e_y_ss = e_y[mask]
        mae = np.mean(np.abs(e_y_ss))
        rmse = np.sqrt(np.mean(e_y_ss**2))

        mae_values.append(mae)
        rmse_values.append(rmse)

        plt.plot(t_ss, e_y_ss, linewidth=0.8, alpha=0.7)

    plt.axhline(0, color='black', linewidth=0.5)
    plt.xlabel("Tiempo")
    plt.ylabel("Error en y")
    plt.title("Error de Sincronización en y para diferentes KD")
    plt.legend()
    plt.grid()
    ruta_experimento = CARPETA_RESULTADOS / "experimento_KD_SlavePD.png"
    plt.savefig(ruta_experimento)
    print(f"Gráfico del experimento guardado en {ruta_experimento}")

    # ---- Análisis numérico para encontrar el mejor KD ----
    mae_values = np.array(mae_values)
    rmse_values = np.array(rmse_values)
    KD_values_np = np.array(KD_values)

    # Elegimos el KD con menor MAE como "óptimo"
    idx_best = int(np.argmin(mae_values))
    KD_best = KD_values_np[idx_best]

    print("\nResultados del barrido de KD:")
    for kd, mae, rmse in zip(KD_values_np, mae_values, rmse_values):
        print(f"  KD = {kd:5.2f}  ->  MAE = {mae:.3e},  RMSE = {rmse:.3e}")

    print(f"\nKD óptimo (por MAE mínimo) = {KD_best:.2f}")
    print(f"MAE(KD_best)  = {mae_values[idx_best]:.3e}")
    print(f"RMSE(KD_best) = {rmse_values[idx_best]:.3e}")
    # (Opcional) Gráfica de barras con MAE por KD
    plt.figure(figsize=(8, 5))
    plt.bar([str(kd) for kd in KD_values_np], mae_values)
    plt.xlabel("KD")
    plt.ylabel("MAE del error en y")
    plt.title("Comparación de MAE para distintos KD")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    ruta_mae = CARPETA_RESULTADOS / "MAE_vs_KD_SlavePD.png"
    plt.savefig(ruta_mae)
    print(f"Gráfico MAE vs KD guardado en {ruta_mae}")

    print("Barrido de KD completado.")

def main():
    global RECEIVED_KEYS, RECEIVED_DATA

    # Conectar a MQTT
    client = mqtt.Client()
    client.username_pw_set(USERNAME, PASSWORD)
    client.tls_set(ca_certs=CA_CERT_PATH, tls_version=ssl.PROTOCOL_TLS_CLIENT)
    client.tls_insecure_set(False)
    client.on_connect = on_connect
    client.message_callback_add(TOPIC_KEYS, on_message_keys)
    client.message_callback_add(TOPIC_DATA, on_message_data)
    client.connect(BROKER, PORT, 60)
    client.loop_start()
    print("Esperando datos...")
    while RECEIVED_KEYS is None or RECEIVED_DATA is None:
        time.sleep(0.4)
    client.loop_stop()
    client.disconnect()
    print("Datos recibidos...")

    a, b, c, times, x_master, y_master, z_master = recibir_datos(RECEIVED_KEYS, RECEIVED_DATA)

    t_slave, x_slave, y_slave, z_slave = sincronizar_rossler(a, b, c, times, y_master)
    
    error_x = np.abs(x_master - x_slave)
    error_y = np.abs(y_master - y_slave)
    error_z = np.abs(z_master - z_slave)

    print(f"Minimo error en x: {np.min(error_x)} en tiempo {t_slave[np.argmin(error_x)]}")
    print(f"Minimo error en y: {np.min(error_y)} en tiempo {t_slave[np.argmin(error_y)]}")
    print(f"Minimo error en z: {np.min(error_z)} en tiempo {t_slave[np.argmin(error_z)]}")

    pd.DataFrame({
        "Tiempo": t_slave,
        "Error_X": error_x,
        "Error_Y": error_y,
        "Error_Z": error_z
    }).to_csv(RUTA_DATOS, index=False)
    print(f"Datos guardados en {RUTA_DATOS}")

    experimento_KD(a, b, c, times, y_master)
    graficar_x(t_slave, x_slave, x_master)
    graficar_y(t_slave, y_slave, y_master)
    graficar_z(t_slave, z_slave, z_master)
    graficar_error_x(t_slave, error_x)

if __name__ == "__main__":
    main()
