"""
================================================================================
ANALISIS GRAFICO DE LA INTERPOLACION CUBICA
Proyecto: Encriptacion con sistema dinamico de Rossler
================================================================================

Este script genera un conjunto de figuras de diagnostico que caracterizan el
comportamiento del spline cubico empleado en 'Esclavo_TLS KEYSTREAM.py' para
reconstruir la senal de acoplamiento y_m(t) del maestro.

No requiere MQTT ni conexion al broker: regenera localmente la trayectoria del
maestro con los mismos parametros del experimento, de modo que las figuras son
reproducibles de forma aislada.

Figuras generadas (carpeta Analisis_Interpolacion/):
    01_zoom_interpolante.png       Reconstruccion local: nodos, spline y recta
    02_error_temporal.png          |epsilon(t)| a lo largo de la senal
    03_convergencia_orden.png      Error maximo vs h en log-log (ordenes 2 y 4)
    04_derivadas.png               Continuidad de S'(t) y S''(t)
    05_perfil_subintervalo.png     Distribucion del error dentro de cada tramo
    06_paso_adaptativo.png         Paso elegido por RK45 con cada interpolante
    07_error_sincronizacion.png    |e_x(t)| resultante en el esclavo
    08_piso_vs_h.png               Error de sincronizacion vs error de interpolacion
    09_extrapolacion.png           Comportamiento de fill_value="extrapolate"

Uso:
    python Analisis_Interpolacion.py
================================================================================
"""

import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# ============================================================================
# PARAMETROS DEL EXPERIMENTO (identicos a Maestro_TLS_Principal.py)
# ============================================================================
A, B, C = 0.2, 0.2, 5.7          # Parametros de Rossler
Y0_MAESTRO = [0.1, 0.1, 0.1]     # Condicion inicial del maestro
X0_ESCLAVO = [1.0, 1.0, 1.0]     # Condicion inicial del esclavo
K_ACOPLE = 2.0                   # Ganancia de acoplamiento
H = 0.01                         # Paso de muestreo
TIEMPO_SINC = 6000               # Muestras descartadas (transitorio)
KEYSTREAM = 30000                # Muestras utiles
N_TOTAL = TIEMPO_SINC + KEYSTREAM

RTOL, ATOL = 1e-5, 1e-6

CARPETA = Path("Analisis_Interpolacion")
CARPETA.mkdir(parents=True, exist_ok=True)

DPI = 300


# ============================================================================
# SISTEMAS DINAMICOS
# ============================================================================
def rossler_maestro(t, s, a, b, c):
    """Sistema de Rossler autonomo (emisor)."""
    x, y, z = s
    return [-y - z, x + a * y, b + z * (x - c)]


def rossler_esclavo(t, s, y_interp, a, b, c, k):
    """Sistema de Rossler acoplado (receptor)."""
    x_s, y_s, z_s = s
    y_m = y_interp(t)
    return [-y_s - z_s,
            x_s + a * y_s + k * (y_m - y_s),
            b + z_s * (x_s - c)]


def integrar_maestro(t_final, n_puntos, dense=False):
    """Integra el maestro y devuelve la solucion muestreada."""
    t_eval = np.linspace(0.0, t_final, n_puntos)
    sol = solve_ivp(rossler_maestro, (0.0, t_final), Y0_MAESTRO,
                    args=(A, B, C), t_eval=t_eval, method="RK45",
                    rtol=RTOL, atol=ATOL, dense_output=dense)
    return sol


# ============================================================================
# REFERENCIA DE ALTA PRECISION
# ============================================================================
def referencia_precisa(t_final):
    """
    Solucion de referencia con tolerancias muy estrictas.

    Sirve como 'valor verdadero' contra el cual medir el error de los
    interpolantes. Sus tolerancias son varios ordenes de magnitud mas
    exigentes que las del experimento, de modo que su propio error es
    despreciable en la comparacion.
    """
    sol = solve_ivp(rossler_maestro, (0.0, t_final), Y0_MAESTRO,
                    args=(A, B, C), method="DOP853",
                    rtol=1e-12, atol=1e-14, dense_output=True)
    return sol


# ============================================================================
# FIGURA 01 - RECONSTRUCCION LOCAL
# ============================================================================
def fig_zoom_interpolante(sol_ref):
    """
    Reconstruccion local de la senal, en dos escalas de muestreo.

    Columna izquierda (h = 0.30, didactica): con pocos nodos la diferencia
    geometrica entre spline y recta es visible a simple vista. Sirve para
    explicar el mecanismo.

    Columna derecha (h = 0.01, la del experimento): a la densidad real de
    muestreo las tres curvas son indistinguibles en el panel superior. La
    diferencia solo se aprecia en el panel de error, que abarca varios ordenes
    de magnitud. Esta es justamente la observacion relevante: el efecto es
    invisible en la senal y perfectamente medible en el error.
    """
    # Region de curvatura pronunciada (cresta de la trayectoria)
    t_ref = np.arange(0.0, 120.0, 0.01)
    curv = np.abs(np.gradient(np.gradient(sol_ref.sol(t_ref)[1])))
    t_centro = t_ref[np.argmax(curv)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5),
                             gridspec_kw={"height_ratios": [2, 1]})

    for col, (h_demo, etiqueta) in enumerate([
            (0.30, "$h = 0.30$  (ilustrativo)"),
            (H,    f"$h = {H}$  (experimento)")]):

        t_nodos = np.arange(t_centro - 25 * h_demo, t_centro + 25 * h_demo, h_demo)
        y_nodos = sol_ref.sol(t_nodos)[1]
        f_cub = interp1d(t_nodos, y_nodos, kind="cubic")
        f_lin = interp1d(t_nodos, y_nodos, kind="linear")

        # Ventana de 5 subintervalos centrada en la maxima curvatura
        i0 = len(t_nodos) // 2 - 2
        t_sub = t_nodos[i0:i0 + 6]
        t_fino = np.linspace(t_sub[0], t_sub[-1], 3000)
        y_real = sol_ref.sol(t_fino)[1]

        ax = axes[0, col]
        ax.plot(t_fino, y_real, color="0.3", linewidth=3.0,
                label="Senal real $y_m(t)$")
        ax.plot(t_fino, f_cub(t_fino), "--", color="C0", linewidth=2.0,
                label="Spline cubico $S(t)$")
        ax.plot(t_fino, f_lin(t_fino), ":", color="C3", linewidth=2.0,
                label="Interpolacion lineal $L(t)$")
        ax.plot(t_sub, sol_ref.sol(t_sub)[1], "o", color="black",
                markersize=9, zorder=5, label="Muestras")
        ax.set_title(etiqueta)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel("$y_m(t)$")
            ax.legend(frameon=False, loc="best", fontsize=9)

        ax = axes[1, col]
        e_cub = np.abs(f_cub(t_fino) - y_real)
        e_lin = np.abs(f_lin(t_fino) - y_real)
        ax.semilogy(t_fino, e_lin, color="C3", linewidth=1.8,
                    label=f"Lineal (max {e_lin.max():.1e})")
        ax.semilogy(t_fino, e_cub, color="C0", linewidth=1.8,
                    label=f"Cubico (max {e_cub.max():.1e})")
        for tn in t_sub:
            ax.axvline(tn, color="0.85", linewidth=0.9, zorder=0)
        ax.set_xlabel("Tiempo")
        ax.legend(frameon=False, fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3, which="both")
        if col == 0:
            ax.set_ylabel(r"$|\epsilon(t)|$")

    fig.suptitle("Reconstruccion local de la senal de acoplamiento", y=0.99)
    plt.tight_layout()
    ruta = CARPETA / "01_zoom_interpolante.png"
    plt.savefig(ruta, dpi=DPI)
    plt.close()
    print(f"[FIG 01] {ruta}")


# ============================================================================
# FIGURA 02 - ERROR A LO LARGO DE LA SENAL
# ============================================================================
def fig_error_temporal(sol_ref):
    """
    Error de interpolacion evaluado en los puntos medios de cada subintervalo,
    a lo largo de toda la ventana util del keystream.

    Revela donde se concentra el error: en las regiones de mayor curvatura de
    la trayectoria, no de forma uniforme.
    """
    t_final = N_TOTAL * H
    t_nodos = np.linspace(0, t_final, N_TOTAL)
    y_nodos = sol_ref.sol(t_nodos)[1]

    f_cub = interp1d(t_nodos, y_nodos, kind="cubic")
    f_lin = interp1d(t_nodos, y_nodos, kind="linear")

    t_med = (t_nodos[:-1] + t_nodos[1:]) / 2.0
    y_ref = sol_ref.sol(t_med)[1]
    err_cub = np.abs(f_cub(t_med) - y_ref)
    err_lin = np.abs(f_lin(t_med) - y_ref)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    axes[0].plot(t_med, y_ref, color="0.4", linewidth=0.7)
    axes[0].axvline(TIEMPO_SINC * H, color="C2", linestyle="--", linewidth=1.5)
    axes[0].text(TIEMPO_SINC * H + 3, axes[0].get_ylim()[1] * 0.75,
                 "inicio del keystream", color="C2", fontsize=9)
    axes[0].set_ylabel("$y_m(t)$")
    axes[0].set_title("Error de interpolacion sobre la ventana completa")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_med, err_lin, color="C3", linewidth=0.6,
                 label=f"Lineal  (max = {err_lin.max():.2e})")
    axes[1].plot(t_med, err_cub, color="C0", linewidth=0.6,
                 label=f"Cubico  (max = {err_cub.max():.2e})")
    axes[1].axvline(TIEMPO_SINC * H, color="C2", linestyle="--", linewidth=1.5)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Tiempo")
    axes[1].set_ylabel(r"$|\epsilon(t)|$")
    axes[1].legend(frameon=False, loc="lower right")
    axes[1].grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    ruta = CARPETA / "02_error_temporal.png"
    plt.savefig(ruta, dpi=DPI)
    plt.close()
    print(f"[FIG 02] {ruta}  |  cubico={err_cub.max():.3e}  lineal={err_lin.max():.3e}")
    return err_cub.max(), err_lin.max()


# ============================================================================
# FIGURA 03 - ORDEN DE CONVERGENCIA
# ============================================================================
def fig_convergencia(sol_ref):
    """
    Error maximo en funcion del paso de muestreo h, en escala log-log.

    Es la figura mas concluyente: la pendiente de cada recta es el orden del
    metodo. El spline cubico debe exhibir pendiente 4 y la interpolacion
    lineal pendiente 2. Se incluyen rectas de referencia para comparar.
    """
    t_final = 50.0  # Ventana corta: basta para caracterizar el orden
    pasos = np.array([0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002])
    err_cub, err_lin = [], []

    for h in pasos:
        t_nodos = np.arange(0.0, t_final, h)
        y_nodos = sol_ref.sol(t_nodos)[1]
        f_cub = interp1d(t_nodos, y_nodos, kind="cubic")
        f_lin = interp1d(t_nodos, y_nodos, kind="linear")

        t_med = (t_nodos[:-1] + t_nodos[1:]) / 2.0
        y_ref = sol_ref.sol(t_med)[1]
        err_cub.append(np.abs(f_cub(t_med) - y_ref).max())
        err_lin.append(np.abs(f_lin(t_med) - y_ref).max())

    err_cub, err_lin = np.array(err_cub), np.array(err_lin)

    # Pendientes empiricas por ajuste lineal en log-log
    p_cub = np.polyfit(np.log(pasos), np.log(err_cub), 1)[0]
    p_lin = np.polyfit(np.log(pasos), np.log(err_lin), 1)[0]

    plt.figure(figsize=(9, 7))
    plt.loglog(pasos, err_lin, "o-", color="C3", linewidth=1.8, markersize=7,
               label=f"Lineal   (pendiente ajustada = {p_lin:.2f})")
    plt.loglog(pasos, err_cub, "s-", color="C0", linewidth=1.8, markersize=7,
               label=f"Cubico  (pendiente ajustada = {p_cub:.2f})")

    # Rectas de referencia de orden 2 y 4
    plt.loglog(pasos, err_lin[0] * (pasos / pasos[0]) ** 2, "--",
               color="0.6", linewidth=1.2, label=r"referencia $\mathcal{O}(h^2)$")
    plt.loglog(pasos, err_cub[0] * (pasos / pasos[0]) ** 4, ":",
               color="0.6", linewidth=1.2, label=r"referencia $\mathcal{O}(h^4)$")

    plt.axvline(H, color="C2", linestyle="-.", linewidth=1.5)
    plt.text(H * 1.15, err_cub.min() * 3, "h del experimento",
             color="C2", fontsize=9, rotation=90, va="bottom")

    plt.xlabel("Paso de muestreo $h$")
    plt.ylabel("Error maximo de interpolacion")
    plt.title("Orden de convergencia de los interpolantes")
    plt.legend(frameon=False)
    plt.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    ruta = CARPETA / "03_convergencia_orden.png"
    plt.savefig(ruta, dpi=DPI)
    plt.close()
    print(f"[FIG 03] {ruta}  |  pendiente cubico={p_cub:.2f}  lineal={p_lin:.2f}")


# ============================================================================
# FIGURA 04 - CONTINUIDAD DE LAS DERIVADAS
# ============================================================================
def fig_derivadas(sol_ref):
    """
    Primera y segunda derivada de ambos interpolantes en una region reducida.

    Justifica visualmente el argumento de suavidad: la derivada de la
    interpolacion lineal es una escalera con saltos en cada nodo, lo que
    penaliza al controlador de paso adaptativo de RK45. El spline cubico
    mantiene ambas derivadas continuas.
    """
    t_nodos = np.arange(20.0, 21.0, H)
    y_nodos = sol_ref.sol(t_nodos)[1]
    f_cub = interp1d(t_nodos, y_nodos, kind="cubic")
    f_lin = interp1d(t_nodos, y_nodos, kind="linear")

    # Ventana de 8 subintervalos para que los saltos sean visibles
    t0, t1 = t_nodos[20], t_nodos[28]
    t_fino = np.linspace(t0, t1, 4000)
    dt = t_fino[1] - t_fino[0]

    d1_cub = np.gradient(f_cub(t_fino), dt)
    d1_lin = np.gradient(f_lin(t_fino), dt)
    d2_cub = np.gradient(d1_cub, dt)

    # np.gradient usa diferencias unilaterales en los extremos, lo que produce
    # artefactos en la segunda derivada. Se recortan los bordes.
    m = slice(30, -30)

    nodos_ventana = t_nodos[(t_nodos >= t0) & (t_nodos <= t1)]

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(t_fino, f_lin(t_fino), color="C3", linewidth=3.0,
                 label="Lineal")
    axes[0].plot(t_fino, f_cub(t_fino), "--", color="C0", linewidth=1.8,
                 label="Cubico")
    axes[0].set_ylabel("$S(t)$")
    axes[0].set_title("Continuidad de las derivadas del interpolante\n"
                      "(ambas funciones coinciden a simple vista; sus derivadas no)",
                      fontsize=11)
    axes[0].legend(frameon=False)

    axes[1].plot(t_fino, d1_lin, color="C3", linewidth=1.8,
                 label="Lineal: $S'$ escalonada, salta en cada nodo")
    axes[1].plot(t_fino, d1_cub, color="C0", linewidth=1.8,
                 label="Cubico: $S'$ continua")
    axes[1].set_ylabel("$S'(t)$")
    axes[1].legend(frameon=False, fontsize=9, loc="upper right")

    axes[2].plot(t_fino[m], d2_cub[m], color="C0", linewidth=1.8,
                 label="Cubico: $S''$ continua")
    axes[2].set_ylabel("$S''(t)$")
    axes[2].set_xlabel("Tiempo")
    axes[2].legend(frameon=False, fontsize=9, loc="upper right")
    axes[2].text(0.02, 0.08,
                 "La lineal no aparece aqui: su segunda derivada es nula\n"
                 "dentro de cada tramo e indefinida en los nodos.",
                 transform=axes[2].transAxes, fontsize=9, color="C3")

    for ax in axes:
        for tn in nodos_ventana:
            ax.axvline(tn, color="0.85", linewidth=0.8, zorder=0)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    ruta = CARPETA / "04_derivadas.png"
    plt.savefig(ruta, dpi=DPI)
    plt.close()
    print(f"[FIG 04] {ruta}")


# ============================================================================
# FIGURA 05 - PERFIL DEL ERROR DENTRO DEL SUBINTERVALO
# ============================================================================
def fig_perfil_subintervalo(sol_ref):
    """
    Error en funcion de la posicion relativa s = (t - t_i)/h dentro del tramo.

    Cada subintervalo aporta una curva; superponerlas revela el perfil
    caracteristico de cada metodo. La interpolacion lineal presenta un unico
    maximo centrado (s = 0.5); el spline cubico presenta un perfil de dos
    lobulos y se anula en los nodos y cerca de ellos.
    """
    t_final = 60.0
    t_nodos = np.arange(0.0, t_final, H)
    y_nodos = sol_ref.sol(t_nodos)[1]
    f_cub = interp1d(t_nodos, y_nodos, kind="cubic")
    f_lin = interp1d(t_nodos, y_nodos, kind="linear")

    s = np.linspace(0.0, 1.0, 41)
    n_tramos = 400          # Muestra representativa de subintervalos
    salto = max(len(t_nodos) // n_tramos, 1)

    perf_cub = np.zeros((0, len(s)))
    perf_lin = np.zeros((0, len(s)))

    for i in range(1, len(t_nodos) - 2, salto):
        t_ev = t_nodos[i] + s * H
        ref = sol_ref.sol(t_ev)[1]
        perf_cub = np.vstack([perf_cub, np.abs(f_cub(t_ev) - ref)])
        perf_lin = np.vstack([perf_lin, np.abs(f_lin(t_ev) - ref)])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, perf, color, titulo in [
        (axes[0], perf_lin, "C3", "Interpolacion lineal"),
        (axes[1], perf_cub, "C0", "Spline cubico"),
    ]:
        # Normalizacion por tramo para comparar la forma, no la magnitud
        norm = perf / (perf.max(axis=1, keepdims=True) + 1e-300)
        for fila in norm[::10]:
            ax.plot(s, fila, color=color, alpha=0.12, linewidth=0.8)
        ax.plot(s, norm.mean(axis=0), color="black", linewidth=2.2,
                label="perfil promedio")
        ax.set_xlabel("Posicion relativa dentro del tramo  $s=(t-t_i)/h$")
        ax.set_title(titulo)
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Error normalizado")
    fig.suptitle("Distribucion del error dentro de cada subintervalo", y=1.00)
    plt.tight_layout()
    ruta = CARPETA / "05_perfil_subintervalo.png"
    plt.savefig(ruta, dpi=DPI)
    plt.close()
    print(f"[FIG 05] {ruta}")


# ============================================================================
# FIGURA 06 - EFECTO SOBRE EL PASO ADAPTATIVO
# ============================================================================
def fig_paso_adaptativo(sol_ref):
    """
    Costo de integracion en funcion de la tolerancia exigida al integrador.

    Los quiebres de derivada de la interpolacion lineal penalizan al
    controlador de paso adaptativo de RK45, pero SOLO cuando la tolerancia es
    lo bastante estricta como para que el controlador los perciba. Esta figura
    delimita ese umbral: a las tolerancias del experimento (rtol=1e-5) ambos
    interpolantes cuestan lo mismo, y la ventaja del spline aparece a partir
    de rtol ~ 1e-8.

    Es un resultado importante para no sobrevender el argumento de suavidad.
    """
    t_final = 60.0
    t_nodos = np.arange(0.0, t_final, H)
    y_nodos = sol_ref.sol(t_nodos)[1]

    tolerancias = [(1e-5, 1e-6), (1e-6, 1e-8), (1e-7, 1e-9),
                   (1e-8, 1e-10), (1e-9, 1e-11), (1e-10, 1e-12),
                   (1e-11, 1e-13)]
    datos = {"cubic": [], "linear": []}

    for rtol, atol in tolerancias:
        for kind in ("cubic", "linear"):
            f = interp1d(t_nodos, y_nodos, kind=kind, fill_value="extrapolate")
            sol = solve_ivp(rossler_esclavo, (0.0, t_final - 1.0), X0_ESCLAVO,
                            args=(f, A, B, C, K_ACOPLE), method="RK45",
                            rtol=rtol, atol=atol)
            datos[kind].append(sol.nfev)
        print(f"[FIG 06] rtol={rtol:<8} cubico={datos['cubic'][-1]:>7} eval   "
              f"lineal={datos['linear'][-1]:>7} eval   "
              f"sobrecosto={datos['linear'][-1]/datos['cubic'][-1]:.2f}x")

    rtols = np.array([t[0] for t in tolerancias])
    cub = np.array(datos["cubic"], dtype=float)
    lin = np.array(datos["linear"], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    axes[0].loglog(rtols, cub, "s-", color="C0", linewidth=1.8, markersize=8,
                   label="Spline cubico")
    axes[0].loglog(rtols, lin, "o-", color="C3", linewidth=1.8, markersize=8,
                   label="Interpolacion lineal")
    axes[0].axvline(RTOL, color="C2", linestyle="--", linewidth=1.6)
    axes[0].text(RTOL * 0.55, cub.max() * 0.5, "rtol del\nexperimento",
                 color="C2", fontsize=9, ha="right")
    axes[0].invert_xaxis()
    axes[0].set_xlabel("Tolerancia relativa exigida  (mas estricta $\\rightarrow$)")
    axes[0].set_ylabel("Evaluaciones del campo vectorial")
    axes[0].set_title("Costo de integracion")
    axes[0].legend(frameon=False)
    axes[0].grid(True, alpha=0.3, which="both")

    axes[1].semilogx(rtols, lin / cub, "d-", color="0.25", linewidth=1.8,
                     markersize=8)
    axes[1].axhline(1.0, color="0.6", linestyle="--", linewidth=1.2)
    axes[1].axvline(RTOL, color="C2", linestyle="--", linewidth=1.6)
    axes[1].invert_xaxis()
    axes[1].fill_between(rtols, 1.0, lin / cub, where=(lin / cub > 1.05),
                         color="C3", alpha=0.15)
    axes[1].set_xlabel("Tolerancia relativa exigida  (mas estricta $\\rightarrow$)")
    axes[1].set_ylabel("Sobrecosto de la lineal  (x veces)")
    axes[1].set_title("La penalizacion aparece solo con tolerancias estrictas")
    axes[1].grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    ruta = CARPETA / "06_paso_adaptativo.png"
    plt.savefig(ruta, dpi=DPI)
    plt.close()
    print(f"[FIG 06] {ruta}")


# ============================================================================
# FIGURA 07 - ERROR DE SINCRONIZACION RESULTANTE
# ============================================================================
def fig_error_sincronizacion(sol_ref):
    """
    Error |e_x(t)| = |x_m(t) - x_s(t)| con cada interpolante.

    Muestra las dos fases del proceso: el decaimiento exponencial del
    transitorio, comun a ambos casos, y el piso residual posterior, que
    depende directamente de la calidad de la interpolacion.
    """
    t_final = N_TOTAL * H
    t_nodos = np.linspace(0.0, t_final, N_TOTAL)
    y_nodos = sol_ref.sol(t_nodos)[1]
    x_ref = sol_ref.sol(t_nodos)[0]

    plt.figure(figsize=(11, 6))
    resumen = {}

    for nombre, kind, color in [("Lineal", "linear", "C3"),
                                ("Cubico", "cubic", "C0")]:
        f = interp1d(t_nodos, y_nodos, kind=kind, fill_value="extrapolate")
        sol = solve_ivp(rossler_esclavo, (0.0, t_final), X0_ESCLAVO,
                        args=(f, A, B, C, K_ACOPLE), t_eval=t_nodos,
                        method="RK45", rtol=RTOL, atol=ATOL)
        err = np.abs(x_ref - sol.y[0])
        piso = np.median(err[TIEMPO_SINC:])
        resumen[nombre] = piso
        plt.semilogy(t_nodos, err, color=color, linewidth=0.7,
                     label=f"{nombre}  (piso mediano = {piso:.2e})")

    plt.axvline(TIEMPO_SINC * H, color="C2", linestyle="--", linewidth=1.6)
    plt.text(TIEMPO_SINC * H + 4, plt.ylim()[1] * 0.2,
             "fin del transitorio\n(TIEMPO_SINC)", color="C2", fontsize=9)

    # Umbral por debajo del cual ningun pixel cambia de valor
    umbral = 0.5 / (100.0 * 255.0)
    plt.axhline(umbral, color="0.35", linestyle="-.", linewidth=1.4)
    plt.text(t_final * 0.55, umbral * 1.4,
             f"umbral de alteracion de 1 pixel  ({umbral:.1e})",
             fontsize=9, color="0.35")

    plt.xlabel("Tiempo")
    plt.ylabel("$|x_m(t) - x_s(t)|$")
    plt.title("Error de sincronizacion segun el interpolante empleado")
    plt.legend(frameon=False, loc="upper right")
    plt.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    ruta = CARPETA / "07_error_sincronizacion.png"
    plt.savefig(ruta, dpi=DPI)
    plt.close()
    print(f"[FIG 07] {ruta}  |  " +
          "  ".join(f"{k}={v:.3e}" for k, v in resumen.items()))


# ============================================================================
# FIGURA 08 - PISO DE SINCRONIZACION VS ERROR DE INTERPOLACION
# ============================================================================
def fig_piso_vs_h(sol_ref):
    """
    Relacion entre el error de interpolacion y el piso de sincronizacion,
    barriendo el paso de muestreo h.

    El piso resulta ser proporcional al error de interpolacion, pero con una
    constante de proporcionalidad muy inferior a la que predice un analisis
    estatico. La estimacion de ganancia continua K/|a-K| = 1.11 supone una
    perturbacion lenta; en realidad epsilon(t) oscila a la frecuencia de los
    nodos, y la dinamica del error (polo en -(K-a) = -1.8) la atenua como un
    filtro pasa-bajas:

        |G(jw)| = K / sqrt(w^2 + (K-a)^2),   w ~ 2*pi/h

    Para h = 0.01 esto da |G| ~ 3e-3, del mismo orden que la razon medida.
    """
    t_final = 120.0
    n_desc = int(60.0 / H)   # Transitorio a descartar
    pasos = np.array([0.08, 0.05, 0.03, 0.02, 0.01])

    eps_list, piso_list = [], []

    for h in pasos:
        t_nodos = np.arange(0.0, t_final, h)
        y_nodos = sol_ref.sol(t_nodos)[1]
        f = interp1d(t_nodos, y_nodos, kind="cubic", fill_value="extrapolate")

        # Error de interpolacion
        t_med = (t_nodos[:-1] + t_nodos[1:]) / 2.0
        eps = np.abs(f(t_med) - sol_ref.sol(t_med)[1]).max()

        # Piso de sincronizacion (integrando con tolerancias estrictas para
        # que el limitante sea la interpolacion y no el integrador)
        t_out = np.arange(0.0, t_final - 1.0, H)
        sol = solve_ivp(rossler_esclavo, (0.0, t_final - 1.0), X0_ESCLAVO,
                        args=(f, A, B, C, K_ACOPLE), t_eval=t_out,
                        method="RK45", rtol=1e-10, atol=1e-12)
        err = np.abs(sol_ref.sol(t_out)[0] - sol.y[0])
        piso = np.median(err[n_desc:])

        eps_list.append(eps)
        piso_list.append(piso)
        print(f"[FIG 08] h={h:<6} eps={eps:.3e}  piso={piso:.3e}  "
              f"razon={piso/eps:.2f}")

    eps_arr, piso_arr = np.array(eps_list), np.array(piso_list)
    ganancia = np.median(piso_arr / eps_arr)

    plt.figure(figsize=(9, 7))
    plt.loglog(eps_arr, piso_arr, "o-", color="C0", linewidth=1.8,
               markersize=9, label=f"Medido  (ganancia = {ganancia:.3f})")

    plt.loglog(eps_arr, 1.11 * eps_arr, "--", color="C3", linewidth=1.5,
               label=r"Estimacion estatica  $\frac{K}{|a-K|}=1.11$  (no aplica)")
    plt.loglog(eps_arr, ganancia * eps_arr, ":", color="0.4", linewidth=1.5,
               label="Proporcionalidad medida")

    for h, x, y in zip(pasos, eps_arr, piso_arr):
        plt.annotate(f"h={h}", (x, y), textcoords="offset points",
                     xytext=(8, -12), fontsize=9)

    plt.xlabel(r"Error de interpolacion  $\|\epsilon\|_\infty$")
    plt.ylabel("Piso del error de sincronizacion")
    plt.title("El piso de sincronizacion es proporcional al error de interpolacion\n"
              "(atenuado por el filtrado pasa-bajas de la dinamica del error)",
              fontsize=11)
    plt.legend(frameon=False, fontsize=9)
    plt.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    ruta = CARPETA / "08_piso_vs_h.png"
    plt.savefig(ruta, dpi=DPI)
    plt.close()
    print(f"[FIG 08] {ruta}")


# ============================================================================
# FIGURA 09 - COMPORTAMIENTO DE LA EXTRAPOLACION
# ============================================================================
def fig_extrapolacion(sol_ref):
    """
    Comportamiento de fill_value='extrapolate' mas alla del ultimo nodo.

    Justifica por que esa opcion es admisible en el codigo: el integrador
    excede el rango a lo sumo en una fraccion de h, region en la que el error
    sigue siendo despreciable. A distancias mayores el polinomio diverge.
    """
    t_fin_nodos = 40.0
    t_nodos = np.arange(0.0, t_fin_nodos, H)
    y_nodos = sol_ref.sol(t_nodos)[1]
    f_cub = interp1d(t_nodos, y_nodos, kind="cubic", fill_value="extrapolate")

    t_ext = np.linspace(t_nodos[-1], t_nodos[-1] + 0.6, 1500)
    y_real = sol_ref.sol(t_ext)[1]
    y_ext = f_cub(t_ext)
    distancia = (t_ext - t_nodos[-1]) / H   # en multiplos de h

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(distancia, y_real, color="0.3", linewidth=2.2,
                 label="Senal real")
    axes[0].plot(distancia, y_ext, "--", color="C0", linewidth=1.8,
                 label="Extrapolacion del spline")
    axes[0].axvspan(0, 1, color="C2", alpha=0.15)
    axes[0].text(0.12, axes[0].get_ylim()[0] * 0.9,
                 "rango real de uso\n($<1\\,h$)", fontsize=9, color="C2")
    axes[0].set_ylabel("$y(t)$")
    axes[0].set_title("Extrapolacion mas alla del ultimo nodo")
    axes[0].legend(frameon=False)
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(distancia, np.abs(y_ext - y_real), color="C0",
                     linewidth=1.8)
    axes[1].axvspan(0, 1, color="C2", alpha=0.15)
    axes[1].set_xlabel("Distancia mas alla del ultimo nodo (en multiplos de $h$)")
    axes[1].set_ylabel("Error de extrapolacion")
    axes[1].grid(True, alpha=0.3, which="both")

    err_1h = np.abs(y_ext - y_real)[distancia <= 1].max()
    err_max = np.abs(y_ext - y_real).max()

    plt.tight_layout()
    ruta = CARPETA / "09_extrapolacion.png"
    plt.savefig(ruta, dpi=DPI)
    plt.close()
    print(f"[FIG 09] {ruta}  |  error a 1h={err_1h:.3e}  "
          f"error a 60h={err_max:.3e}")


# ============================================================================
# PROGRAMA PRINCIPAL
# ============================================================================
def main():
    t0 = time.perf_counter()
    print("=" * 70)
    print("ANALISIS GRAFICO DE LA INTERPOLACION CUBICA")
    print("=" * 70)

    print("[REF] Calculando solucion de referencia de alta precision...")
    sol_ref = referencia_precisa(N_TOTAL * H + 5.0)
    print(f"[REF] Listo ({sol_ref.nfev} evaluaciones)")

    fig_zoom_interpolante(sol_ref)
    fig_error_temporal(sol_ref)
    fig_convergencia(sol_ref)
    fig_derivadas(sol_ref)
    fig_perfil_subintervalo(sol_ref)
    fig_paso_adaptativo(sol_ref)
    fig_error_sincronizacion(sol_ref)
    fig_piso_vs_h(sol_ref)
    fig_extrapolacion(sol_ref)

    print("=" * 70)
    print(f"Completado en {time.perf_counter() - t0:.1f} s")
    print(f"Figuras en: {CARPETA.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
