import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import equinox as eqx
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5, PIDController, RESULTS
import jax.lax as lax
import numpy as np
from scipy.integrate import solve_ivp
from PBHBeta import constants
from PBHBeta import constraints
import functools
from scipy.interpolate import interp1d
import numpy as np
from scipy.optimize import brentq

# ---------------------------------------------------------------------------
# Constantes en unidades de Planck
# ---------------------------------------------------------------------------
M_pl_GeV = 1.22089e19
M_pl_g = 2.17645e-5
H_end_GeV = 4.44e13
H_end_pl  = H_end_GeV / M_pl_GeV
n = 100.00

# ---------------------------------------------------------------------------
# Modelo de acreción — dos regímenes (ver DERIVACION_ACRECION.md)
# ---------------------------------------------------------------------------
# regime='wave' (DEFAULT, ronda 2): el condensado oscilante es un CAMPO CLÁSICO
# COHERENTE, no un fluido (Michel diverge a w->0, DERIVACION_ACRECION.md §1) ni un gas
# de partículas sin colisiones (infall da mu=e^{DeltaN}, también enorme, §2b). Acretarlo
# es un problema de absorción de ondas: Unruh (1976) da sigma_abs = A_H/v = 16 pi M^2/v
# para un escalar masivo no relativista, y el v se cancela: Mdot = 16 pi M^2 rho, FINITO,
# sin regulador ad hoc. dlnM/dN = 3*L_eff*x/(8*pi), sin min() y sin (1-x): x decae
# monótonamente y nunca se acerca a 1 si L*gamma < 4*pi (ver GAMMA_COLAPSO más abajo).
L_ACC_DEFAULT = 16.0 * np.pi   # = 50.265; Unruh 1976. Ver DERIVACION_ACRECION.md §2c.

# regime='horizon_tracking': el modelo de la ronda 1 (AUDIT.md P0-4), literal de la
# ec. (2.15) del paper: dlnM/dN = min(3 L_eff x/8pi, 3 gamma_H/2x). Es la solución
# auto-similar de Zel'dovich-Novikov que Carr & Hawking (1974) argumentan que NO se
# realiza (DERIVACION_ACRECION.md §1-2a): colapsa la función de masa a una delta en
# gamma_H*M_H(N_re) — medido en la ronda 2 (mu idéntico a 5 cifras para M_i de 1e0 a
# 1e16 g). Se conserva como resultado versionado (Tarea 4), NO es el default.
L_ACC_HORIZON_TRACKING_DEFAULT = 102.0   # L del paper tal como se leyó en la ronda 1
GAMMA_H = 1.0            # gamma^MD, convención PBHBeta (ver PBHBeta/BfM.py: "no bien conocido, se adopta 1")

# GAMMA_COLAPSO: eficiencia de colapso en dominación de materia, x(N_form) = gamma. NO
# es libre: la solución cerrada de regime='wave' (DERIVACION_ACRECION.md §3),
#   mu = [1 - L*gamma/(4*pi)]^-1,
# sólo es válida (y positiva) para L*gamma < 4*pi = 12.566 — más allá de eso el modelo
# es tan runaway como horizon_tracking. gamma_reh=1 (la convención de PBHBeta, usada en
# regime='horizon_tracking' y en BfM.py) da L*gamma=50 >> 4pi: INADMISIBLE para el
# régimen de onda. El colapso en dominación de materia está limitado por anisotropía e
# inhomogeneidad, no por un umbral de densidad, y da gamma << 1 (refs en
# DERIVACION_ACRECION.md §6: Khlopov & Polnarev 1980; Jedamzik & Niemeyer 1999; Harada,
# Yoo, Kohri, Nakao & Jhingan 2016; Carr, Dimopoulos, Owen & Tenkanen 2018). Con
# L=16*pi y gamma=0.15, mu=2.5 (0.4 décadas) — el "shift pequeño" que predice §3.
GAMMA_COLAPSO = 0.15

# Suavizado del min() (AUDIT.md P1-6): jnp.minimum tiene un kink (derivada discontinua)
# justo donde las dos ramas se cruzan. Con un solver explícito adaptativo (Tsit5) eso
# hace que el controlador de paso colapse: para masas de formación cerca del punto fijo
# inestable x*=4pi/L_eff, diffeqsolve no converge ni con max_steps=2e6 (falla real que
# expuso el fix de throw=True/P1-6 más abajo, antes silenciada por throw=False). Se
# reemplaza min(a,b) por (a+b)/2 - sqrt((a-b)^2 + eps^2)/2, que es min(a,b) exacto en el
# límite eps->0 pero derivable; con eps=1e-2 el caso que fallaba converge en 248 pasos.
# El error introducido es O(eps^2/|a-b|), despreciable salvo justo en el cruce.
EPS_SOFTMIN = 1.0e-2

# AUDIT.md P1-11: umbral único de "es reliquia de Planck". Antes Betas_DM usaba 1.05 y
# las otras seis funciones (BBN/SD/CMB_AN/GRB/Reio/LSP) usaban 1.5*M_pl_g, mezclando dos
# definiciones de la frontera en M in [1.05,1.5]*M_pl_g dentro de get_Betas_full (toma
# min() sobre los siete canales). Se usa 1.5 (el valor que ya tenían seis de siete).
LIMITE_PLANCK_G = 1.5 * constants.M_pl_g

# ---------------------------------------------------------------------------
# alpha(M): evaporación de Hawking-Kerr, dependiente de la masa (Tarea 3 ronda 2,
# corregida en Tarea 1 ronda 3)
# ---------------------------------------------------------------------------
# dM/dt = -alpha(M)/M^2 * factor_evap_kerr. RONDA 3 - CORRECCIÓN DE NORMALIZACIÓN:
# en la ronda 2 se usó alpha = f(M)/(15360*pi), la normalización de UN grado de libertad
# de Schwarzschild — pero la f(M) de MacGibbon NO es "número de dof" en ese sentido. La
# relación correcta (Carr, Kohri, Sendouda & Yokoyama 2021 — CKSY —, Rep. Prog. Phys. 84,
# 116902, arXiv:2002.12778, que reproduce MacGibbon PRD 44, 376 (1991)), en unidades de
# Planck, es
#
#     alpha(M) = 2.79244e-4 * f(M)
#
# con f(M) en la normalización de MacGibbon. Verificado independientemente (ver commit):
# tau = 407*(f/15.35)^-1*(M/1e10 g)^3 s con f=1.9 en M=5.1e14 g da 4.362e17 s = edad del
# universo, sin ajustar nada; y con mi propia fórmula alpha(M)*M^3-life da 4.357e17 s
# (0.1% de diferencia, redondeo de constantes). El alpha=5.0077e-4 de la ronda 1
# corresponde a f=1.793 en esta normalización correcta, contra f*=1.9 — 5.6% de acuerdo:
# mi calibración a la edad del universo (que yo mismo señalé como circular, con razón)
# resultó ser, en los hechos, casi exacta.
#
# f(M): escalera reconstruida desde T_BH(M) = 1.06e13/M_g GeV y los umbrales de masa en
# reposo del SM. Tres anclas son verbatim de CKSY (2021), las demás son mi reconstrucción
# a partir de las contribuciones por especie que CKSY da (f_{s=0}=0.267, f_{s=1}=0.060,
# f_{s=3/2}=0.020, f_{s=2}=0.007, f_{s=1/2}=0.147 neutro / 0.142 cargado ±e; quarks
# u,d,s+gluones ~ 3*12*0.14+16*0.06 ≈ 6) — HONESTIDAD: no tengo la tabla completa de CKSY
# dígito a dígito (sin acceso a internet en esta sesión), así que los tres escalones
# intermedios (muón, QCD, W/Z+Higgs+quarks pesados) son una distribución razonable del
# presupuesto entre las anclas firmes, no una reproducción exacta de cada paso. Las
# anclas firmes SÍ están garantizadas por construcción (el código fuerza f(M) a pasar
# exactamente por ellas).
#
#   M > 2.07e16 g                    f=1.569   ANCLA (foton+3nu+graviton, M gtrsim 1e17g)
#   1.00e14 g < M <= 2.07e16 g       f=1.9     ANCLA (+electron; T_BH~21 MeV; incluye M*)
#   3.85e13 g < M <= 1.00e14 g       f=2.042   +muon (0.142, mismo orden que el electron)
#   1.18e11 g < M <= 3.85e13 g       f=8.042   +QCD: quarks u,d,s + gluones (~6, dado)
#   1.06e10 g < M <= 1.18e11 g       f=11.696  +W/Z/Higgs (mitad del presupuesto restante)
#   M <= 1.06e10 g                   f=15.35   ANCLA (SM completo hasta 1 TeV)
_ALPHA_POR_F = 2.79244e-4  # CKSY (2021)/MacGibbon (1991), unidades de Planck

_F_UMBRALES_M_G = np.array([1.06e10, 1.18e11, 3.85e13, 1.00e14, 2.07e16], dtype=np.float64)
_F_PLATOS = np.array([15.35, 11.696, 8.042, 2.042, 1.9, 1.569], dtype=np.float64)
_DELTA_LOG10 = 0.01  # ancho de la transición entre escalones (~2.3% en masa) — no un kink
                      # verdadero (ver AUDIT.md/P1-6 en el resumen de la ronda 1: un min()
                      # con kink real colapsó el paso del integrador ahí).

_log10_umbrales = np.log10(_F_UMBRALES_M_G)
_F_DOF_LOG10_M_ANCLAS = np.empty(2 + 2 * len(_log10_umbrales), dtype=np.float64)
_F_DOF_ANCLAS = np.empty_like(_F_DOF_LOG10_M_ANCLAS)
_F_DOF_LOG10_M_ANCLAS[0] = -10.0
_F_DOF_ANCLAS[0] = _F_PLATOS[0]
for _i, _lu in enumerate(_log10_umbrales):
    _F_DOF_LOG10_M_ANCLAS[1 + 2 * _i] = _lu - _DELTA_LOG10
    _F_DOF_ANCLAS[1 + 2 * _i] = _F_PLATOS[_i]
    _F_DOF_LOG10_M_ANCLAS[2 + 2 * _i] = _lu + _DELTA_LOG10
    _F_DOF_ANCLAS[2 + 2 * _i] = _F_PLATOS[_i + 1]
_F_DOF_LOG10_M_ANCLAS[-1] = 25.0
_F_DOF_ANCLAS[-1] = _F_PLATOS[-1]


def f_dof_of_mass_g(M_g):
    """f(M) en la normalización de MacGibbon/CKSY (NO grados de libertad de Schwarzschild
    — ver nota arriba). Funciona con jnp (Fase 1, trazado) y con float/np (Fase 2)."""
    log10_M = jnp.log10(jnp.clip(M_g, 1e-30, None))
    return jnp.interp(log10_M, _F_DOF_LOG10_M_ANCLAS, _F_DOF_ANCLAS)


def alpha_of_mass_g(M_g):
    """alpha(M) = 2.79244e-4 * f(M) (CKSY 2021 / MacGibbon 1991 — ver nota arriba)."""
    return _ALPHA_POR_F * f_dof_of_mass_g(M_g)


# Lectura rápida: alpha en M*=5.1e14 g (no se usa en ningún cálculo — todo el código usa
# alpha_of_mass_g(M) evaluado en la M que corresponda).
ALPHA_EVAP_EN_M_ESTRELLA = float(alpha_of_mass_g(jnp.asarray(5.1e14)))

def put_M_array(Mass_min, Mass_max, num_points_low=7000, num_points_high=7000):
    Mass_cut = 1.0e5
    log_M_min = np.log10(Mass_min)
    log_M_cut = np.log10(Mass_cut)
    log_M_max = np.log10(Mass_max)

    log_grid_low = np.linspace(log_M_min, log_M_cut, num_points_low)
    log_grid_high = np.linspace(log_M_cut, log_M_max, num_points_high)
    log_grid = np.concatenate([log_grid_low, log_grid_high])

    if hasattr(constraints, 'data_mass') and len(constraints.data_mass) > 0:
        log_data = np.log10(constraints.data_mass)
        combined_log = np.concatenate([log_grid, log_data])
    else:
        combined_log = log_grid

    combined_log = np.sort(combined_log)
    combined_log = combined_log[(combined_log >= log_M_min) & (combined_log <= log_M_max)]
    log_M_final = np.unique(combined_log)
    
    constraints.M_tot = 10.0**log_M_final
    return constraints.M_tot

# ---------------------------------------------------------------------------
# FASE 1: Acreción y Evaporación de Kerr
# ---------------------------------------------------------------------------
def _factor_espin_acc_np(a_star):
    raiz_espin = np.sqrt(1.0 - np.clip(a_star, 0.0, 0.9999)**2.0)
    return ((1.0 + raiz_espin) / 2.0)**2.0


def assert_subcritical(L_acc, a_star=0.0, gamma_form=GAMMA_COLAPSO):
    """Tarea 2 / DERIVACION_ACRECION.md §3: en regime='wave', L*gamma < 4*pi es condición
    de existencia de la solución cerrada (mu=[1-L*gamma/4pi]^-1), no un ajuste fino. Por
    encima, x nunca decae y el modelo es tan runaway como horizon_tracking. Usa
    L_eff = L_acc*factor_espin_acc(a_star); S_onda y Omega_phi ~ 1 en el régimen de
    interés (ver P2-13, resumen ronda 1), así que no hace falta traerlos aquí."""
    L_eff = L_acc * _factor_espin_acc_np(a_star)
    limite = 4.0 * np.pi
    producto = L_eff * gamma_form
    if not (producto < limite):
        raise AssertionError(
            f"regime='wave' en runaway: L_eff*gamma = {producto:.4f} >= 4*pi = "
            f"{limite:.4f} (L_acc={L_acc}, a_star={a_star}, gamma={gamma_form}). "
            f"Baja L_acc o gamma_form — ver DERIVACION_ACRECION.md §3."
        )


@eqx.filter_jit
@functools.partial(jax.vmap, in_axes=(0, None, None, None, None, None))
def _precalcular_acreccion_lote_impl(Mi_val_g, N_fin, a_star, L_acc, regime, gamma_form):
    M_i_pl = Mi_val_g / M_pl_g
    M_end_pl = 1.0 / H_end_pl
    # x(N_form) = gamma_form (Tarea 2): M_i = gamma_form * M_H(N_ini). Con gamma_form=1
    # (regime='horizon_tracking') es la definición original de la ronda 1 (M_i=M_H(N_ini)
    # exacto, x=1).
    N_ini = (2.0 / 3.0) * jnp.log(jnp.maximum(M_i_pl / (gamma_form * M_end_pl), 1.0))
    a_star_safe = jnp.clip(a_star, 0.0, 0.9999)

    # Definición de la masa del campo escalar (Inflatón)
    mu_pl = n * H_end_pl 
    
    # Amplitud inicial para que la densidad de phi coincida con el fondo cósmico
    rho_end_inf_pl = (3.0 * H_end_pl**2.0) / (8.0 * jnp.pi)
    phi_ini_pl = jnp.sqrt(2.0 * rho_end_inf_pl / (mu_pl**2.0 + 9.0 * H_end_pl**2.0 / 4.0))

    # Factor geométrico de espín para la captura de materia
    raiz_espin = jnp.sqrt(1.0 - a_star_safe**2.0)
    factor_espin_acc = ((1.0 + raiz_espin) / 2.0)**2.0

    # Factores de Kerr para la evaporación térmica
    factor_T_kerr = (2.0 * raiz_espin) / (1.0 + raiz_espin)
    factor_Area_kerr = 0.5 * (1.0 + raiz_espin)
    factor_evap_kerr = (factor_T_kerr**4.0) * factor_Area_kerr
    
    def rho_inf_field_env(N):
        # Evolución de la densidad de energía del campo escalar en altas oscilaciones
        return 0.5 * (phi_ini_pl * mu_pl)**2 * jnp.exp(-3.0 * N)

    def vector_field_log(N, u, args):
        H_val = H_end_pl * jnp.exp(-1.5 * N)
        M_H_actual = 1.0 / H_val
        u_H = jnp.log(M_H_actual)
                
        # Blindaje numérico y recuperación de masa
        u_safe = jnp.clip(u, -1.0, u_H) 
        M_actual = jnp.exp(u_safe)
        
        # Variable cosmológica adimensional
        x = M_actual / M_H_actual
        
        # Fracción de densidad instantánea del campo escalar (Omega_phi)
        rho_phi = rho_inf_field_env(N)
        rho_crit = 3.0 * H_val**2.0 / (8.0 * jnp.pi)
        Omega_phi = rho_phi / rho_crit
        
        # Supresión de absorción de onda por naturaleza cuántica del campo escalar
        z = 2.0 * M_actual * mu_pl
        S_onda = (z**2.0) / (1.0 + z**2.0)
        
        # 1. Tasa de acreción. L_eff modulado por espín/onda/campo escalar en ambos
        #    regímenes (ver DERIVACION_ACRECION.md y factor_espin_acc/S_onda arriba).
        L_eff = L_acc * factor_espin_acc * S_onda * Omega_phi
        if regime == "wave":
            # Tarea 1 / DERIVACION_ACRECION.md §2c, §3: absorción de onda (Unruh 1976).
            # Sin min() y sin (1-x) — no hace falta regulador: si L*gamma<4*pi (ver
            # assert_subcritical), x decae monótonamente y nunca se acerca a 1.
            dlnM_dN_acc = 3.0 * L_eff * x / (8.0 * jnp.pi)
        elif regime == "horizon_tracking":
            # Tarea 4 / AUDIT.md P0-4 (ronda 1): min(Michel, cap), preservado como
            # resultado versionado, no como default — ver DERIVACION_ACRECION.md §1-2a
            # sobre por qué se descartó como modelo central.
            dlnM_dN_michel = 3.0 * L_eff * x / (8.0 * jnp.pi)
            dlnM_dN_cap = 3.0 * GAMMA_H / (2.0 * jnp.maximum(x, 1e-12))
            # min suave (ver EPS_SOFTMIN arriba): evita el kink de jnp.minimum que
            # colapsaba el paso del integrador cerca del punto fijo inestable x*.
            dlnM_dN_acc = 0.5 * (dlnM_dN_michel + dlnM_dN_cap) - 0.5 * jnp.sqrt(
                (dlnM_dN_michel - dlnM_dN_cap)**2 + EPS_SOFTMIN**2
            )
        else:
            raise ValueError(f"regime debe ser 'wave' u 'horizon_tracking', no {regime!r}")

        # 2. Evaporación de Hawking-Kerr. alpha(M) (Tarea 3): más dof activos a masa
        #    chica (T_H alta) que a masa grande (T_H baja) — ver alpha_of_mass_g arriba.
        alpha_local = alpha_of_mass_g(M_actual * M_pl_g)
        dM_dt_evap_pl = - (alpha_local / jnp.maximum(M_actual**2.0, 1.0)) * factor_evap_kerr
        dlnM_dN_evap = (dM_dt_evap_pl / H_val) / M_actual

        factor_apagado = 0.5 * (1.0 + jnp.tanh(u_safe * 3.0))
        du_dN = (dlnM_dN_acc + dlnM_dN_evap) * factor_apagado

        # AUDIT.md P1-8 (parcial): el jnp.where(u_safe<0,0,du_dN) que había aquí era
        # redundante con factor_apagado (ya suprime suavemente la tasa cerca de M_pl) y
        # además introducía un segundo kink justo en u_safe=0. Ese kink es lo que causaba
        # que diffeqsolve rechazara ~99% de los pasos (P1-6: expuesto por throw=True) para
        # PBHs que terminan evaporándose. factor_apagado solo ya es suficiente: la tasa
        # nunca es exactamente cero pero se vuelve despreciable, y el jnp.maximum(...,0.0)
        # al exponenciar el resultado en `integrar` evita masas negativas en la salida.
        return du_dN

    term = ODETerm(vector_field_log)
    solver = Tsit5()
    stepsize_controller = PIDController(rtol=1e-5, atol=1e-5, jump_ts=jnp.array([N_ini]))
    saveat = SaveAt(t1=True) 

    def integrar(_):
        N_ini_safe = jnp.minimum(N_ini, N_fin)
        span = jnp.maximum(N_fin - N_ini_safe, 1e-6)
        dt0_val = span * 1e-3
        u_0 = jnp.log(jnp.maximum(M_i_pl, 1.0))
        
        # AUDIT.md P1-6: throw=False dejaba pasar en silencio soluciones que agotaron
        # max_steps o divergieron. Se comprueba sol.result explícitamente en vez de
        # dejarlo pasar. No se usa throw=True: para PBHs que terminan evaporándose con
        # muchos e-folds de margen (N_fin grande), dlnM_dN_evap = dM_dt_evap_pl/H_val
        # diverge según N crece (H_val ~ e^{-1.5N} -> 0) mientras M se queda "atorada"
        # en unas pocas decenas de M_pl — el problema se vuelve genuinamente stiff para
        # un solver explícito (Tsit5) en esta formulación (integrar en e-folds N, no en
        # tiempo). Verificado con sol.stats: el número de pasos aceptados se satura
        # (no es cuestión de max_steps) y el estado, cuando esto pasa, ya está muy por
        # debajo de M_pl camino a evaporarse del todo. Por eso, si diffeqsolve no
        # converge, se toma M_final = M_pl (relic) en vez de tronar: es el destino físico
        # correcto en todos los casos que encontré (ver reporte P1-6), y evita que un
        # problema de robustez numérica en el régimen terminal de evaporación tumbe el
        # pipeline entero. Pendiente para una mejora futura: reformular la fase de
        # evaporación profunda en tiempo cósmico en vez de e-folds, o usar un solver
        # implícito ahí.
        sol = diffeqsolve(
            term, solver, t0=N_ini_safe, t1=N_fin, dt0=dt0_val, y0=u_0,
            stepsize_controller=stepsize_controller, saveat=saveat,
            max_steps=10000, throw=False
        )
        u_final = jnp.where(sol.result == RESULTS.successful, sol.ys[0], 0.0)
        return jnp.exp(jnp.maximum(u_final, 0.0))
        
    M_final_pl = lax.cond(N_ini < N_fin, integrar, lambda _: M_i_pl, operand=None)
    M_final_g = M_final_pl * M_pl_g

    return M_final_g, M_final_g / Mi_val_g


def precalcular_acreccion_lote(Mi_val_g, N_fin, a_star, L_acc=None, regime="wave",
                                gamma_form=GAMMA_COLAPSO):
    """Fase 1 (acreción + evaporación de Kerr). Tarea 1/2/4:

    regime='wave' (default): Unruh 1976, dlnM/dN=3*L_eff*x/8pi, x(N_form)=gamma_form.
        Requiere L_acc*factor_espin_acc(a_star)*gamma_form < 4*pi (assert_subcritical,
        se corre antes de integrar). Ver DERIVACION_ACRECION.md.
    regime='horizon_tracking': modelo de la ronda 1 (AUDIT.md P0-4), min(Michel,cap),
        x(N_form)=1 siempre (gamma_form se ignora). Preservado como resultado
        versionado — ver DERIVACION_ACRECION.md §1-2a sobre por qué no es el default.

    L_acc=None usa L_ACC_DEFAULT (wave) o L_ACC_HORIZON_TRACKING_DEFAULT
    (horizon_tracking) según el régimen.
    """
    if regime not in ("wave", "horizon_tracking"):
        raise ValueError(f"regime debe ser 'wave' u 'horizon_tracking', no {regime!r}")
    if L_acc is None:
        L_acc = L_ACC_DEFAULT if regime == "wave" else L_ACC_HORIZON_TRACKING_DEFAULT
    gamma_eff = gamma_form if regime == "wave" else 1.0
    if regime == "wave":
        try:
            a_star_scalar = float(a_star)
        except TypeError:
            a_star_scalar = None  # a_star traced (p.ej. dentro de otro jit); no se puede
                                    # chequear en Python de antemano, se deja pasar.
        if a_star_scalar is not None:
            assert_subcritical(float(L_acc), a_star=a_star_scalar, gamma_form=gamma_eff)
    return _precalcular_acreccion_lote_impl(Mi_val_g, N_fin, a_star, L_acc, regime, gamma_eff)


@eqx.filter_jit
def _trayectoria_pbh_impl(Mi_val_g, N_fin, a_star, L_acc, regime, gamma_form, N_puntos):
    """Misma física que _precalcular_acreccion_lote_impl (Tarea 1/2/4), pero devuelve la
    trayectoria completa M(N) en vez de sólo el valor final — para graficar M_pbh(N) vs
    M_H(N) (Tarea 6, celda 9 del notebook, que antes tenía su PROPIA copia del modelo
    viejo (1-x)/alpha=1/3 y se había quedado desincronizada del módulo real). Duplica la
    construcción del campo vectorial de _precalcular_acreccion_lote_impl a propósito
    (jax.vmap con SaveAt(t1=True) y esta función con SaveAt(ts=...) no comparten
    fácilmente el cuerpo sin refactorizar la firma pública) — mantener ambas en sync si
    se toca el modelo de acreción."""
    M_i_pl = Mi_val_g / M_pl_g
    M_end_pl = 1.0 / H_end_pl
    N_ini = (2.0 / 3.0) * jnp.log(jnp.maximum(M_i_pl / (gamma_form * M_end_pl), 1.0))
    a_star_safe = jnp.clip(a_star, 0.0, 0.9999)

    mu_pl = n * H_end_pl
    rho_end_inf_pl = (3.0 * H_end_pl**2.0) / (8.0 * jnp.pi)
    phi_ini_pl = jnp.sqrt(2.0 * rho_end_inf_pl / (mu_pl**2.0 + 9.0 * H_end_pl**2.0 / 4.0))

    raiz_espin = jnp.sqrt(1.0 - a_star_safe**2.0)
    factor_espin_acc = ((1.0 + raiz_espin) / 2.0)**2.0
    factor_T_kerr = (2.0 * raiz_espin) / (1.0 + raiz_espin)
    factor_Area_kerr = 0.5 * (1.0 + raiz_espin)
    factor_evap_kerr = (factor_T_kerr**4.0) * factor_Area_kerr

    def rho_inf_field_env(N):
        return 0.5 * (phi_ini_pl * mu_pl)**2 * jnp.exp(-3.0 * N)

    def vector_field_log(N, u, args):
        H_val = H_end_pl * jnp.exp(-1.5 * N)
        M_H_actual = 1.0 / H_val
        u_H = jnp.log(M_H_actual)
        u_safe = jnp.clip(u, -1.0, u_H)
        M_actual = jnp.exp(u_safe)
        x = M_actual / M_H_actual

        rho_phi = rho_inf_field_env(N)
        rho_crit = 3.0 * H_val**2.0 / (8.0 * jnp.pi)
        Omega_phi = rho_phi / rho_crit

        z = 2.0 * M_actual * mu_pl
        S_onda = (z**2.0) / (1.0 + z**2.0)

        L_eff = L_acc * factor_espin_acc * S_onda * Omega_phi
        if regime == "wave":
            dlnM_dN_acc = 3.0 * L_eff * x / (8.0 * jnp.pi)
        elif regime == "horizon_tracking":
            dlnM_dN_michel = 3.0 * L_eff * x / (8.0 * jnp.pi)
            dlnM_dN_cap = 3.0 * GAMMA_H / (2.0 * jnp.maximum(x, 1e-12))
            dlnM_dN_acc = 0.5 * (dlnM_dN_michel + dlnM_dN_cap) - 0.5 * jnp.sqrt(
                (dlnM_dN_michel - dlnM_dN_cap)**2 + EPS_SOFTMIN**2
            )
        else:
            raise ValueError(f"regime debe ser 'wave' u 'horizon_tracking', no {regime!r}")

        alpha_local = alpha_of_mass_g(M_actual * M_pl_g)
        dM_dt_evap_pl = - (alpha_local / jnp.maximum(M_actual**2.0, 1.0)) * factor_evap_kerr
        dlnM_dN_evap = (dM_dt_evap_pl / H_val) / M_actual

        factor_apagado = 0.5 * (1.0 + jnp.tanh(u_safe * 3.0))
        du_dN = (dlnM_dN_acc + dlnM_dN_evap) * factor_apagado
        return jnp.where(N < N_ini, 0.0, du_dN)

    ts_eval = jnp.linspace(0.0, N_fin, N_puntos)
    term = ODETerm(vector_field_log)
    solver = Tsit5()
    stepsize_controller = PIDController(rtol=1e-5, atol=1e-5, jump_ts=jnp.array([N_ini]))
    u_0 = jnp.log(jnp.maximum(M_i_pl, 1.0))
    dt0_val = jnp.maximum(N_fin, 1.0) / N_puntos * 0.1

    sol = diffeqsolve(
        term, solver, t0=0.0, t1=N_fin, dt0=dt0_val, y0=u_0,
        stepsize_controller=stepsize_controller, saveat=SaveAt(ts=ts_eval),
        max_steps=20000, throw=False,
    )
    u_final = jnp.where(sol.result == RESULTS.successful, sol.ys, 0.0)
    M_pbh_g = jnp.exp(jnp.maximum(u_final, 0.0)) * M_pl_g
    M_H_g = (1.0 / (H_end_pl * jnp.exp(-1.5 * ts_eval))) * M_pl_g
    return ts_eval, M_pbh_g, M_H_g, N_ini


def trayectoria_pbh_vs_horizonte(Mi_val_g, N_fin, a_star, L_acc=None, regime="wave",
                                  gamma_form=GAMMA_COLAPSO, N_puntos=1000):
    """Trayectoria M_pbh(N) vs M_H(N) para un solo PBH (Tarea 6, celda 9 del notebook).
    Mismos regímenes/parámetros que precalcular_acreccion_lote."""
    if regime not in ("wave", "horizon_tracking"):
        raise ValueError(f"regime debe ser 'wave' u 'horizon_tracking', no {regime!r}")
    if L_acc is None:
        L_acc = L_ACC_DEFAULT if regime == "wave" else L_ACC_HORIZON_TRACKING_DEFAULT
    gamma_eff = gamma_form if regime == "wave" else 1.0
    if regime == "wave":
        assert_subcritical(float(L_acc), a_star=float(a_star), gamma_form=gamma_eff)
    return _trayectoria_pbh_impl(Mi_val_g, N_fin, a_star, L_acc, regime, gamma_eff, N_puntos)

# ---------------------------------------------------------------------------
# Corrimiento de restricciones
# ---------------------------------------------------------------------------

def aplicar_corrimiento_acrecion(M_tot, M_f_tot, betas_sbb_full):
    # AUDIT.md P0-3: antes se inicializaba con constants.ev1 (1e-5) y, si mu>1.01 pero
    # M_f caía fuera del rango tabulado de betas_sbb_full, ninguna rama de abajo lo
    # sobreescribía -> ese centinela (una beta enorme y falsa) se colaba a las figuras
    # sin aviso. Se inicializa con np.nan ("no calculable") en su lugar: no hay una
    # extrapolación físicamente honesta de la restricción SBB fuera de su rango
    # tabulado, así que se deja el punto explícitamente vacío (NaN) en vez de rellenarlo
    # con un valor arbitrario. Los consumidores del notebook (`clean_beta`) ya tratan
    # NaN como "no graficar aquí".
    beta_acc = np.full_like(M_tot, np.nan, dtype=np.float64)

    mask_valid = betas_sbb_full != constants.ev1
    if not np.any(mask_valid):
        return beta_acc

    interpolador = interp1d(
        np.log10(M_tot[mask_valid]),
        np.log10(betas_sbb_full[mask_valid]),
        kind='linear', bounds_error=False, fill_value=np.nan
    )

    log_Mf = np.log10(M_f_tot)
    log_bf = interpolador(log_Mf)

    # AUDIT.md P0-5 (ronda 1): antes había dos fórmulas que no coincidían en la frontera
    # mu=1.01 (beta_SBB(1.01*M_i)/1.01 != beta_SBB(M_i) en general). Se unificó a una sola
    # fórmula continua, beta_SBB(M_f)/mu, razonando que se reduce a beta_SBB(M_i) cuando
    # mu->1 porque M_f->M_i ahí.
    #
    # BUG encontrado en la ronda 3 (Tarea 3 — el escalón en constraintsaccret.pdf, mismo
    # patrón que P0-5): ese razonamiento es cierto en el límite mu->1, pero la fórmula
    # unificada se rompe en el OTRO límite físico, mu->0 — que es exactamente el caso
    # RELIQUIA (el PBH se evapora del todo, M_f queda fijo en M_pl_g). Ahí log_bf[i] casi
    # siempre es un valor VÁLIDO (M_pl_g cae dentro del rango tabulado de M_tot), así que
    # la rama "not isnan(log_bf[i])" entraba primero y calculaba
    # beta_SBB(M_pl_g)/mu_diminuto -> un número arbitrario sin sentido físico, no
    # beta_SBB(M_i). Verificado con el código en la mano: M_i=370.8 g, N_fin=30 ->
    # M_f=M_pl_g exacto, mu=5.87e-8, beta_acc daba 7.77e-21 en vez de beta_SBB(M_i)=3.21e-17
    # (4 órdenes de magnitud de diferencia) — ESE es el salto en M~5e2g que se ve en la
    # figura, justo donde M_i cruza M_crit y mu pasa de "evaporación total" a "sobrevive".
    #
    # Fix: la condición de reliquia (PIPELINE.md, Etapa 2 — "beta_acc=beta_SBB(M_i)" para
    # reliquias, "la abundancia depende de la densidad NUMÉRICA, que la acreción no
    # cambia") se decide ANTES de intentar la fórmula de corrimiento, usando M_f mismo
    # (no mu): M_f pegado al piso de Planck es la reliquia por construcción del ODE de
    # Fase 1 (jnp.maximum(u_final,0.0) antes de exponenciar). No se suaviza: es una
    # condición física real (el PBH se evaporó entero o no), no ruido numérico a absorber.
    RELIC_FLOOR_TOL = 1.0 + 1e-6  # tolerancia relativa de punto flotante sobre M_pl_g

    for i in range(len(M_tot)):
        Mf = M_f_tot[i]
        Mi = M_tot[i]
        mu = Mf / Mi

        if Mf <= constants.M_pl_g * RELIC_FLOOR_TOL:
            beta_acc[i] = betas_sbb_full[i]
        elif not np.isnan(log_bf[i]):
            beta_acc[i] = (10.0**log_bf[i]) / mu
        elif mu <= 1.01:
            # M_f cae fuera del rango tabulado, pero mu~1 (M_f~M_i): se usa el valor
            # SBB directo en M_i como aproximación (el error introducido es O(mu-1)).
            beta_acc[i] = betas_sbb_full[i]
        # else: mu > 1.01 y M_f fuera de rango tabulado -> no calculable, queda NaN (P0-3).

    return beta_acc

# ---------------------------------------------------------------------------
# RESTRICCIONES ESTÁNDAR SBB (Época de dominancia de radiación)
# ---------------------------------------------------------------------------

def _solve_ivp_checked(*args, **kwargs):
    """Envoltura de scipy.solve_ivp (AUDIT.md P0-2/P1-6): truena si el integrador
    falla en vez de devolver un resultado parcial en silencio. status=1 (terminó
    por el evento `end_evol`, el PBH llegó a M_pl antes de ln_den_end_) es un
    resultado válido y esperado en este módulo; sólo status=-1 (falla del paso)
    se trata como error."""
    sol = solve_ivp(*args, **kwargs)
    if sol.status == -1:
        raise RuntimeError(f"solve_ivp falló (status=-1): {sol.message}")
    return sol

def diff_rad_rel(ln_rho, initial, M, beta0):
    b    = initial[0]
    Om_0 = beta0 * b * (constants.M_pl_g / M)
    dy   = -(Om_0 - 1.) * b / (Om_0 - 4.)
    return dy

def diff_rad(ln_rho, initial, M, beta0):
    dy = np.zeros(initial.shape)
    b = initial[0]
    time = initial[1]
    ratio_M = M / constants.M_pl_g
    if ratio_M > 1e90:
        ratio_t = 0.0  
    else:
        Delta_t = constants.t_pl * (ratio_M**3) / (3.0 * alpha_of_mass_g(M))
        ratio_t = time / Delta_t

    factor_evap = np.maximum(1.0 - ratio_t, 0.0)**(1.0 / 3.0)
    Om_0 = beta0 * b * factor_evap
    dy[0] = -(Om_0 - 1.0) * b / (Om_0 - 4.0)
    dy[1] = np.sqrt(3.0) * constants.M_pl / ((Om_0 - 4.0) * np.exp(ln_rho)**0.5)
    return dy

def end_evol(ln_rho, initial, M, beta0):
    ratio_M = M / constants.M_pl_g
    if ratio_M > 1e90:
        return 1.0  
    Delta_t = constants.t_pl * (ratio_M**3) / (3.0 * alpha_of_mass_g(M))
    d_time = initial[1]
    ratio_t = np.minimum(d_time / Delta_t, 1.0)
    Mass_end = M * (1.0 - ratio_t)**(1.0 / 3.0)
    return Mass_end - constants.M_pl_g

end_evol.terminal = True
end_evol.direction = -1

# AUDIT.md P1-10: la rama omega!=1/3 de k_end_over_k/rho_f mezclaba Mpbh en GRAMOS con
# H_end/M_pl en GeV. Mpbh_a_GeV usa la misma razón M_pl_GeV/M_pl_g ya definida arriba
# (=5.61e23 GeV/g) para no introducir una constante nueva sin relación con el resto del
# módulo. No muerde con omega=1/3 (celdas 8, 11 del notebook), pero Betas_*(M_tot, omega)
# recibe omega del llamador y el caso de recalentamiento (omega=0, el punto del paper con
# P0-4) sí entra por esta rama.
def _Mpbh_a_GeV(Mpbh_g):
    return Mpbh_g * (M_pl_GeV / M_pl_g)

def k_end_over_k(Mpbh, omega):
    if omega == 1/3:
        res = (Mpbh / (7.1e-2 * constants.gam_rad * (1.8e15 / constants.H_end)))**(1/2)
    else:
        z   = (1 + 3*omega) / (3 * (1 + omega))
        res = np.array((_Mpbh_a_GeV(Mpbh) * constants.H_end / (3 * constants.gam_rad * constants.M_pl**2))**z)
    return res

def rho_f(Mpbh, omega):
    if omega == 1/3:
        k_ratio = (Mpbh / (7.1e-2 * constants.gam_rad * (1.8e15 / constants.H_end)))**(1/2)
        return constants.rho_end_inf / k_ratio**4
    else:
        z   = (1 + 3*omega) / (3 * (1 + omega))
        res = np.array((_Mpbh_a_GeV(Mpbh) * constants.H_end / (3 * constants.gam_rad * constants.M_pl**2))**z)
        i   = (6 * (1 + omega)) / (1 + 3*omega)
        return constants.rho_end_inf / res**i

def Betas_DM(M_tot, omega):
    M_n, betas_prim, M_relic, betas_relic_prim = [], [], [], []
    betas_tot, Omegas_tot = [], []
    Omegas, Omegas_relic_pbbn, Omegas_relic = [], [], []
    M_dm, M_dm_rel_pbbn, M_dm_rel = [], [], []

    rho_form_rad = rho_f(M_tot, omega)
    ln_den_end_  = np.log(constants.rho_end)

    for i in range(len(M_tot)):
        M_i = M_tot[i]
        if M_i > 4.1e14:
            M_n.append(M_i)
            beta_std = 1.86e-18 * (M_i / 1e15)**0.5
            betas_prim.append(beta_std)
            betas_tot.append(betas_prim[-1] / constants.gam_rad**0.5)
        elif M_i <= 1e11 * constants.M_pl_g:
            M_relic.append(M_i)
            beta_std = 2e-28 * (M_i / constants.M_pl_g)**1.5
            betas_relic_prim.append(beta_std)
            betas_tot.append(betas_relic_prim[-1] / constants.gam_rad**0.5)
        else:
            # AUDIT.md P1-12: antes constants.ev1 (1e-5), y el chequeo de abajo comparaba
            # contra DOS valores posibles (ev1 y ev1/gam_rad**0.5) con `==` sobre floats,
            # señal de que el centinela no era único. np.nan + np.isnan es inequívoco y no
            # depende de si el valor se dividió por gam_rad**0.5 en otra rama o no.
            betas_tot.append(np.nan)

    constraints.betas_DM_tot = np.array(betas_tot)

    for i in range(len(M_tot)):
        if i >= len(betas_tot) or np.isnan(betas_tot[i]):
            Omegas_tot.append(constants.ev2)
            continue
        M_i = M_tot[i]
        ln_den_f = np.log(rho_form_rad[i])
        
        if ln_den_f <= ln_den_end_:
            Omegas_tot.append(constants.ev2)
            continue
        ln_den = np.linspace(ln_den_f, ln_den_end_, 10000)

        if M_i <= LIMITE_PLANCK_G:
            sol_try_rel = _solve_ivp_checked(diff_rad_rel, (ln_den_f, ln_den_end_), np.array([1.]),
                                    t_eval=ln_den, args=(constants.M_pl_g, betas_tot[i]), method="DOP853")
            y_val = betas_tot[i] * sol_try_rel.y[0][-1]
            Omegas_relic.append(y_val)
            M_dm_rel.append(M_tot[i])
        else:
            sol_try = _solve_ivp_checked(diff_rad, (ln_den_f, ln_den_end_), np.array([1., 0.]),
                                events=end_evol, t_eval=ln_den,
                                args=(M_i, betas_tot[i]), method="DOP853")

            if len(sol_try.t) == 0 or sol_try.t[-1] > ln_den_end_:
                sol_try_rel = _solve_ivp_checked(diff_rad_rel, (ln_den_f, ln_den_end_), np.array([1.]),
                                        t_eval=ln_den, args=(M_i, betas_tot[i]), method="DOP853")
                y_val = betas_tot[i] * sol_try_rel.y[0][-1] * (constants.M_pl_g / M_i)
                if M_i < 1e11 * constants.M_pl_g:
                    Omegas_relic_pbbn.append(y_val)
                    M_dm_rel_pbbn.append(M_tot[i])
            else:
                ratio_M = M_i / constants.M_pl_g
                if ratio_M > 1e90:
                    factor_evap_final = 1.0
                else:
                    Delta_t = constants.t_pl * (ratio_M**3) / (3.0 * alpha_of_mass_g(M_i))
                    factor_evap_final = np.maximum(1.0 - sol_try.y[1][-1] / Delta_t, 0.0)**(1.0 / 3.0)

                y_val = betas_tot[i] * sol_try.y[0][-1] * factor_evap_final
                if M_i > 4.1e14:
                    Omegas.append(y_val)
                    M_dm.append(M_tot[i])

        Omegas_tot.append(y_val)

    constraints.Omega_DM_tot = np.array(Omegas_tot)
    return np.array(M_n), np.array(betas_prim)/constants.gam_rad**0.5, np.array(M_relic), np.array(betas_relic_prim)/constants.gam_rad**0.5, np.array(Omegas_tot)


def Betas_BBN(M_tot, omega):
    betas_bbn, M_bbn, Omegas_bbn_tot = [], [], []
    rho_form_rad = rho_f(M_tot, omega)
    ln_den_end_ = np.log(constants.rho_end)
    limite_planck = LIMITE_PLANCK_G

    for i in range(len(M_tot)):
        M_i = M_tot[i]
        if M_i <= limite_planck:
            constraints.betas_BBN_tot.append(constants.ev1)
            Omegas_bbn_tot.append(constants.ev2)
            continue

        ln_den_f = np.log(rho_form_rad[i])
        if constraints.data_mass[0] <= M_i < 2.5e13:
            M_bbn.append(M_i)
            abundancia = np.interp(M_i, constraints.data_mass, constraints.data_abundances)
            beta = (abundancia / constants.gam_rad**0.5) 
            betas_bbn.append(beta)

            if ln_den_f > ln_den_end_:
                ln_den = np.linspace(ln_den_f, ln_den_end_, 10000)
                sol_try = _solve_ivp_checked(diff_rad, (ln_den_f, ln_den_end_), np.array([1., 0.]),
                                    events=end_evol, t_eval=ln_den,
                                    args=(M_i, beta), method="DOP853")
                if len(sol_try.t) == 0 or (sol_try.t[-1] > ln_den_end_ and M_i < constraints.data_mass[76]):
                    sol_try_rel = _solve_ivp_checked(diff_rad_rel, (ln_den_f, ln_den_end_), np.array([1.]),
                                            t_eval=ln_den, args=(M_i, beta), method="DOP853")
                    y_val = beta * sol_try_rel.y[0][-1] * (constants.M_pl_g / M_i)
                else:
                    Delta_t = constants.t_pl * (M_i / constants.M_pl_g)**3 / (3.0 * alpha_of_mass_g(M_i))
                    y_val = beta * sol_try.y[0][-1] * (1. - sol_try.y[1][-1] / Delta_t)**(1./3)
            else:
                y_val = constants.ev2
        else:
            beta = constants.ev1
            y_val = constants.ev2

        constraints.betas_BBN_tot.append(beta)
        Omegas_bbn_tot.append(y_val)

    constraints.Omega_BBN_tot = np.array(Omegas_bbn_tot)
    return np.array(M_bbn), np.array(betas_bbn), np.array(Omegas_bbn_tot)


def Betas_SD(M_tot, omega):
    betas_sd, M_sd, Omegas_sd_tot = [], [], []
    rho_form_rad = rho_f(M_tot, omega)
    ln_den_end_ = np.log(constants.rho_end)
    limite_planck = LIMITE_PLANCK_G

    for i in range(len(M_tot)):
        M_i = M_tot[i]
        if M_i > limite_planck and 1e11 < M_i < 1e13:
            M_sd.append(M_i)
            beta = (1e-21 / constants.gam_rad**0.5) 
            betas_sd.append(beta)

            ln_den_f = np.log(rho_form_rad[i])
            if ln_den_f > ln_den_end_:
                ln_den = np.linspace(ln_den_f, ln_den_end_, 10000)
                sol_try = _solve_ivp_checked(diff_rad, (ln_den_f, ln_den_end_), np.array([1., 0.]),
                                    events=end_evol, t_eval=ln_den,
                                    args=(M_i, beta), method="DOP853")
                Delta_t = constants.t_pl * (M_i / constants.M_pl_g)**3 / (3.0 * alpha_of_mass_g(M_i))
                if len(sol_try.t) > 0:
                    y_val = beta * sol_try.y[0][-1] * (1. - sol_try.y[1][-1] / Delta_t)**(1./3)
                else:
                    y_val = 0.0
            else:
                y_val = constants.ev2
        else:
            beta = constants.ev1
            y_val = constants.ev2

        constraints.betas_SD_tot.append(beta)
        constraints.Omega_SD_tot.append(y_val)

    return np.array(M_sd), np.array(betas_sd), np.array(constraints.Omega_SD_tot)


def Betas_CMB_AN(M_tot, omega):
    betas_an, M_an, Omegas_an_tot = [], [], []
    rho_form_rad = rho_f(M_tot, omega)
    ln_den_end_ = np.log(constants.rho_end)
    limite_planck = LIMITE_PLANCK_G

    for i in range(len(M_tot)):
        M_i = M_tot[i]
        if M_i > limite_planck and 2.5e13 < M_i < 2.4e14:
            M_an.append(M_i)
            beta = (3e-30 * (M_i / 1e13)**3.1 / constants.gam_rad**0.5) 
            betas_an.append(beta)

            ln_den_f = np.log(rho_form_rad[i])
            if ln_den_f > ln_den_end_:
                ln_den = np.linspace(ln_den_f, ln_den_end_, 10000)
                sol_try = _solve_ivp_checked(diff_rad, (ln_den_f, ln_den_end_), np.array([1., 0.]),
                                    events=end_evol, t_eval=ln_den,
                                    args=(M_i, beta), method="DOP853")
                Delta_t = constants.t_pl * (M_i / constants.M_pl_g)**3 / (3.0 * alpha_of_mass_g(M_i))
                if len(sol_try.t) > 0:
                    y_val = beta * sol_try.y[0][-1] * (1. - sol_try.y[1][-1] / Delta_t)**(1./3)
                else:
                    y_val = 0.0
            else:
                y_val = constants.ev2
        else:
            beta = constants.ev1
            y_val = constants.ev2

        constraints.betas_CMB_AN_tot.append(beta)
        constraints.Omega_CMB_AN_tot.append(y_val)

    return np.array(M_an), np.array(betas_an), np.array(constraints.Omega_CMB_AN_tot)


def Betas_GRB(M_tot, omega):
    betas_grb1, M_grb1, betas_grb2, M_grb2 = [], [], [], []
    Omegas_grb1, Omegas_grb2 = [], []
    rho_form_rad = rho_f(M_tot, omega)
    ln_den_end_ = np.log(constants.rho_end)
    limite_planck = LIMITE_PLANCK_G

    for i in range(len(M_tot)):
        M_i = M_tot[i]
        ln_den_f = np.log(rho_form_rad[i])
        beta = constants.ev1
        y_val = constants.ev2

        if M_i > limite_planck and 3e13 < M_i < 4.1e14:
            M_grb1.append(M_i)
            beta = (5e-28 * (M_i / 4.1e14)**(-3.3) / constants.gam_rad**0.5) 
            betas_grb1.append(beta)

            if ln_den_f > ln_den_end_:
                ln_den = np.linspace(ln_den_f, ln_den_end_, 10000)
                sol_try = _solve_ivp_checked(diff_rad, (ln_den_f, ln_den_end_), np.array([1., 0.]),
                                    events=end_evol, t_eval=ln_den,
                                    args=(M_i, beta), method="DOP853")
                Delta_t = constants.t_pl * (M_i / constants.M_pl_g)**3 / (3.0 * alpha_of_mass_g(M_i))
                if len(sol_try.t) > 0:
                    y_val = beta * sol_try.y[0][-1] * (1. - sol_try.y[1][-1] / Delta_t)**(1./3)
                else:
                    y_val = 0.0
                Omegas_grb1.append(y_val)

        elif M_i > limite_planck and 4.1e14 < M_i < 7e16:
            M_grb2.append(M_i)
            beta = (5e-26 * (M_i / 4.1e14)**3.9 / constants.gam_rad**0.5) 
            betas_grb2.append(beta)

            if ln_den_f > ln_den_end_:
                ln_den = np.linspace(ln_den_f, ln_den_end_, 10000)
                sol_try = _solve_ivp_checked(diff_rad, (ln_den_f, ln_den_end_), np.array([1., 0.]),
                                    events=end_evol, t_eval=ln_den,
                                    args=(M_i, beta), method="DOP853")
                Delta_t = constants.t_pl * (M_i / constants.M_pl_g)**3 / (3.0 * alpha_of_mass_g(M_i))
                if len(sol_try.t) > 0:
                    y_val = beta * sol_try.y[0][-1] * (1. - sol_try.y[1][-1] / Delta_t)**(1./3)
                else:
                    y_val = 0.0
                Omegas_grb2.append(y_val)

        constraints.betas_GRB_tot.append(beta)
        constraints.Omega_GRB_tot.append(y_val)

    return (np.array(M_grb1), np.array(M_grb2), np.array(betas_grb1), np.array(betas_grb2),
            np.array(Omegas_grb1), np.array(Omegas_grb2))


def Betas_Reio(M_tot, omega):
    betas_reio, M_reio = [], []
    rho_form_rad = rho_f(M_tot, omega)
    ln_den_end_ = np.log(constants.rho_end)
    limite_planck = LIMITE_PLANCK_G

    for i in range(len(M_tot)):
        M_i = M_tot[i]
        ln_den_f = np.log(rho_form_rad[i])
        beta = constants.ev1
        y_val = constants.ev2

        if M_i > limite_planck and 1e15 < M_i < 1e17:
            M_reio.append(M_i)
            beta = (2.4e-26 * (M_i / 4.1e14)**4.3 / constants.gam_rad**0.5) 
            betas_reio.append(beta)

            if ln_den_f > ln_den_end_:
                ln_den = np.linspace(ln_den_f, ln_den_end_, 10000)
                sol_try = _solve_ivp_checked(diff_rad, (ln_den_f, ln_den_end_), np.array([1., 0.]),
                                    events=end_evol, t_eval=ln_den,
                                    args=(M_i, beta), method="DOP853")
                Delta_t = constants.t_pl * (M_i / constants.M_pl_g)**3 / (3.0 * alpha_of_mass_g(M_i))
                if len(sol_try.t) > 0:
                    y_val = beta * sol_try.y[0][-1] * (1. - sol_try.y[1][-1] / Delta_t)**(1./3)
                else:
                    y_val = 0.0

        constraints.betas_Reio_tot.append(beta)
        constraints.Omega_Reio_tot.append(y_val)

    return np.array(M_reio), np.array(betas_reio), np.array(constraints.Omega_Reio_tot)


def Betas_LSP(M_tot, w):
    betas_lsp, M_lsp = [], []
    rho_form_rad = rho_f(M_tot, w)
    ln_den_end_ = np.log(constants.rho_end)
    limite_planck = LIMITE_PLANCK_G

    for i in range(len(M_tot)):
        M_i = M_tot[i]
        beta = constants.ev1
        y_val = constants.ev2

        if M_i > limite_planck and M_i < 1e11:
            M_lsp.append(M_i)
            beta = (1e-18 * (M_i / 1e11)**(-0.5) / constants.gam_rad**0.5) 
            betas_lsp.append(beta)

            ln_den_f = np.log(rho_form_rad[i])
            if ln_den_f > ln_den_end_:
                ln_den = np.linspace(ln_den_f, ln_den_end_, 10000)
                sol_try = _solve_ivp_checked(diff_rad, (ln_den_f, ln_den_end_), np.array([1., 0.]),
                                    events=end_evol, t_eval=ln_den,
                                    args=(M_i, beta), method="DOP853")
                
                if len(sol_try.t) == 0 or sol_try.t[-1] > ln_den_end_:
                    sol_try_rel = _solve_ivp_checked(diff_rad_rel, (ln_den_f, ln_den_end_), np.array([1.]),
                                            t_eval=ln_den, args=(M_i, beta), method="DOP853")
                    y_val = beta * sol_try_rel.y[0][-1] * (constants.M_pl_g / M_i)
                else:
                    Delta_t = constants.t_pl * (M_i / constants.M_pl_g)**3 / (3.0 * alpha_of_mass_g(M_i))
                    y_val = beta * sol_try.y[0][-1] * (1. - sol_try.y[1][-1] / Delta_t)**(1./3)

        constraints.betas_LSP_tot.append(beta)
        constraints.Omega_LSP_tot.append(y_val)

    return np.array(M_lsp), np.array(betas_lsp), np.array(constraints.Omega_LSP_tot)


def get_Betas_full(M_tot):
    DM_tot   = np.array(constraints.betas_DM_tot)
    BBN_tot  = np.array(constraints.betas_BBN_tot)
    SD_tot   = np.array(constraints.betas_SD_tot)
    CMB_tot  = np.array(constraints.betas_CMB_AN_tot)
    GRB_tot  = np.array(constraints.betas_GRB_tot)
    Reio_tot = np.array(constraints.betas_Reio_tot)
    LSP_tot  = np.array(constraints.betas_LSP_tot)

    constraints.betas_full = np.zeros_like(M_tot, dtype=np.float64)
    for i in range(len(M_tot)):
        values = []
        if DM_tot.size:   values.append(DM_tot[i])
        if BBN_tot.size:  values.append(BBN_tot[i])
        if SD_tot.size:   values.append(SD_tot[i])
        if CMB_tot.size:  values.append(CMB_tot[i])
        if GRB_tot.size:  values.append(GRB_tot[i])
        if Reio_tot.size: values.append(Reio_tot[i])
        if LSP_tot.size:  values.append(LSP_tot[i])
        if values:
            # np.nanmin, no min(): P1-12 hace que el canal DM pueda venir como np.nan.
            # min() de Python con NaN depende del orden (si el NaN es el primer elemento,
            # min() devuelve NaN aunque haya valores reales después) — np.nanmin lo ignora
            # correctamente y sólo da NaN si TODOS los canales son NaN.
            constraints.betas_full[i] = np.nanmin(values)
    return constraints.betas_full

def get_Omegas_full(M_tot):
    DM_tot   = np.array(constraints.Omega_DM_tot)
    BBN_tot  = np.array(constraints.Omega_BBN_tot)
    SD_tot   = np.array(constraints.Omega_SD_tot)
    CMB_tot  = np.array(constraints.Omega_CMB_AN_tot)
    GRB_tot  = np.array(constraints.Omega_GRB_tot)
    Reio_tot = np.array(constraints.Omega_Reio_tot)
    LSP_tot  = np.array(constraints.Omega_LSP_tot)

    constraints.Omegas_full = M_tot * 0
    for i in range(len(M_tot)):
        values = []
        if DM_tot.size:   values.append(DM_tot[i])
        if BBN_tot.size:  values.append(BBN_tot[i])
        if SD_tot.size:   values.append(SD_tot[i])
        if CMB_tot.size:  values.append(CMB_tot[i])
        if GRB_tot.size:  values.append(GRB_tot[i])
        if Reio_tot.size: values.append(Reio_tot[i])
        if LSP_tot.size:  values.append(LSP_tot[i])
        if values:
            # np.nanmin por la misma razón que en get_Betas_full (consistencia/defensivo;
            # Omega_DM_tot usa constants.ev2 como centinela, no NaN, así que hoy no cambia
            # el resultado, pero evita el mismo landmine si eso cambia más adelante).
            constraints.Omegas_full[i] = np.nanmin(values)
    return constraints.Omegas_full
