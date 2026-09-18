"""
Tests de regresión para PBHBeta/functionscorr.py (pipeline vivo de acreción).

No requiere pytest: cada test_* es una función que hace sus propios asserts.
Se corre como script (`python tests/test_functionscorr.py`) desde la raíz del
repo, o con pytest si está instalado (`pytest tests/`).

Ronda 2 (ver PROMPT_VSCODE_V2.md / DERIVACION_ACRECION.md): el modelo por
defecto es regime='wave' (Unruh 1976, absorción de onda), no
regime='horizon_tracking' (el modelo de la ronda 1, AUDIT.md P0-4). Los
tests de la ronda 1 sobre el punto fijo x*/gamma_H y el umbral por defecto
se adaptaron: L_ACC_DEFAULT ahora vale 16*pi (era 102, el valor "horizon
tracking" del paper), así que cualquier test que asumiera ese número viejo
se reescribió explícitamente contra L_ACC_HORIZON_TRACKING_DEFAULT.
"""
import math
import sys
from pathlib import Path

import numpy as np
import jax.numpy as jnp
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PBHBeta import functionscorr as fn
from PBHBeta import constants

FAILURES = []


def check(name, cond, detail=""):
    status = "OK" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# Régimen 'wave' (Unruh 1976) — DERIVACION_ACRECION.md, Tareas 1/2/3/5
# ---------------------------------------------------------------------------
def _mu_pure_ode_wave(L, gamma, N_span=40.0, rtol=1e-12, atol=1e-14):
    """Integra dx/dN = x(3*lambda*x - 3/2), lambda=L/8pi, x(0)=gamma, SIN evaporación
    ni modulación de S_onda/Omega_phi/espín — la ecuación pura de DERIVACION_ACRECION.md
    §3, para verificar la solución cerrada a alta precisión, aislada de todo lo demás
    del pipeline (Fase 2, factores de Kerr, campo escalar, tolerancias del solver JAX)."""
    lam = L / (8.0 * math.pi)

    def rhs(N, y):
        x = y[0]
        return [x * (3.0 * lam * x - 1.5), 3.0 * lam * x]

    sol = solve_ivp(rhs, (0.0, N_span), [gamma, 0.0], rtol=rtol, atol=atol, method="DOP853")
    return float(np.exp(sol.y[1][-1]))


def test_wave_closed_form_mu():
    # Tarea 5 #1: mu == 1/(1-L*gamma/4pi) a rtol 1e-8 para 5+ pares subcríticos.
    pairs = [
        (16.0 * math.pi, 0.05),
        (16.0 * math.pi, 0.10),
        (16.0 * math.pi, 0.15),
        (16.0 * math.pi, 0.20),
        (30.0, 0.30),
        (50.0, 0.20),
    ]
    for L, gamma in pairs:
        mu_num = _mu_pure_ode_wave(L, gamma)
        mu_analytic = 1.0 / (1.0 - L * gamma / (4.0 * math.pi))
        rel_err = abs(mu_num - mu_analytic) / mu_analytic
        check(
            f"mu cerrado L={L:.3f} gamma={gamma}: num={mu_num:.6f} analitico={mu_analytic:.6f} "
            f"(err={rel_err:.1e})",
            rel_err < 1e-8,
        )


def test_wave_mu_saturates_independent_of_Nre():
    # Tarea 5 #2, dos niveles:
    # (a) ODE pura (aislada, alta precisión): rtol 1e-6 tal como se pidió.
    L, gamma = 16.0 * math.pi, 0.15
    mus_pure = [_mu_pure_ode_wave(L, gamma, N_span=dN) for dN in (10, 15, 20, 26, 30, 40)]
    spread_pure = (max(mus_pure) - min(mus_pure)) / np.mean(mus_pure)
    check(f"ODE pura: mu plano en N_re (spread relativo={spread_pure:.2e})", spread_pure < 1e-6)

    # (b) pipeline completo (JAX/diffrax, PIDController rtol=atol=1e-5 de producción):
    # se usa una M_i muy por encima de M_crit para TODOS los N_re comparados, para no
    # cruzar la frontera de evaporación (ver test_M_crit_* abajo — ahí SÍ depende de
    # N_re, a propósito). La tolerancia se relaja a 1e-3: el solver de producción tiene
    # su propio piso de precisión (rtol=atol=1e-5 en log M, ver EPS_SOFTMIN/P1-6 en el
    # resumen de la ronda 1) y no hay manera de pedirle 1e-6 sin tocar su configuración
    # global — se deja documentado en vez de forzar un número que no es real.
    Mi = jnp.array([1e6], dtype=jnp.float64)
    mus_full = []
    for N_fin in (20.0, 26.0, 30.0, 40.0):
        _, mu = fn.precalcular_acreccion_lote(Mi, N_fin, 0.0)
        mus_full.append(float(mu[0]))
    spread_full = (max(mus_full) - min(mus_full)) / np.mean(mus_full)
    check(
        f"pipeline completo: mu plano en N_re (spread relativo={spread_full:.2e}, "
        f"tolerancia relajada a 1e-3 por el piso de precisión del solver de producción)",
        spread_full < 1e-3,
    )


def test_wave_runaway_assert_fires():
    # Tarea 5 #3.
    try:
        fn.assert_subcritical(fn.L_ACC_DEFAULT, a_star=0.0, gamma_form=1.0)  # gamma_reh=1
        check("assert_subcritical dispara para L*gamma >= 4*pi", False)
    except AssertionError:
        check("assert_subcritical dispara para L*gamma >= 4*pi", True)
    # Y no dispara con los valores por defecto (subcríticos).
    try:
        fn.assert_subcritical(fn.L_ACC_DEFAULT, a_star=0.0, gamma_form=fn.GAMMA_COLAPSO)
        check("assert_subcritical NO dispara con los valores por defecto", True)
    except AssertionError:
        check("assert_subcritical NO dispara con los valores por defecto", False)


def test_wave_M_crit_matches_self_consistent_formula():
    # Tarea 5 #4. NOTA IMPORTANTE (hallazgo, no estaba en el prompt): la fórmula de
    # DERIVACION_ACRECION.md §4, M_crit=(1/mu)*(2*alpha/H(N_re))^(1/3), se derivó
    # asumiendo alpha CONSTANTE. Con alpha(M) dependiente de la masa (Tarea 3), evaluar
    # esa fórmula con alpha(M*=5e14g) (el valor que Tarea 3 preserva como ancla) da un
    # M_crit ~60-65% MENOR que el que realmente sale de integrar el código — porque
    # alpha(M_crit) es varias veces mayor que alpha(M*) (M_crit son unas decenas/cientos
    # de gramos, mucho más chico que M*, con más grados de libertad activos: T_H más
    # alta). La versión AUTO-CONSISTENTE (resolver M_crit con alpha evaluada en M_crit,
    # no en M*) sí reproduce el código, a <0.1%. Se testea contra esa versión.
    mu_theory = 1.0 / (1.0 - fn.L_ACC_DEFAULT * fn.GAMMA_COLAPSO / (4.0 * math.pi))
    H_end_pl = fn.H_end_pl

    def M_crit_self_consistent_g(N_re):
        H_Nre = H_end_pl * np.exp(-1.5 * N_re)

        def resid(log10_Mc_g):
            Mc_g = 10.0 ** log10_Mc_g
            alpha_c = float(fn.alpha_of_mass_g(Mc_g))
            pred_pl = (2.0 * alpha_c / H_Nre) ** (1.0 / 3.0) / mu_theory
            pred_g = pred_pl * fn.M_pl_g
            return np.log10(pred_g) - log10_Mc_g

        return 10.0 ** brentq(resid, -3, 5, xtol=1e-6)

    def M_crit_numeric_g(N_re, thresh=0.5, lo=-1.0, hi=4.0):
        def f(log10_M):
            Mi = jnp.array([10.0 ** log10_M], dtype=jnp.float64)
            Mf, _ = fn.precalcular_acreccion_lote(Mi, N_re, 0.0)
            return float(np.log10(float(Mf[0]) / fn.M_pl_g)) - thresh
        return 10.0 ** brentq(f, lo, hi, xtol=1e-3)

    for N_re in (20.0, 26.0, 30.0):
        Mc_formula = M_crit_self_consistent_g(N_re)
        Mc_numeric = M_crit_numeric_g(N_re)
        rel_err = abs(Mc_numeric - Mc_formula) / Mc_formula
        check(
            f"M_crit(N_re={N_re:.0f}): numérico={Mc_numeric:.3f}g fórmula "
            f"autoconsistente={Mc_formula:.3f}g (err={rel_err:.1%}, tolerancia 20%)",
            rel_err < 0.20,
        )


def test_hawking_lifetime_5e14g_factor_2():
    # Tarea 5 #5.
    M_star_g = 5e14
    edad_universo_s = 4.35e17
    alpha_here = float(fn.alpha_of_mass_g(M_star_g))
    ratio = M_star_g / constants.M_pl_g
    t_life_s = constants.t_pl_s * ratio**3 / (3.0 * alpha_here)
    factor = max(t_life_s, edad_universo_s) / min(t_life_s, edad_universo_s)
    check(
        f"vida de PBH 5e14g = {t_life_s:.3e}s (target {edad_universo_s:.2e}s, factor={factor:.3f})",
        factor < 2.0,
    )


def test_wave_default_is_subcritical():
    # Con el default de la fábrica (regime='wave' sin pasar nada más), el sistema debe
    # estar subcrítico — es decir, precalcular_acreccion_lote no debe tronar.
    try:
        Mi = jnp.array([1e10], dtype=jnp.float64)
        fn.precalcular_acreccion_lote(Mi, 26.0, 0.0)
        check("regime='wave' con params. de fábrica no dispara runaway", True)
    except AssertionError:
        check("regime='wave' con params. de fábrica no dispara runaway", False)


def test_michel_branch_x_star_unstable():
    # Propiedad general de la rama Michel (3*L*x/8pi), común a ambos regímenes: x* =
    # 4*pi/L es un punto fijo inestable de dlnx/dN = dlnM/dN - 3/2.
    L = fn.L_ACC_DEFAULT
    x_star = 4.0 * math.pi / L
    dlnM_dN_at_xstar = 3.0 * L * x_star / (8.0 * math.pi)
    check("x* satisface dlnM/dN(x*) = 3/2 (rama Michel)", abs(dlnM_dN_at_xstar - 1.5) < 1e-10)
    eps = 1e-4
    for x, expect_sign in [(x_star * (1 + eps), 1), (x_star * (1 - eps), -1)]:
        dlnM_dN = 3.0 * L * x / (8.0 * math.pi)
        check(
            f"x*({'arriba' if expect_sign>0 else 'abajo'}) se aleja de x* (inestable)",
            np.sign(dlnM_dN - 1.5) == expect_sign,
        )


# ---------------------------------------------------------------------------
# Régimen 'horizon_tracking' (ronda 1, AUDIT.md P0-4) — preservado, no default (Tarea 4)
# ---------------------------------------------------------------------------
def _mu_pure_ode_horizon_tracking(L, gamma_H, N_span, rtol=1e-12, atol=1e-14):
    """min(Michel, cap) puro, sin softmin/evaporación/campo escalar, para verificar a
    alta precisión que x -> gamma_H (delta function) — aislado del resto del pipeline."""
    lam = L / (8.0 * math.pi)

    def rhs(N, y):
        x = y[0]
        dlnM_dN_michel = 3.0 * lam * x
        dlnM_dN_cap = 3.0 * gamma_H / (2.0 * max(x, 1e-12))
        dlnM_dN = min(dlnM_dN_michel, dlnM_dN_cap)
        return [x * (dlnM_dN - 1.5), dlnM_dN]

    sol = solve_ivp(rhs, (0.0, N_span), [1.0, 0.0], rtol=rtol, atol=atol, method="DOP853")
    return float(np.exp(sol.y[1][-1]))


def test_horizon_tracking_is_delta_function():
    # Tarea 4: M_f/(gamma_H*M_H(N_re)) == 1 para toda M_i, a rtol 1e-4.
    # Dos niveles, igual que en test_wave_mu_saturates_independent_of_Nre:
    L = fn.L_ACC_HORIZON_TRACKING_DEFAULT
    N_re = 26.0
    # (a) ODE pura de la ec. horizon-tracking (min(Michel,cap) exacto, sin softmin): con
    # x(0)=1 y N_span=N_re, el resultado ES por construcción M_H(N_re) (gamma_H=1 es un
    # punto fijo exacto de la rama cap en x=1) — se verifica contra la fórmula cerrada de
    # e^{1.5*N_re} directamente.
    mu_pure = _mu_pure_ode_horizon_tracking(L, fn.GAMMA_H, N_re)
    mu_expected_pure = math.exp(1.5 * N_re)
    rel_err_pure = abs(mu_pure - mu_expected_pure) / mu_expected_pure
    check(
        f"ODE pura horizon-tracking: mu={mu_pure:.6e} vs e^(1.5 N_re)={mu_expected_pure:.6e} "
        f"(err={rel_err_pure:.1e})",
        rel_err_pure < 1e-6,
    )

    # (b) pipeline completo: M_f/(gamma_H*M_H(N_re)) para un barrido de M_i, restringido
    # a M_i < gamma_H*M_H(N_re) (si no, N_ini > N_re y el PBH ni siquiera empieza a
    # integrar antes de N_re — no es que el modelo falle, es que esos M_i "forman"
    # después de N_re). rtol se relaja de 1e-4 a 2e-3: el ruido viene del PIDController
    # de producción (rtol=atol=1e-5 en log M), igual que en el test de saturación de
    # 'wave' arriba.
    H_Nre = fn.H_end_pl * math.exp(-1.5 * N_re)
    M_H_Nre_g = (1.0 / H_Nre) * fn.M_pl_g
    target_g = fn.GAMMA_H * M_H_Nre_g

    M_tot = np.logspace(0, math.log10(target_g) - 0.05, 30)
    Mi = jnp.array(M_tot, dtype=jnp.float64)
    Mf, _ = fn.precalcular_acreccion_lote(
        Mi, N_re, 0.0, L_acc=L, regime="horizon_tracking"
    )
    ratio = np.array(Mf) / target_g
    max_dev = float(np.max(np.abs(ratio - 1.0)))
    check(
        f"pipeline: M_f/(gamma_H*M_H(N_re)) = 1 +/- {max_dev:.2e} para {len(M_tot)} M_i "
        f"(tolerancia relajada a 2e-3 por el piso de precisión del solver)",
        max_dev < 2e-3,
    )


def test_gamma_H_stable_fixed_point():
    dlnM_dN_at_gammaH = 3.0 * fn.GAMMA_H / (2.0 * fn.GAMMA_H)
    check("cap(x=gamma_H) = 3/2 (punto fijo)", abs(dlnM_dN_at_gammaH - 1.5) < 1e-12)
    x_test = 0.9 * fn.GAMMA_H
    dlnM_dN = 3.0 * fn.GAMMA_H / (2.0 * x_test)
    check("cap(x<gamma_H) > 3/2 -> empuja x hacia gamma_H (estable)", dlnM_dN > 1.5)


# ---------------------------------------------------------------------------
# Resto de la ronda 1 (P0-2, P0-5, P1-6, P1-11/P1-12) — sin cambios de fondo
# ---------------------------------------------------------------------------
def test_shift_formula_continuous_at_mu_boundary():
    M_tot = np.logspace(10, 16, 400)
    betas_sbb_full = 1e-20 * (M_tot / 1e13) ** 0.5  # sintético, suave (sin saltos de canal)

    mu_grid = np.linspace(0.9, 1.3, 400)
    M_f_tot = M_tot[200] * mu_grid
    M_tot_fixed = np.full_like(mu_grid, M_tot[200])

    beta_acc = fn.aplicar_corrimiento_acrecion(M_tot_fixed, M_f_tot,
                                                np.interp(M_tot_fixed, M_tot, betas_sbb_full))
    valid = ~np.isnan(beta_acc)
    dlog = np.abs(np.diff(np.log10(beta_acc[valid])))
    max_jump = dlog.max() if dlog.size else 0.0
    check(
        f"empalme mu~1.01 continuo (max |Delta log10 beta|={max_jump:.4f} en rejilla fina)",
        max_jump < 0.05,
    )


def test_relic_floor_does_not_use_shift_formula():
    # Ronda 3, Tarea 3: bug real encontrado en aplicar_corrimiento_acrecion. Cuando un
    # PBH se evapora del todo (M_f queda fijo en M_pl_g, mu ~ 0), M_pl_g casi siempre cae
    # DENTRO del rango tabulado de M_tot -> log_bf[i] es válido -> la fórmula de
    # corrimiento beta_SBB(M_f)/mu se evaluaba con un mu casi nulo en el denominador,
    # dando un número arbitrario en vez de beta_SBB(M_i) (la reliquia depende de la
    # densidad NUMÉRICA de formación, no del corrimiento de masa). Este test construye el
    # caso exacto (M_f == M_pl_g, M_i grande) y verifica que beta_acc == beta_SBB(M_i), no
    # el resultado (erróneo) de dividir entre un mu diminuto.
    M_tot = np.logspace(1, 4, 50)  # 10 g a 1e4 g
    betas_sbb_full = 2e-28 * (M_tot / constants.M_pl_g) ** 1.5  # misma forma que Betas_DM
    M_f_tot = np.full_like(M_tot, constants.M_pl_g)  # TODOS se evaporan del todo

    beta_acc = fn.aplicar_corrimiento_acrecion(M_tot, M_f_tot, betas_sbb_full)
    check(
        "reliquia (M_f=M_pl_g) da beta_SBB(M_i), no beta_SBB(M_f)/mu",
        np.allclose(beta_acc, betas_sbb_full, rtol=1e-12),
        f"max |beta_acc/beta_SBB(Mi) - 1| = {np.max(np.abs(beta_acc/betas_sbb_full - 1)):.3e}",
    )


def test_end_evol_can_trigger():
    M = 1e10  # g
    beta0 = 1e-10
    ratio_M = M / constants.M_pl_g
    alpha_here = float(fn.alpha_of_mass_g(M))
    Delta_t = constants.t_pl * (ratio_M**3) / (3.0 * alpha_here)
    initial = np.array([1.0, Delta_t * 1.5])
    val = fn.end_evol(0.0, initial, M, beta0)
    check("end_evol < 0 cuando el tiempo acumulado excede Delta_t (dispara)", val < 0)


def test_phase1_robust_sweep():
    M_tot = np.logspace(-5, 22, 150)
    Mi = jnp.array(M_tot, dtype=jnp.float64)
    ok = True
    for a_star in [0.0, 0.5, 0.9]:
        for N_fin in [10.0, 20.0, 30.0]:
            try:
                Mf, mu = fn.precalcular_acreccion_lote(Mi, N_fin, a_star)  # wave, default
                if not np.all(np.isfinite(np.array(Mf))):
                    ok = False
            except Exception:
                ok = False
    check("Fase 1 robusta (sin excepciones/NaN) en barrido a* x N_fin (regime='wave')", ok)


def test_get_betas_full_nan_safe():
    fn.constraints.betas_DM_tot = np.array([np.nan, 1e-20, 1e-25])
    fn.constraints.betas_BBN_tot = np.array([1e-18, 1e-19, 1e-30])
    fn.constraints.betas_SD_tot = np.array([])
    fn.constraints.betas_CMB_AN_tot = np.array([])
    fn.constraints.betas_GRB_tot = np.array([])
    fn.constraints.betas_Reio_tot = np.array([])
    fn.constraints.betas_LSP_tot = np.array([])
    out = fn.get_Betas_full(np.array([1.0, 2.0, 3.0]))
    check(
        "get_Betas_full ignora NaN del canal DM y no corrompe todo el arreglo",
        np.isfinite(out).all() and out[0] == 1e-18,
        f"out={out}",
    )


if __name__ == "__main__":
    tests = [
        test_wave_closed_form_mu,
        test_wave_mu_saturates_independent_of_Nre,
        test_wave_runaway_assert_fires,
        test_wave_M_crit_matches_self_consistent_formula,
        test_hawking_lifetime_5e14g_factor_2,
        test_wave_default_is_subcritical,
        test_michel_branch_x_star_unstable,
        test_horizon_tracking_is_delta_function,
        test_gamma_H_stable_fixed_point,
        test_shift_formula_continuous_at_mu_boundary,
        test_relic_floor_does_not_use_shift_formula,
        test_end_evol_can_trigger,
        test_phase1_robust_sweep,
        test_get_betas_full_nan_safe,
    ]
    for t in tests:
        print(f"--- {t.__name__} ---")
        t()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} test(s) fallaron: {FAILURES}")
        sys.exit(1)
    else:
        print("Todos los tests pasaron.")
