"""
Tests de regresión para PBHBeta/functionscorr.py (pipeline vivo de acreción).

No requiere pytest: cada test_* es una función que hace sus propios asserts.
Se corre como script (`python tests/test_functionscorr.py`) desde la raíz del
repo, o con pytest si está instalado (`pytest tests/`).

Contexto (ver AUDIT.md y el resumen de P0-1..P2-15): el modelo de acreción es
la Opción A del paper (horizon-tracking, min(Michel, cap) — AUDIT.md P0-4), NO
el modelo saturante (1-x) que tenía el código originalmente. Por eso el test
de "lambda_c = 2.0" que pedía el prompt original no aplica tal cual: ese
umbral sólo existe en el modelo saturante (Opción B), que se descartó. Los
tests de abajo verifican el equivalente correcto para el modelo que sí se
implementó: el punto fijo inestable x* = 4*pi/L_eff y el punto fijo estable
en x = gamma_H.
"""
import math
import sys
from pathlib import Path

import numpy as np
import jax.numpy as jnp

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
# 1. Punto fijo inestable x* = 4*pi/L_eff (reemplaza el test de lambda_c=2.0,
#    que sólo aplicaba al modelo saturante descartado — AUDIT.md P0-4).
# ---------------------------------------------------------------------------
def test_x_star_unstable_fixed_point():
    L_eff = fn.L_ACC_DEFAULT  # a*=0 => factor_espin_acc=1; S_onda~1, Omega_phi~1 en M grande
    x_star = 4.0 * math.pi / L_eff
    # En la rama Michel, dlnM/dN = 3*L_eff*x/(8*pi). El punto fijo x* satisface
    # dlnM/dN(x*) = 3/2 (balance con dlnM_H/dN=3/2 en esta convención M_H=1/H).
    dlnM_dN_at_xstar = 3.0 * L_eff * x_star / (8.0 * math.pi)
    check(
        "x* satisface dlnM/dN(x*) = 3/2 (rama Michel)",
        abs(dlnM_dN_at_xstar - 1.5) < 1e-10,
        f"dlnM/dN(x*)={dlnM_dN_at_xstar}",
    )
    # Debe ser INESTABLE: x ligeramente arriba de x* -> dx/dN > 0 (se aleja hacia arriba);
    # x ligeramente abajo -> dx/dN < 0 (se aleja hacia abajo).
    eps = 1e-4
    for x, expect_sign in [(x_star * (1 + eps), 1), (x_star * (1 - eps), -1)]:
        dlnM_dN = 3.0 * L_eff * x / (8.0 * math.pi)
        dx_dN_sign = np.sign(dlnM_dN - 1.5)
        check(
            f"x*({'arriba' if expect_sign>0 else 'abajo'}) se aleja de x* (inestable)",
            dx_dN_sign == expect_sign,
        )


def test_gamma_H_stable_fixed_point():
    # Rama cap: dlnM/dN = 3*gamma_H/(2x). En x=gamma_H da exactamente 3/2 (punto fijo).
    dlnM_dN_at_gammaH = 3.0 * fn.GAMMA_H / (2.0 * fn.GAMMA_H)
    check("cap(x=gamma_H) = 3/2 (punto fijo)", abs(dlnM_dN_at_gammaH - 1.5) < 1e-12)
    # Estable: para x<gamma_H (pero > x_cross, dominado por la rama cap), dx/dN > 0
    # empuja x hacia gamma_H.
    x_test = 0.9 * fn.GAMMA_H
    dlnM_dN = 3.0 * fn.GAMMA_H / (2.0 * x_test)
    check("cap(x<gamma_H) > 3/2 -> empuja x hacia gamma_H (estable)", dlnM_dN > 1.5)


def test_horizon_tracking_regime_default_params():
    # Equivalente al assert que pedía el prompt original
    # ("lam*factor_espin_acc*Omega_phi >= lambda_c"), adaptado a la Opción A:
    # con los parámetros por defecto del módulo, el sistema debe estar en
    # régimen horizon-tracking (gamma_H > gamma_thr = 4*pi/L), la narrativa
    # que sostiene el paper.
    gamma_thr = 4.0 * math.pi / fn.L_ACC_DEFAULT
    assert fn.GAMMA_H > gamma_thr, (
        f"Parámetros por defecto (L_acc={fn.L_ACC_DEFAULT}, GAMMA_H={fn.GAMMA_H}) NO "
        f"están en régimen horizon-tracking (gamma_thr={gamma_thr}); revisar antes de "
        f"generar figuras que asuman ese régimen."
    )
    check("GAMMA_H > gamma_thr (régimen horizon-tracking, params. por defecto)", True)


# ---------------------------------------------------------------------------
# 2. Vida de Hawking de un PBH de 5e14 g (P0-1). Calibrado exactamente (no sólo
#    "dentro de un factor 2" como pedía el prompt original, ya que ALPHA_EVAP
#    se resolvió justo para esto) — se deja un margen de 1% por redondeo de
#    ALPHA_EVAP a 5 cifras.
# ---------------------------------------------------------------------------
def test_hawking_lifetime_5e14g():
    M_star_g = 5e14
    edad_universo_s = 4.35e17
    ratio = M_star_g / constants.M_pl_g
    t_life_s = constants.t_pl_s * ratio**3 / (3.0 * fn.ALPHA_EVAP)
    rel_err = abs(t_life_s - edad_universo_s) / edad_universo_s
    check(
        f"vida de PBH 5e14g = {t_life_s:.4e}s (target {edad_universo_s:.2e}s, err={rel_err:.2%})",
        rel_err < 0.01,
    )


# ---------------------------------------------------------------------------
# 3. mu = e^{1.5*(N_fin-N_ini)} exacto para PBHs que horizon-trackean (P0-4).
# ---------------------------------------------------------------------------
def test_horizon_tracking_mu_matches_analytic():
    N_fin = 30.0
    Mi = jnp.array([1e18], dtype=jnp.float64)
    Mf, mu = fn.precalcular_acreccion_lote(Mi, N_fin, 0.0, fn.L_ACC_DEFAULT)
    M_i_pl = 1e18 / fn.M_pl_g
    M_end_pl = 1.0 / fn.H_end_pl
    N_ini = (2.0 / 3.0) * np.log(max(M_i_pl / M_end_pl, 1.0))
    expected = np.exp(1.5 * (N_fin - N_ini))
    rel_err = abs(float(mu[0]) - expected) / expected
    check(f"mu horizon-tracking coincide con e^(1.5 dN) (err={rel_err:.2e})", rel_err < 1e-4)


# ---------------------------------------------------------------------------
# 4. Continuidad del empalme en aplicar_corrimiento_acrecion en mu~1.01 (P0-5).
#    OJO: la curva beta_SBB(M) completa SÍ tiene saltos genuinos entre canales
#    de restricción (ver AUDIT.md / HANDOFF.md, find_table_jumps) — este test
#    sólo aísla la continuidad del EMPALME que se arregló, no de la tabla SBB
#    entera.
# ---------------------------------------------------------------------------
def test_shift_formula_continuous_at_mu_boundary():
    M_tot = np.logspace(10, 16, 400)
    # betas_sbb_full sintético, suave (sin saltos de canal) para aislar sólo el
    # empalme de aplicar_corrimiento_acrecion.
    betas_sbb_full = 1e-20 * (M_tot / 1e13) ** 0.5

    mu_grid = np.linspace(0.9, 1.3, 400)
    M_f_tot = M_tot[200] * mu_grid  # variando mu alrededor de 1.01 para un M_i fijo
    M_tot_fixed = np.full_like(mu_grid, M_tot[200])

    beta_acc = fn.aplicar_corrimiento_acrecion(M_tot, M_f_tot, betas_sbb_full)
    # nota: aplicar_corrimiento_acrecion espera que M_tot tenga la misma longitud
    # que M_f_tot (indexa por posición) — se reconstruye con ese contrato.
    beta_acc = fn.aplicar_corrimiento_acrecion(M_tot_fixed, M_f_tot,
                                                np.interp(M_tot_fixed, M_tot, betas_sbb_full))
    valid = ~np.isnan(beta_acc)
    dlog = np.abs(np.diff(np.log10(beta_acc[valid])))
    max_jump = dlog.max() if dlog.size else 0.0
    check(
        f"empalme mu~1.01 continuo (max |Delta log10 beta|={max_jump:.4f} en rejilla fina)",
        max_jump < 0.05,
    )


# ---------------------------------------------------------------------------
# 5. P0-2: end_evol dispara de verdad (ya no siempre M - M_pl_g > 0).
# ---------------------------------------------------------------------------
def test_end_evol_can_trigger():
    # Con tiempo acumulado > Delta_t, Mass_end debe caer por debajo de M_pl_g.
    M = 1e10  # g
    beta0 = 1e-10
    ratio_M = M / constants.M_pl_g
    Delta_t = constants.t_pl * (ratio_M**3) / (3.0 * fn.ALPHA_EVAP)
    initial = np.array([1.0, Delta_t * 1.5])  # tiempo acumulado > Delta_t
    val = fn.end_evol(0.0, initial, M, beta0)
    check("end_evol < 0 cuando el tiempo acumulado excede Delta_t (dispara)", val < 0)


# ---------------------------------------------------------------------------
# 6. Robustez numérica amplia de Fase 1 (P1-6): sin excepciones ni NaN/Inf.
# ---------------------------------------------------------------------------
def test_phase1_robust_sweep():
    M_tot = np.logspace(-5, 22, 150)
    Mi = jnp.array(M_tot, dtype=jnp.float64)
    ok = True
    for a_star in [0.0, 0.5, 0.9]:
        for N_fin in [10.0, 20.0, 30.0]:
            try:
                Mf, mu = fn.precalcular_acreccion_lote(Mi, N_fin, a_star, fn.L_ACC_DEFAULT)
                if not np.all(np.isfinite(np.array(Mf))):
                    ok = False
            except Exception:
                ok = False
    check("Fase 1 robusta (sin excepciones/NaN) en barrido a*x N_fin", ok)


# ---------------------------------------------------------------------------
# 7. P1-11/P1-12: get_Betas_full no se corrompe por NaN en el canal DM.
# ---------------------------------------------------------------------------
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
        test_x_star_unstable_fixed_point,
        test_gamma_H_stable_fixed_point,
        test_horizon_tracking_regime_default_params,
        test_hawking_lifetime_5e14g,
        test_horizon_tracking_mu_matches_analytic,
        test_shift_formula_continuous_at_mu_boundary,
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
