import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import equinox as eqx
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5, PIDController
import jax.lax as lax
import numpy as np
from scipy.optimize import fsolve, least_squares
from scipy.integrate import solve_ivp
from numpy import diff
import sys
from scipy import special
from PBHBeta import constants
from PBHBeta import constraints
import functools


# ---------------------------------------------------------------------------
# Constantes en unidades de Planck (usadas internamente en la ODE de acreción)
# ---------------------------------------------------------------------------
_M_pl_GeV  = 1.22089e19    # masa de Planck en GeV
_M_pl_g    = 2.17645e-5    # masa de Planck en gramos
_H_end_GeV = 4.44e13       # H al fin de inflación en GeV
_H_end_pl  = _H_end_GeV / _M_pl_GeV   # ~3.637e-6 en unidades de Planck


def put_M_array(Mass_min, Mass_max):
    """
    Generate an array of primordial black hole (PBH) masses in grams.

    Parameters:
        Mass_min (float): Minimum PBH mass in grams.
        Mass_max (float): Maximum PBH mass in grams.

    Returns:
        np.ndarray: Array of PBH masses.
    """
    i = 0
    M = 0
    delta_M = 0.0123
    M_tot_try = []
    num_values = 20

    mass_array = np.geomspace(Mass_min, 10**(i*delta_M), num_values)

    while M < constraints.data_mass[0]:
        M = 10**(i*delta_M)
        M_tot_try.append(M)
        i += 1

    M_tot_try = np.concatenate((mass_array, M_tot_try, constraints.data_mass))
    M_tot_try = np.unique(M_tot_try)
    M = M_tot_try[-1]

    A = M
    j = 0
    while M < Mass_max:
        j += 1
        M = A * 10**(j * delta_M)
        M_tot_try = np.append(M_tot_try, [M])

    constraints.M_tot = np.array(M_tot_try)
    return constraints.M_tot


# ---------------------------------------------------------------------------
# ODE de acreción — campo escalar oscilante durante el recalentamiento
# ---------------------------------------------------------------------------

@eqx.filter_jit
@functools.partial(jax.vmap, in_axes=(0, None, None))
def precalcular_acreccion_lote(Mi_val_g, N_fin, a_star):
    """
    Incorpora la densidad de envolvente y regularizadores suaves para
    garantizar una compilación y ejecución en segundos.
    """
    # 1. ESCALAS FÍSICAS Y UNIDADES NATURALES
    M_pl_GeV = 1.22089e19
    M_pl_g = 2.17645e-5
    H_end_GeV = 4.44e13 
    
    M_i_pl = Mi_val_g / M_pl_g
    H_end_pl = H_end_GeV / M_pl_GeV

    n = 1000.0 #Este factor es el que más afecta a la acreción
    mu_pl = n * H_end_pl 
    
    rho_end_inf_pl = (3.0 * H_end_pl**2.0) / (8.0 * jnp.pi)
    
    phi_ini_pl = jnp.sqrt(2.0 * rho_end_inf_pl / (mu_pl**2.0 + 9.0 * H_end_pl**2.0 / 4.0))

    M_end_pl = 1.0 / H_end_pl
    N_ini = (2.0 / 3.0) * jnp.log(jnp.maximum(M_i_pl / M_end_pl, 1.0))

    # 2. FUNCIONES DE BACKGROUND (Envolvente optimizada)
    def Hubble(N):
        return H_end_pl * jnp.exp(-3.0 * N / 2.0)

    def rho_inf_field_env(N):
        return 0.5 * (phi_ini_pl * mu_pl)**2 * jnp.exp(-3.0 * N)

    # 3. C4 ACTUALIZADO (Aproximación de fluido w=0 con espín variable)
    def C4(N, M, rho_inf_val):
        w = 0.0 
        a = a_star * M # MODIFICACIÓN: Cálculo del factor de rotación dinámico
        r_plus = M + jnp.sqrt(M**2.0 - a**2.0)
        u_c = M / (2.0 * r_plus)
        rho_h = 3.0 * M / (4.0 * jnp.pi * r_plus**3.0)
        return u_c * (rho_h / rho_inf_val)**(1.0 / (1.0 + w))

    # 4. CAMPO VECTORIAL 
    def vector_field(N, M, args):
        M_safe = jnp.maximum(M, 1.1)
        
        rho_val = rho_inf_field_env(N) 
        H_val   = Hubble(N)
        
        C_4_val = C4(N, M_safe, rho_val)
        a = a_star * M_safe # MODIFICACIÓN: Integración del espín en el horizonte
        r_plus = M_safe + jnp.sqrt(M_safe**2.0 - a**2.0)
        
        f_acc_base = 1e-38 # Eficiencia de acreción
        
        supresion = M_safe * mu_pl
        f_acc = f_acc_base * supresion
        
        tasa_acrecion = f_acc * 4.0 * jnp.pi * C_4_val * r_plus**2.0 * (rho_val / H_val)
        
        return tasa_acrecion 
    
    # 5. SOLVER JAX
    term = ODETerm(vector_field)
    solver = Tsit5()
    stepsize_controller = PIDController(rtol=1e-5, atol=1e-5) 
    saveat = SaveAt(t1=True) 

    def integrar(_):
        sol = diffeqsolve(term, solver, t0=N_ini, t1=N_fin, dt0=None, 
                          y0=M_i_pl, stepsize_controller=stepsize_controller, 
                          saveat=saveat, max_steps=100000)
        return sol.ys[0]
        
    M_final_pl = lax.cond(N_ini < N_fin, integrar, lambda _: M_i_pl, operand=None)
    
    M_final_g = M_final_pl * M_pl_g
    
    return M_final_g, M_final_g / Mi_val_g


def diagnostico_acrecion(M_tot, N_fin, a_star):
    """
    Función de diagnóstico: muestra mu = M_f/M_i vs M_i.

    Llama esto ANTES de calcular restricciones para verificar que la
    acreción produce mu >> 1 en el rango de masas de interés.

    Parámetros:
        M_tot (array): Masas iniciales.
        N_fin (float): Número de E-folds del periodo de recalentamiento.
        a_star (float): Parámetro de espín del PBH (0.0 a 0.999).
    """
    import matplotlib.pyplot as plt

    # MODIFICACIÓN: Pasamos los nuevos argumentos al solver de JAX
    M_f_arr, mu_arr = precalcular_acreccion_lote(jnp.array(M_tot, dtype=jnp.float64), N_fin, a_star)
    M_f_arr = np.array(M_f_arr)
    mu_arr  = np.array(mu_arr)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].loglog(M_tot, M_f_arr, 'b-', label=r'$M_f$ (con acreción)')
    axes[0].loglog(M_tot, M_tot,   'k--', alpha=0.5, label=r'$M_f = M_i$ (sin acreción)')
    axes[0].set_xlabel(r'$M_i$ [g]')
    axes[0].set_ylabel(r'$M_f$ [g]')
    axes[0].set_title('Masa final vs masa inicial')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogx(M_tot, mu_arr, 'r-')
    axes[1].axhline(1.0, color='k', linestyle='--', alpha=0.5, label=r'$\mu=1$ (sin acreción)')
    axes[1].set_xlabel(r'$M_i$ [g]')
    axes[1].set_ylabel(r'$\mu = M_f / M_i$')
    axes[1].set_title(f'Ratio de acreción ($a_*={a_star}$, $N_{{fin}}={N_fin}$)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"mu min = {mu_arr.min():.4f}, mu max = {mu_arr.max():.4f}")
    print(f"Corrimiento esperado: las restricciones se desplazan ~{np.log10(mu_arr.max()):.1f} décadas a la izquierda")

    return M_f_arr, mu_arr


# ---------------------------------------------------------------------------
# Funciones de la dinámica de radiación (incluyendo la evaporación en la métrica de Kerr)
# ---------------------------------------------------------------------------

def diff_rad_rel(ln_rho, initial, M, beta0):
    b    = initial[0]
    Om_0 = beta0 * b * (constants.M_pl_g / M)
    dy   = -(Om_0 - 1.) * b / (Om_0 - 4.)
    return dy


def diff_rad(ln_rho, initial, M, beta0):
    dy     = np.zeros(initial.shape)
    b      = initial[0]
    time   = initial[1]
    Delta_t = constants.t_pl * (M / constants.M_pl_g)**3
    Om_0   = beta0 * b * (1. - time / Delta_t)**(1./3)
    dy[0]  = -(Om_0 - 1.) * b / (Om_0 - 4.)
    dy[1]  = 3**(1./2) * constants.M_pl / ((Om_0 - 4.) * np.exp(ln_rho)**(1./2))
    return dy


def end_evol(ln_rho, initial, M, beta0):
    Delta_t  = constants.t_pl * (M / constants.M_pl_g)**3
    Mass_end = M * (1. - diff_rad(ln_rho, initial, M, beta0)[1] / Delta_t)**(1./3)
    return Mass_end - constants.M_pl_g

end_evol.terminal  = True
end_evol.direction = -1


# ---------------------------------------------------------------------------
# Funciones auxiliares de densidad y escala de modos k
# ---------------------------------------------------------------------------

def k_end_over_k(Mpbh, omega):
    if omega == 1/3:
        res = (Mpbh / (7.1e-2 * constants.gam_rad * (1.8e15 / constants.H_end)))**(1/2)
    else:
        z   = (1 + 3*omega) / (3 * (1 + omega))
        res = np.array((Mpbh * constants.H_end / (3 * constants.gam_rad * constants.M_pl**2))**z)
    return res


def rho_f(Mpbh, omega):
    if omega == 1/3:
        k_ratio = (Mpbh / (7.1e-2 * constants.gam_rad * (1.8e15 / constants.H_end)))**(1/2)
        return constants.rho_end_inf / k_ratio**4
    else:
        z   = (1 + 3*omega) / (3 * (1 + omega))
        res = np.array((Mpbh * constants.H_end / (3 * constants.gam_rad * constants.M_pl**2))**z)
        i   = (6 * (1 + omega)) / (1 + 3*omega)
        return constants.rho_end_inf / res**i


ln_den_end = np.log(constants.rho_end)


# ---------------------------------------------------------------------------
# RESTRICCIONES CON ACRECIÓN
# ---------------------------------------------------------------------------

def Betas_DM(M_tot, omega, M_f_tot=None, mu_tot=None):
    M_n, betas_prim, M_relic, betas_relic_prim = [], [], [], []
    betas_tot, Omegas_tot = [], []
    Omegas, Omegas_relic_pbbn, Omegas_relic = [], [], []
    M_dm, M_dm_rel_pbbn, M_dm_rel = [], [], []

    M_pl_g  = constants.M_pl_g
    t_pl    = constants.t_pl
    gam_rad = constants.gam_rad

    rho_form_rad = rho_f(M_tot, omega)
    rho_end      = constants.rho_end
    ln_den_end_  = np.log(rho_end)

    for i in range(len(M_tot)):
        M_i = M_tot[i]

        if M_f_tot is not None and mu_tot is not None:
            M_f = float(M_f_tot[i])
            mu  = float(mu_tot[i])
        else:
            M_f, mu = M_i, 1.0

        if M_f <= _M_pl_g or mu <= 0.0:
            betas_tot.append(constants.ev1)
            continue

        if M_f > 4.1e14:
            M_n.append(M_i)
            beta_std = 1.86e-18 * (M_f / 1e15)**(1/2)
            beta = beta_std / mu
            betas_prim.append(beta)

        elif M_f < 1e11 * M_pl_g:
            M_relic.append(M_i)
            beta_std = 2e-28 * (M_f / M_pl_g)**(3/2)
            beta     = beta_std / mu
            betas_relic_prim.append(beta)

        else:
            beta = constants.ev1

        betas_tot.append(beta / gam_rad**(1/2))

    betas_prim       = np.array(betas_prim)
    betas_relic_prim = np.array(betas_relic_prim)
    betas_tot        = np.array(betas_tot)
    constraints.betas_DM_tot = betas_tot

    M_n     = np.array(M_n)
    M_relic = np.array(M_relic)
    betas         = betas_prim / gam_rad**(1/2)
    betas_relic   = betas_relic_prim / gam_rad**(1/2)

    for i in range(len(M_tot)):
        if i >= len(betas_tot):
            Omegas_tot.append(constants.ev2)
            continue

        M_f = float(M_f_tot[i]) if M_f_tot is not None else M_tot[i]

        if betas_tot[i] == constants.ev1 / gam_rad**(1/2):
            Omegas_tot.append(constants.ev2)
            continue

        ln_den_f = np.log(rho_form_rad[i])
        if ln_den_f <= ln_den_end_:
            Omegas_tot.append(constants.ev2)
            continue

        ln_den  = np.linspace(ln_den_f, ln_den_end_, 10000)
        sol_try = solve_ivp(diff_rad, (ln_den_f, ln_den_end_), np.array([1., 0.]),
                            events=end_evol, t_eval=ln_den,
                            args=(M_f, betas_tot[i]), method="DOP853")

        if sol_try.t[-1] > ln_den_end_:
            sol_try = solve_ivp(diff_rad_rel, (ln_den_f, ln_den_end_), np.array([1.]),
                                t_eval=ln_den, args=(M_f, betas_tot[i]), method="DOP853")
            y = betas_tot[i] * sol_try.y[0][-1] * (M_pl_g / M_f)
            if M_f < 1e11 * M_pl_g:
                Omegas_relic_pbbn.append(y)
                M_dm_rel_pbbn.append(M_tot[i])
        else:
            Delta_t = t_pl * (M_f / M_pl_g)**3
            y = betas_tot[i] * sol_try.y[0][-1] * (1. - sol_try.y[1][-1] / Delta_t)**(1./3)
            if M_f > 4.1e14:
                Omegas.append(y)
                M_dm.append(M_tot[i])
            elif M_f < 1e11 * M_pl_g:
                Omegas_relic.append(y)
                M_dm_rel.append(M_tot[i])

        Omegas_tot.append(y)

    Omegas_tot = np.array(Omegas_tot)
    constraints.Omega_DM_tot = Omegas_tot

    return M_n, betas, M_relic, betas_relic, Omegas_tot


def Betas_BBN(M_tot, omega, M_f_tot=None, mu_tot=None):
    betas_bbn, M_bbn, M_bbn_bbn = [], [], []
    Omegas_bbn, Omegas_bbn_tot, Omegas_bbn_pbbn, M_bbn_pbbn = [], [], [], []

    rho_form_rad = rho_f(M_tot, omega)

    for i in range(len(M_tot)):
        M_i = M_tot[i]

        if M_f_tot is not None and mu_tot is not None:
            M_f = float(M_f_tot[i])
            mu  = float(mu_tot[i])
        else:
            M_f, mu = M_i, 1.0

        if M_f <= _M_pl_g or mu <= 0.0:
            constraints.betas_BBN_tot.append(constants.ev1)
            Omegas_bbn_tot.append(constants.ev2)
            continue

        ln_den_f = np.log(rho_form_rad[i])

        if constraints.data_mass[0] <= M_f < constraints.data_mass[76]:
            M_bbn.append(M_i)
            abundancia = np.interp(M_f, constraints.data_mass, constraints.data_abundances)
            beta       = (abundancia / constants.gam_rad**(1/2)) / mu
            betas_bbn.append(beta)

            if ln_den_f > ln_den_end:
                ln_den  = np.linspace(ln_den_f, ln_den_end, 10000)
                sol_try = solve_ivp(diff_rad, (ln_den_f, ln_den_end), np.array([1., 0.]),
                                    events=end_evol, t_eval=ln_den,
                                    args=(M_f, beta), method="DOP853")
                if sol_try.t[-1] > ln_den_end:
                    sol_try = solve_ivp(diff_rad_rel, (ln_den_f, ln_den_end), np.array([1.]),
                                        t_eval=ln_den, args=(M_f, beta), method="DOP853")
                    y = beta * sol_try.y[0][-1] * (constants.M_pl_g / M_f)
                    Omegas_bbn_pbbn.append(y)
                    M_bbn_pbbn.append(M_i)
                else:
                    Delta_t = constants.t_pl * (M_f / constants.M_pl_g)**3
                    y = beta * sol_try.y[0][-1] * (1. - sol_try.y[1][-1] / Delta_t)**(1./3)
                    Omegas_bbn.append(y)
                    M_bbn_bbn.append(M_i)
            else:
                y = constants.ev2

        elif constraints.data_mass[76] <= M_f < 2.5e13:
            M_bbn.append(M_i)
            abundancia = np.interp(M_f, constraints.data_mass, constraints.data_abundances)
            beta       = (abundancia / constants.gam_rad**(1/2)) / mu
            betas_bbn.append(beta)

            if ln_den_f > ln_den_end:
                ln_den  = np.linspace(ln_den_f, ln_den_end, 10000)
                sol_try = solve_ivp(diff_rad, (ln_den_f, ln_den_end), np.array([1., 0.]),
                                    events=end_evol, t_eval=ln_den,
                                    args=(M_f, beta), method="DOP853")
                Delta_t = constants.t_pl * (M_f / constants.M_pl_g)**3
                y = beta * sol_try.y[0][-1] * (1. - sol_try.y[1][-1] / Delta_t)**(1./3)
                Omegas_bbn.append(y)
                M_bbn_bbn.append(M_i)
            else:
                y = constants.ev2

        else:
            beta = constants.ev1
            y    = constants.ev2

        constraints.betas_BBN_tot.append(beta)
        Omegas_bbn_tot.append(y)

    constraints.Omega_BBN_tot = np.array(Omegas_bbn_tot)
    return np.array(M_bbn), np.array(betas_bbn), np.array(Omegas_bbn_tot)


def Betas_SD(M_tot, omega, M_f_tot=None, mu_tot=None):
    betas_sd, M_sd, M_sd_bbn, Omegas_sd = [], [], [], []
    rho_form_rad = rho_f(M_tot, omega)

    for i in range(len(M_tot)):
        M_i = M_tot[i]

        if M_f_tot is not None and mu_tot is not None:
            M_f = float(M_f_tot[i])
            mu  = float(mu_tot[i])
        else:
            M_f, mu = M_i, 1.0

        if M_f > _M_pl_g and 1e11 < M_f < 1e13:
            M_sd.append(M_i)
            beta = (1e-21 / constants.gam_rad**(1/2)) / mu
            betas_sd.append(beta)

            ln_den_f = np.log(rho_form_rad[i])
            if ln_den_f > ln_den_end:
                ln_den  = np.linspace(ln_den_f, ln_den_end, 10000)
                sol_try = solve_ivp(diff_rad, (ln_den_f, ln_den_end), np.array([1., 0.]),
                                    events=end_evol, t_eval=ln_den,
                                    args=(M_f, beta), method="DOP853")
                Delta_t = constants.t_pl * (M_f / constants.M_pl_g)**3
                y = beta * sol_try.y[0][-1] * (1. - sol_try.y[1][-1] / Delta_t)**(1./3)
                Omegas_sd.append(y)
                M_sd_bbn.append(M_i)
            else:
                y = constants.ev2
        else:
            beta = constants.ev1
            y    = constants.ev2

        constraints.betas_SD_tot.append(beta)
        constraints.Omega_SD_tot.append(y)

    return np.array(M_sd), np.array(betas_sd), np.array(Omegas_sd)


def Betas_CMB_AN(M_tot, omega, M_f_tot=None, mu_tot=None):
    betas_an, M_an, M_an_bbn, Omegas_an = [], [], [], []
    rho_form_rad = rho_f(M_tot, omega)

    for i in range(len(M_tot)):
        M_i = M_tot[i]

        if M_f_tot is not None and mu_tot is not None:
            M_f = float(M_f_tot[i])
            mu  = float(mu_tot[i])
        else:
            M_f, mu = M_i, 1.0

        if M_f > _M_pl_g and 2.5e13 < M_f < 2.4e14:
            M_an.append(M_i)
            beta = (3e-30 * (M_f / 1e13)**3.1 / constants.gam_rad**(1/2)) / mu
            betas_an.append(beta)

            ln_den_f = np.log(rho_form_rad[i])
            if ln_den_f > ln_den_end:
                ln_den  = np.linspace(ln_den_f, ln_den_end, 10000)
                sol_try = solve_ivp(diff_rad, (ln_den_f, ln_den_end), np.array([1., 0.]),
                                    events=end_evol, t_eval=ln_den,
                                    args=(M_f, beta), method="DOP853")
                Delta_t = constants.t_pl * (M_f / constants.M_pl_g)**3
                y = beta * sol_try.y[0][-1] * (1. - sol_try.y[1][-1] / Delta_t)**(1./3)
                Omegas_an.append(y)
                M_an_bbn.append(M_i)
            else:
                y = constants.ev2
        else:
            beta = constants.ev1
            y    = constants.ev2

        constraints.betas_CMB_AN_tot.append(beta)
        constraints.Omega_CMB_AN_tot.append(y)

    return np.array(M_an), np.array(betas_an), np.array(Omegas_an)


def Betas_GRB(M_tot, omega, M_f_tot=None, mu_tot=None):
    betas_grb1, M_grb1, betas_grb2, M_grb2 = [], [], [], []
    M_grb1_bbn, Omegas_grb1, M_grb2_bbn, Omegas_grb2 = [], [], [], []
    rho_form_rad = rho_f(M_tot, omega)

    for i in range(len(M_tot)):
        M_i = M_tot[i]

        if M_f_tot is not None and mu_tot is not None:
            M_f = float(M_f_tot[i])
            mu  = float(mu_tot[i])
        else:
            M_f, mu = M_i, 1.0

        ln_den_f = np.log(rho_form_rad[i])
        beta     = constants.ev1
        y        = constants.ev2

        if M_f > _M_pl_g and 3e13 < M_f < 4.1e14:
            M_grb1.append(M_i)
            beta = (5e-28 * (M_f / 4.1e14)**(-3.3) / constants.gam_rad**(1/2)) / mu
            betas_grb1.append(beta)

            if ln_den_f > ln_den_end:
                ln_den  = np.linspace(ln_den_f, ln_den_end, 10000)
                sol_try = solve_ivp(diff_rad, (ln_den_f, ln_den_end), np.array([1., 0.]),
                                    events=end_evol, t_eval=ln_den,
                                    args=(M_f, beta), method="DOP853")
                Delta_t = constants.t_pl * (M_f / constants.M_pl_g)**3
                y = beta * sol_try.y[0][-1] * (1. - sol_try.y[1][-1] / Delta_t)**(1./3)
                Omegas_grb1.append(y)
                M_grb1_bbn.append(M_i)

        elif M_f > _M_pl_g and 4.1e14 < M_f < 7e16:
            M_grb2.append(M_i)
            beta = (5e-26 * (M_f / 4.1e14)**3.9 / constants.gam_rad**(1/2)) / mu
            betas_grb2.append(beta)

            if ln_den_f > ln_den_end:
                ln_den  = np.linspace(ln_den_f, ln_den_end, 10000)
                sol_try = solve_ivp(diff_rad, (ln_den_f, ln_den_end), np.array([1., 0.]),
                                    events=end_evol, t_eval=ln_den,
                                    args=(M_f, beta), method="DOP853")
                Delta_t = constants.t_pl * (M_f / constants.M_pl_g)**3
                y = beta * sol_try.y[0][-1] * (1. - sol_try.y[1][-1] / Delta_t)**(1./3)
                Omegas_grb2.append(y)
                M_grb2_bbn.append(M_i)

        constraints.betas_GRB_tot.append(beta)
        constraints.Omega_GRB_tot.append(y)

    return (np.array(M_grb1), np.array(M_grb2),
            np.array(betas_grb1), np.array(betas_grb2),
            np.array(Omegas_grb1), np.array(Omegas_grb2))


def Betas_Reio(M_tot, omega, M_f_tot=None, mu_tot=None):
    betas_reio, M_reio, M_reio_bbn, Omegas_reio = [], [], [], []
    rho_form_rad = rho_f(M_tot, omega)

    for i in range(len(M_tot)):
        M_i = M_tot[i]

        if M_f_tot is not None and mu_tot is not None:
            M_f = float(M_f_tot[i])
            mu  = float(mu_tot[i])
        else:
            M_f, mu = M_i, 1.0

        ln_den_f = np.log(rho_form_rad[i])
        beta     = constants.ev1
        y        = constants.ev2

        if M_f > _M_pl_g and 1e15 < M_f < 1e17:
            M_reio.append(M_i)
            beta = (2.4e-26 * (M_f / 4.1e14)**4.3 / constants.gam_rad**(1/2)) / mu
            betas_reio.append(beta)

            if ln_den_f > ln_den_end:
                ln_den  = np.linspace(ln_den_f, ln_den_end, 10000)
                sol_try = solve_ivp(diff_rad, (ln_den_f, ln_den_end), np.array([1., 0.]),
                                    events=end_evol, t_eval=ln_den,
                                    args=(M_f, beta), method="DOP853")
                Delta_t = constants.t_pl * (M_f / constants.M_pl_g)**3
                y = beta * sol_try.y[0][-1] * (1. - sol_try.y[1][-1] / Delta_t)**(1./3)
                Omegas_reio.append(y)
                M_reio_bbn.append(M_i)

        constraints.betas_Reio_tot.append(beta)
        constraints.Omega_Reio_tot.append(y)

    return np.array(M_reio), np.array(betas_reio), np.array(Omegas_reio)


def Betas_LSP(M_tot, w, M_f_tot=None, mu_tot=None):
    t_pl    = constants.t_pl
    gam_rad = constants.gam_rad

    betas_lsp, betas_lsp_tot = [], []
    M_lsp, M_lsp_bbn, Omegas_lsp = [], [], []
    M_lsp_pbbn, Omegas_lsp_pbbn, Omegas_lsp_tot = [], [], []

    rho_form_rad = rho_f(M_tot, w)
    rho_end_loc  = constants.rho_end
    ln_den_end_  = np.log(rho_end_loc)

    for i in range(len(M_tot)):
        M_i = M_tot[i]

        if M_f_tot is not None and mu_tot is not None:
            M_f = float(M_f_tot[i])
            mu  = float(mu_tot[i])
        else:
            M_f, mu = M_i, 1.0

        beta = constants.ev1
        y    = constants.ev2

        if M_f > _M_pl_g and M_f < 1e11:
            M_lsp.append(M_i)
            beta = (1e-18 * (M_f / 1e11)**(-1/2) / gam_rad**(1/2)) / mu
            betas_lsp.append(beta)

            ln_den_f = np.log(rho_form_rad[i])
            if ln_den_f > ln_den_end_:
                ln_den  = np.linspace(ln_den_f, ln_den_end_, 10000)
                sol_try = solve_ivp(diff_rad, (ln_den_f, ln_den_end_), np.array([1., 0.]),
                                    events=end_evol, t_eval=ln_den,
                                    args=(M_f, beta), method="DOP853")
                if sol_try.t[-1] > ln_den_end_:
                    sol_try = solve_ivp(diff_rad_rel, (ln_den_f, ln_den_end_), np.array([1.]),
                                        t_eval=ln_den, args=(M_f, beta), method="DOP853")
                    y = beta * sol_try.y[0][-1] * (constants.M_pl_g / M_f)
                    Omegas_lsp_pbbn.append(y)
                    M_lsp_pbbn.append(M_i)
                else:
                    Delta_t = t_pl * (M_f / constants.M_pl_g)**3
                    y = beta * sol_try.y[0][-1] * (1. - sol_try.y[1][-1] / Delta_t)**(1./3)
                    Omegas_lsp.append(y)
                    M_lsp_bbn.append(M_i)

        constraints.betas_LSP_tot.append(beta)
        constraints.Omega_LSP_tot.append(y)
        betas_lsp_tot.append(beta)
        Omegas_lsp_tot.append(y)

    return np.array(M_lsp), np.array(betas_lsp), np.array(Omegas_lsp_tot)


# ---------------------------------------------------------------------------
# Funciones de envoltura para obtener la restricción completa
# ---------------------------------------------------------------------------

def get_Betas_full(M_tot):
    DM_tot   = np.array(constraints.betas_DM_tot)
    BBN_tot  = np.array(constraints.betas_BBN_tot)
    SD_tot   = np.array(constraints.betas_SD_tot)
    CMB_tot  = np.array(constraints.betas_CMB_AN_tot)
    GRB_tot  = np.array(constraints.betas_GRB_tot)
    Reio_tot = np.array(constraints.betas_Reio_tot)
    LSP_tot  = np.array(constraints.betas_LSP_tot)

    constraints.betas_full = M_tot * 0

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
            constraints.betas_full[i] = min(values)

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
            constraints.Omegas_full[i] = min(values)

    return constraints.Omegas_full


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def inverse_error(betas, delta_c):
    return [delta_c / (np.sqrt(2) * special.erfcinv(b)) for b in betas]


def a_endre(rho_r0, rho_end_re):
    return (rho_r0 / rho_end_re)**(1./4)


def k_rad(M):
    a_end_inf_rad     = (constants.rho_r0 / constants.rho_end_inf)**(1./4)
    k_end             = a_end_inf_rad * constants.H_end
    k_end_over_k_rad  = (M / (7.1e-2 * constants.gam_rad * (1.8e15 / constants.H_end)))**(1/2)
    k = (k_end / k_end_over_k_rad) * constants.GeV * constants.metter_m1
    return np.array(k)