import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import equinox as eqx
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5, PIDController
import jax.lax as lax
import numpy as np
from scipy.integrate import solve_ivp
from scipy import special
from PBHBeta import constants
from PBHBeta import constraints
import functools

# ---------------------------------------------------------------------------
# Constantes en unidades de Planck
# ---------------------------------------------------------------------------
_M_pl_GeV  = 1.22089e19    
_M_pl_g    = 2.17645e-5    
_H_end_GeV = 4.44e13       
_H_end_pl  = _H_end_GeV / _M_pl_GeV   

def put_M_array(Mass_min, Mass_max):
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
# FASE 1: Acreción y Evaporación de Kerr durante Recalentamiento (JAX)
# ---------------------------------------------------------------------------

@eqx.filter_jit
@functools.partial(jax.vmap, in_axes=(0, None, None))
def precalcular_acreccion_lote(Mi_val_g, N_fin, a_star):
    M_pl_GeV = 1.22089e19
    M_pl_g = 2.17645e-5
    H_end_GeV = 4.44e13 
    
    M_i_pl = Mi_val_g / M_pl_g
    H_end_pl = H_end_GeV / M_pl_GeV

    n = 1000.0 
    mu_pl = n * H_end_pl 
    rho_end_inf_pl = (3.0 * H_end_pl**2.0) / (8.0 * jnp.pi)
    phi_ini_pl = jnp.sqrt(2.0 * rho_end_inf_pl / (mu_pl**2.0 + 9.0 * H_end_pl**2.0 / 4.0))

    M_end_pl = 1.0 / H_end_pl
    N_ini = (2.0 / 3.0) * jnp.log(jnp.maximum(M_i_pl / M_end_pl, 1.0))
    
    a_star_safe = jnp.clip(a_star, 0.0, 0.999)

    # Factores de Kerr para la evaporación
    raiz_espin = jnp.sqrt(jnp.maximum(1.0 - a_star_safe**2.0, 1e-12))
    factor_T_kerr = (2.0 * raiz_espin) / (1.0 + raiz_espin)
    factor_Area_kerr = 0.5 * (1.0 + raiz_espin)
    factor_evap_kerr = (factor_T_kerr**4.0) * factor_Area_kerr

    def Hubble(N):
        return H_end_pl * jnp.exp(-3.0 * N / 2.0)

    def rho_inf_field_env(N):
        return 0.5 * (phi_ini_pl * mu_pl)**2 * jnp.exp(-3.0 * N)

    def C4(N, M, rho_inf_val):
        w = 0.0 
        a = a_star_safe * M 
        r_plus = M + jnp.sqrt(jnp.maximum(M**2.0 - a**2.0, 1e-12))
        u_c = M / (2.0 * r_plus)
        rho_h = 3.0 * M / (4.0 * jnp.pi * r_plus**3.0)
        return u_c * (rho_h / rho_inf_val)**(1.0 / (1.0 + w))

    def vector_field(N, M, args):
        M_safe = jnp.maximum(M, 1.1)
        rho_val = rho_inf_field_env(N) 
        H_val   = Hubble(N)
        
        # 1. Tasa de acreción
        C_4_val = C4(N, M_safe, rho_val)
        a = a_star_safe * M_safe 
        r_plus = M_safe + jnp.sqrt(jnp.maximum(M_safe**2.0 - a**2.0, 1e-12))
        f_acc_base = 1e-38 
        supresion = M_safe * mu_pl
        f_acc = f_acc_base * supresion
        dM_dN_acrecion = f_acc * 4.0 * jnp.pi * C_4_val * r_plus**2.0 * (rho_val / H_val)
        
        # 2. Tasa de evaporación de Kerr
        dM_dt_evap_pl = - (1.0 / (3.0 * M_safe**2.0)) * factor_evap_kerr
        dM_dN_evap_real = dM_dt_evap_pl / H_val
        dM_dN_evap = jnp.maximum(dM_dN_evap_real, -500.0 * M_safe) # Limitador de rigidez
        
        derivada_total = dM_dN_acrecion + dM_dN_evap
        
        # Si M llega a Planck, apagamos la derivada
        return jnp.where(M < 1.1, 0.0, derivada_total)
    
    term = ODETerm(vector_field)
    solver = Tsit5()
    stepsize_controller = PIDController(rtol=1e-4, atol=1e-4) 
    saveat = SaveAt(t1=True) 

    def integrar(_):
        N_ini_safe = jnp.minimum(N_ini, N_fin)
        sol = diffeqsolve(term, solver, t0=N_ini_safe, t1=N_fin, dt0=None, 
                          y0=M_i_pl, stepsize_controller=stepsize_controller, 
                          saveat=saveat, max_steps=100000)
        return sol.ys[0]
        
    M_final_pl = lax.cond(N_ini < N_fin, integrar, lambda _: M_i_pl, operand=None)
    M_final_g = M_final_pl * M_pl_g
    
    return M_final_g, M_final_g / Mi_val_g


def diagnostico_acrecion(M_tot, N_fin, a_star):
    import matplotlib.pyplot as plt

    M_f_arr, mu_arr = precalcular_acreccion_lote(jnp.array(M_tot, dtype=jnp.float64), N_fin, a_star)
    M_f_arr = np.array(M_f_arr)
    mu_arr  = np.array(mu_arr)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].loglog(M_tot, M_f_arr, 'b-', label=r'$M_f$ (con dinámica)')
    axes[0].loglog(M_tot, M_tot,   'k--', alpha=0.5, label=r'$M_f = M_i$ (sin dinámica)')
    axes[0].set_xlabel(r'$M_i$ [g]')
    axes[0].set_ylabel(r'$M_f$ [g]')
    axes[0].set_title('Masa final vs masa inicial')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogx(M_tot, mu_arr, 'r-')
    axes[1].axhline(1.0, color='k', linestyle='--', alpha=0.5, label=r'$\mu=1$')
    axes[1].set_xlabel(r'$M_i$ [g]')
    axes[1].set_ylabel(r'$\mu = M_f / M_i$')
    axes[1].set_title(f'Ratio de acreción ($a_*={a_star}$, $N_{{fin}}={N_fin}$)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return M_f_arr, mu_arr


# ---------------------------------------------------------------------------
#Evaporación analítica en la era de Radiación 
# ---------------------------------------------------------------------------

def diff_rad_rel(ln_rho, initial, M, beta0):
    b    = initial[0]
    Om_0 = beta0 * b * (constants.M_pl_g / M)
    dy   = -(Om_0 - 1.) * b / (Om_0 - 4.)
    return dy

def diff_rad(ln_rho, initial, M, beta0):
    """
    Evolución de abundancia usando el resultado analítico exacto de PBHBeta:
    M(t) = M_f * (1 - t / Delta_t)^(1/3)
    """
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
# Funciones auxiliares
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
# RESTRICCIONES CON ACRECIÓN (Mapeando M_f a M_i)
# ---------------------------------------------------------------------------

def Betas_DM(M_tot, omega, M_f_tot=None, mu_tot=None):
    M_n, betas_prim, M_relic, betas_relic_prim = [], [], [], []
    betas_tot, Omegas_tot = [], []
    Omegas, Omegas_relic_pbbn, Omegas_relic = [], [], []
    M_dm, M_dm_rel_pbbn, M_dm_rel = [], [], []

    rho_form_rad = rho_f(M_tot, omega)
    ln_den_end_  = np.log(constants.rho_end)
    limite_planck = 1.5 * constants.M_pl_g

    for i in range(len(M_tot)):
        M_i = M_tot[i]
        M_f = float(M_f_tot[i]) if M_f_tot is not None else M_i
        mu  = float(mu_tot[i]) if mu_tot is not None else 1.0

        if M_f <= limite_planck or mu <= 0.0:
            betas_tot.append(constants.ev1)
            continue

        if M_f > 4.1e14:
            M_n.append(M_i)
            beta_std = 1.86e-18 * (M_f / 1e15)**0.5
            betas_prim.append(beta_std / mu)
            betas_tot.append(betas_prim[-1] / constants.gam_rad**0.5)
        elif M_f < 1e11 * constants.M_pl_g:
            M_relic.append(M_i)
            beta_std = 2e-28 * (M_f / constants.M_pl_g)**1.5
            betas_relic_prim.append(beta_std / mu)
            betas_tot.append(betas_relic_prim[-1] / constants.gam_rad**0.5)
        else:
            betas_tot.append(constants.ev1)

    constraints.betas_DM_tot = np.array(betas_tot)

    for i in range(len(M_tot)):
        if i >= len(betas_tot) or betas_tot[i] == constants.ev1 / constants.gam_rad**0.5:
            Omegas_tot.append(constants.ev2)
            continue

        M_f = float(M_f_tot[i]) if M_f_tot is not None else M_tot[i]
        ln_den_f = np.log(rho_form_rad[i])
        
        if ln_den_f <= ln_den_end_:
            Omegas_tot.append(constants.ev2)
            continue

        ln_den = np.linspace(ln_den_f, ln_den_end_, 10000)
        

        sol_try = solve_ivp(diff_rad, (ln_den_f, ln_den_end_), np.array([1., 0.]),
                            events=end_evol, t_eval=ln_den,
                            args=(M_f, betas_tot[i]), method="DOP853")

        if len(sol_try.t) == 0 or sol_try.t[-1] > ln_den_end_:
            sol_try_rel = solve_ivp(diff_rad_rel, (ln_den_f, ln_den_end_), np.array([1.]),
                                    t_eval=ln_den, args=(M_f, betas_tot[i]), method="DOP853")
            y_val = betas_tot[i] * sol_try_rel.y[0][-1] * (constants.M_pl_g / M_f)
            if M_f < 1e11 * constants.M_pl_g:
                Omegas_relic_pbbn.append(y_val)
                M_dm_rel_pbbn.append(M_tot[i])
        else:
            Delta_t = constants.t_pl * (M_f / constants.M_pl_g)**3
            y_val = betas_tot[i] * sol_try.y[0][-1] * (1. - sol_try.y[1][-1] / Delta_t)**(1./3)
            if M_f > 4.1e14:
                Omegas.append(y_val)
                M_dm.append(M_tot[i])
            elif M_f < 1e11 * constants.M_pl_g:
                Omegas_relic.append(y_val)
                M_dm_rel.append(M_tot[i])

        Omegas_tot.append(y_val)

    constraints.Omega_DM_tot = np.array(Omegas_tot)
    return np.array(M_n), np.array(betas_prim)/constants.gam_rad**0.5, np.array(M_relic), np.array(betas_relic_prim)/constants.gam_rad**0.5, np.array(Omegas_tot)


def Betas_BBN(M_tot, omega, M_f_tot=None, mu_tot=None):
    betas_bbn, M_bbn, Omegas_bbn_tot = [], [], []
    rho_form_rad = rho_f(M_tot, omega)
    ln_den_end_ = np.log(constants.rho_end)
    limite_planck = 1.5 * constants.M_pl_g

    for i in range(len(M_tot)):
        M_i = M_tot[i]
        M_f = float(M_f_tot[i]) if M_f_tot is not None else M_i
        mu  = float(mu_tot[i]) if mu_tot is not None else 1.0

        if M_f <= limite_planck or mu <= 0.0:
            constraints.betas_BBN_tot.append(constants.ev1)
            Omegas_bbn_tot.append(constants.ev2)
            continue

        ln_den_f = np.log(rho_form_rad[i])

        if constraints.data_mass[0] <= M_f < 2.5e13:
            M_bbn.append(M_i)
            abundancia = np.interp(M_f, constraints.data_mass, constraints.data_abundances)
            beta = (abundancia / constants.gam_rad**0.5) / mu
            betas_bbn.append(beta)

            if ln_den_f > ln_den_end_:
                ln_den = np.linspace(ln_den_f, ln_den_end_, 10000)
                sol_try = solve_ivp(diff_rad, (ln_den_f, ln_den_end_), np.array([1., 0.]),
                                    events=end_evol, t_eval=ln_den,
                                    args=(M_f, beta), method="DOP853")
                
                if len(sol_try.t) == 0 or (sol_try.t[-1] > ln_den_end_ and M_f < constraints.data_mass[76]):
                    sol_try_rel = solve_ivp(diff_rad_rel, (ln_den_f, ln_den_end_), np.array([1.]),
                                            t_eval=ln_den, args=(M_f, beta), method="DOP853")
                    y_val = beta * sol_try_rel.y[0][-1] * (constants.M_pl_g / M_f)
                else:
                    Delta_t = constants.t_pl * (M_f / constants.M_pl_g)**3
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


def Betas_SD(M_tot, omega, M_f_tot=None, mu_tot=None):
    betas_sd, M_sd, Omegas_sd_tot = [], [], []
    rho_form_rad = rho_f(M_tot, omega)
    ln_den_end_ = np.log(constants.rho_end)
    limite_planck = 1.5 * constants.M_pl_g

    for i in range(len(M_tot)):
        M_i = M_tot[i]
        M_f = float(M_f_tot[i]) if M_f_tot is not None else M_i
        mu  = float(mu_tot[i]) if mu_tot is not None else 1.0

        if M_f > limite_planck and 1e11 < M_f < 1e13:
            M_sd.append(M_i)
            beta = (1e-21 / constants.gam_rad**0.5) / mu
            betas_sd.append(beta)

            ln_den_f = np.log(rho_form_rad[i])
            if ln_den_f > ln_den_end_:
                ln_den = np.linspace(ln_den_f, ln_den_end_, 10000)
                sol_try = solve_ivp(diff_rad, (ln_den_f, ln_den_end_), np.array([1., 0.]),
                                    events=end_evol, t_eval=ln_den,
                                    args=(M_f, beta), method="DOP853")
                Delta_t = constants.t_pl * (M_f / constants.M_pl_g)**3
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


def Betas_CMB_AN(M_tot, omega, M_f_tot=None, mu_tot=None):
    betas_an, M_an, Omegas_an_tot = [], [], []
    rho_form_rad = rho_f(M_tot, omega)
    ln_den_end_ = np.log(constants.rho_end)
    limite_planck = 1.5 * constants.M_pl_g

    for i in range(len(M_tot)):
        M_i = M_tot[i]
        M_f = float(M_f_tot[i]) if M_f_tot is not None else M_i
        mu  = float(mu_tot[i]) if mu_tot is not None else 1.0

        if M_f > limite_planck and 2.5e13 < M_f < 2.4e14:
            M_an.append(M_i)
            beta = (3e-30 * (M_f / 1e13)**3.1 / constants.gam_rad**0.5) / mu
            betas_an.append(beta)

            ln_den_f = np.log(rho_form_rad[i])
            if ln_den_f > ln_den_end_:
                ln_den = np.linspace(ln_den_f, ln_den_end_, 10000)
                sol_try = solve_ivp(diff_rad, (ln_den_f, ln_den_end_), np.array([1., 0.]),
                                    events=end_evol, t_eval=ln_den,
                                    args=(M_f, beta), method="DOP853")
                Delta_t = constants.t_pl * (M_f / constants.M_pl_g)**3
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


def Betas_GRB(M_tot, omega, M_f_tot=None, mu_tot=None):
    betas_grb1, M_grb1, betas_grb2, M_grb2 = [], [], [], []
    Omegas_grb1, Omegas_grb2 = [], []
    rho_form_rad = rho_f(M_tot, omega)
    ln_den_end_ = np.log(constants.rho_end)
    limite_planck = 1.5 * constants.M_pl_g

    for i in range(len(M_tot)):
        M_i = M_tot[i]
        M_f = float(M_f_tot[i]) if M_f_tot is not None else M_i
        mu  = float(mu_tot[i]) if mu_tot is not None else 1.0

        ln_den_f = np.log(rho_form_rad[i])
        beta = constants.ev1
        y_val = constants.ev2

        if M_f > limite_planck and 3e13 < M_f < 4.1e14:
            M_grb1.append(M_i)
            beta = (5e-28 * (M_f / 4.1e14)**(-3.3) / constants.gam_rad**0.5) / mu
            betas_grb1.append(beta)

            if ln_den_f > ln_den_end_:
                ln_den = np.linspace(ln_den_f, ln_den_end_, 10000)
                sol_try = solve_ivp(diff_rad, (ln_den_f, ln_den_end_), np.array([1., 0.]),
                                    events=end_evol, t_eval=ln_den,
                                    args=(M_f, beta), method="DOP853")
                Delta_t = constants.t_pl * (M_f / constants.M_pl_g)**3
                if len(sol_try.t) > 0:
                    y_val = beta * sol_try.y[0][-1] * (1. - sol_try.y[1][-1] / Delta_t)**(1./3)
                else:
                    y_val = 0.0
                Omegas_grb1.append(y_val)

        elif M_f > limite_planck and 4.1e14 < M_f < 7e16:
            M_grb2.append(M_i)
            beta = (5e-26 * (M_f / 4.1e14)**3.9 / constants.gam_rad**0.5) / mu
            betas_grb2.append(beta)

            if ln_den_f > ln_den_end_:
                ln_den = np.linspace(ln_den_f, ln_den_end_, 10000)
                sol_try = solve_ivp(diff_rad, (ln_den_f, ln_den_end_), np.array([1., 0.]),
                                    events=end_evol, t_eval=ln_den,
                                    args=(M_f, beta), method="DOP853")
                Delta_t = constants.t_pl * (M_f / constants.M_pl_g)**3
                if len(sol_try.t) > 0:
                    y_val = beta * sol_try.y[0][-1] * (1. - sol_try.y[1][-1] / Delta_t)**(1./3)
                else:
                    y_val = 0.0
                Omegas_grb2.append(y_val)

        constraints.betas_GRB_tot.append(beta)
        constraints.Omega_GRB_tot.append(y_val)

    return (np.array(M_grb1), np.array(M_grb2), np.array(betas_grb1), np.array(betas_grb2),
            np.array(Omegas_grb1), np.array(Omegas_grb2))


def Betas_Reio(M_tot, omega, M_f_tot=None, mu_tot=None):
    betas_reio, M_reio = [], []
    rho_form_rad = rho_f(M_tot, omega)
    ln_den_end_ = np.log(constants.rho_end)
    limite_planck = 1.5 * constants.M_pl_g

    for i in range(len(M_tot)):
        M_i = M_tot[i]
        M_f = float(M_f_tot[i]) if M_f_tot is not None else M_i
        mu  = float(mu_tot[i]) if mu_tot is not None else 1.0

        ln_den_f = np.log(rho_form_rad[i])
        beta = constants.ev1
        y_val = constants.ev2

        if M_f > limite_planck and 1e15 < M_f < 1e17:
            M_reio.append(M_i)
            beta = (2.4e-26 * (M_f / 4.1e14)**4.3 / constants.gam_rad**0.5) / mu
            betas_reio.append(beta)

            if ln_den_f > ln_den_end_:
                ln_den = np.linspace(ln_den_f, ln_den_end_, 10000)
                sol_try = solve_ivp(diff_rad, (ln_den_f, ln_den_end_), np.array([1., 0.]),
                                    events=end_evol, t_eval=ln_den,
                                    args=(M_f, beta), method="DOP853")
                Delta_t = constants.t_pl * (M_f / constants.M_pl_g)**3
                if len(sol_try.t) > 0:
                    y_val = beta * sol_try.y[0][-1] * (1. - sol_try.y[1][-1] / Delta_t)**(1./3)
                else:
                    y_val = 0.0

        constraints.betas_Reio_tot.append(beta)
        constraints.Omega_Reio_tot.append(y_val)

    return np.array(M_reio), np.array(betas_reio), np.array(constraints.Omega_Reio_tot)


def Betas_LSP(M_tot, w, M_f_tot=None, mu_tot=None):
    betas_lsp, M_lsp = [], []
    rho_form_rad = rho_f(M_tot, w)
    ln_den_end_ = np.log(constants.rho_end)
    limite_planck = 1.5 * constants.M_pl_g

    for i in range(len(M_tot)):
        M_i = M_tot[i]
        M_f = float(M_f_tot[i]) if M_f_tot is not None else M_i
        mu  = float(mu_tot[i]) if mu_tot is not None else 1.0

        beta = constants.ev1
        y_val = constants.ev2

        if M_f > limite_planck and M_f < 1e11:
            M_lsp.append(M_i)
            beta = (1e-18 * (M_f / 1e11)**(-0.5) / constants.gam_rad**0.5) / mu
            betas_lsp.append(beta)

            ln_den_f = np.log(rho_form_rad[i])
            if ln_den_f > ln_den_end_:
                ln_den = np.linspace(ln_den_f, ln_den_end_, 10000)
                sol_try = solve_ivp(diff_rad, (ln_den_f, ln_den_end_), np.array([1., 0.]),
                                    events=end_evol, t_eval=ln_den,
                                    args=(M_f, beta), method="DOP853")
                
                if len(sol_try.t) == 0 or sol_try.t[-1] > ln_den_end_:
                    sol_try_rel = solve_ivp(diff_rad_rel, (ln_den_f, ln_den_end_), np.array([1.]),
                                            t_eval=ln_den, args=(M_f, beta), method="DOP853")
                    y_val = beta * sol_try_rel.y[0][-1] * (constants.M_pl_g / M_f)
                else:
                    Delta_t = constants.t_pl * (M_f / constants.M_pl_g)**3
                    y_val = beta * sol_try.y[0][-1] * (1. - sol_try.y[1][-1] / Delta_t)**(1./3)

        constraints.betas_LSP_tot.append(beta)
        constraints.Omega_LSP_tot.append(y_val)

    return np.array(M_lsp), np.array(betas_lsp), np.array(constraints.Omega_LSP_tot)


# ---------------------------------------------------------------------------
# Funciones de envoltura
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
