import jax.numpy as jnp
from jax import vmap

def get_dataset(n_t, n_x, true_mu, config):
    n_0 = config.setting.n_0 / config.setting.n_inj
    n_inj = config.setting.n_inj / config.setting.n_inj
    T = 0.01 # per case 2
    L = 1 # per case 2
    E_ext = config.setting.E_ext # per case 2
    t_star = jnp.linspace(0, T, n_t)
    x_star = jnp.linspace(0, L, n_x)

    # Analytical solution
    def analytical_solution(t, x):
        condition = x <= E_ext * true_mu * t
        result = jnp.where(condition, n_inj, n_0)
        return result

    # Vecorized format of analytical solution
    u_exact_fn = vmap(vmap(analytical_solution, (None, 0)), (0, None))

    # Analytical solution of training data 
    u_exact = u_exact_fn(t_star, x_star)

    return u_exact, t_star, x_star, u_exact_fn
