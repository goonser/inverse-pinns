import jax.numpy as jnp
from jax import vmap


def get_dataset(r_0, r_1, n_r, true_rho, u0):
   r_star = jnp.linspace(r_0, r_1, n_r)

   r0 = r_star[0]
   r1 = r_star[-1]

   eps = 8.85e-12

   ln = jnp.log(r0 / r1)
   C_2 = u0 / ln - true_rho * (r1**2 - r0**2) / (4 * eps * ln)
   C_1 = true_rho * r1**2 / (4 * eps) - C_2 * jnp.log(r1)
   
   u_exact_fn = lambda r: C_1 + C_2 * jnp.log(r) - (true_rho * r**2) / (4 * eps)
   u_exact = vmap(u_exact_fn)(r_star)
    
   return u_exact, r_star