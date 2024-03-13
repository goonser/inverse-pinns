import jax.numpy as jnp
from jax import vmap
import jax


def get_dataset(r_0, r_1, n_r, true_offset):
    r_star = jnp.linspace(r_0, r_1, n_r)

    r0 = r_star[0]
    r1 = r_star[-1]
   
    true_r0 = r0 + true_offset
    true_r1 = r1 + true_offset

    eps = 8.85e-12
    rho = 5e-10

    C_1 = ((4*eps*jnp.log(true_r1) + rho * true_r0**2 * jnp.log(true_r1) - rho * true_r1**2 * jnp.log(true_r0)) /
       (4 * eps * (-jnp.log(true_r0) + jnp.log(true_r1))))
    
    C_2 = (-4 * eps - 5e-10*true_r0**2 + 5e-10 * true_r1**2) / (4 * eps * (-jnp.log(true_r0) + jnp.log(true_r1)))

    
    u_exact_fn = lambda r: C_1 + C_2 * jnp.log(r+true_offset) - (rho * (r+true_offset)**2) / (4 * eps)
    u_exact = vmap(u_exact_fn)(r_star)
    
    return u_exact, r_star

def get_observations(r0, r1, true_offset, config):
    
   true_r0 = r0 + true_offset
   true_r1 = r1 + true_offset
   eps = 8.85e-12
   rho = 5e-10
   n_obs = config.setting.n_obs
   
   C_1 = ((4*eps*jnp.log(true_r1) + rho * true_r0**2 * jnp.log(true_r1) - rho * true_r1**2 * jnp.log(true_r0)) /
      (4 * eps * (-jnp.log(true_r0) + jnp.log(true_r1))))
   
   C_2 = (-4 * eps - 5e-10*true_r0**2 + 5e-10 * true_r1**2) / (4 * eps * (-jnp.log(true_r0) + jnp.log(true_r1)))
   
   obs_r = jax.random.uniform(jax.random.PRNGKey(config.seed), (n_obs,), minval=r0, maxval=r1)
   
   u_exact_fn = lambda r: C_1 + C_2 * jnp.log(r+true_offset) - (rho * (r+true_offset)**2) / (4 * eps)
   u_exact = vmap(u_exact_fn)(obs_r)

   return obs_r, u_exact

def get_noisy_observations(r0, r1, true_offset, config):
   relative_noise = config.setting.guassian_noise_perc
   r_star, clean_data = get_observations(r0, r1, true_offset, config)
   noise_std = clean_data * relative_noise  # Calculate noise standard deviation
   key = jax.random.PRNGKey(config.seed)
   noise = noise_std * jax.random.normal(key, shape=r_star.shape)  # Generate Gaussian noise
   noisy_data = clean_data + noise
   return r_star, noisy_data

