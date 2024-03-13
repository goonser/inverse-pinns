import jax.numpy as jnp
from jax import vmap
import numpy as np
import jax
import io
import pandas as pd

def get_dataset(n_x):
    L = 1 # per case 2
    x_star = jnp.linspace(0, L, n_x)
    # Dummy function to replace analytical solution.
    u_exact_fn = lambda x: 0
    u_exact = vmap(u_exact_fn)(x_star)
    
    return u_exact, x_star

def get_observations(config):
    n_obs = config.setting.n_obs
    obs_file=config.setting.obs_file
    # open numpy array from file called obs.dat
    obs = np.loadtxt(obs_file)
    obs = jnp.array(obs)

    obs_x = obs[:,0]
    obs_u = obs[:,1]

    #selct n_obs random indices
    key = jax.random.PRNGKey(42) 
    idx = jax.random.randint(key, minval=0, maxval=len(obs_x), shape=(n_obs,))
    obs_x = obs_x[idx]
    obs_u = obs_u[idx]

    return obs_x, obs_u

def get_noisy_observations(config):
    noise_level = config.setting.guassian_noise_perc
    obs_x, clean_data = get_observations(config)
    noise_std = clean_data * noise_level  # Calculate noise standard deviation
    key = jax.random.PRNGKey(config.seed)
    noise = noise_std * jax.random.normal(key, shape=obs_x.shape)  # Generate Gaussian noise
    noisy_data = clean_data + noise
    return obs_x, noisy_data

def get_reference_dataset(config, e_path, u_path):
    # Load data
    # Read the file, skipping lines starting with "%"
    
    # Reading comsom data for E
    with open(e_path, 'r') as file:
        lines = [line for line in file if not line.startswith('%')]

    # Use StringIO to create a virtual file-like object for pandas to read from
    virtual_file = io.StringIO(''.join(lines))

    # Read data into pandas DataFrame
    df_E = pd.read_csv(virtual_file, delim_whitespace=True, names=['x', 'E'])

    x_ref = df_E['x'].values
    E_ref = df_E['E'].values


    # Reading comsom data for U
    with open(u_path, 'r') as file:
        lines = [line for line in file if not line.startswith('%')]

    # Use StringIO to create a virtual file-like object for pandas to read from
    virtual_file = io.StringIO(''.join(lines))

    # Read data into pandas DataFrame
    df_U = pd.read_csv(virtual_file, delim_whitespace=True, names=['x', 'U'])

    x_ref = df_U['x'].values
    u_ref = df_U['U'].values

    return x_ref, E_ref, u_ref