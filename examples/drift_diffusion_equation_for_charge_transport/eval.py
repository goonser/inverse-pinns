import os

import ml_collections

import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

from jaxpi.utils import restore_checkpoint

import models
from utils import get_dataset


def evaluate(config: ml_collections.ConfigDict, workdir: str, step=""):
   # Problem setup
    n_t = 200  # number of time steps
    n_x = 10_000  # number of spatial points

    true_mu = config.setting.true_mu

    # Get  dataset
    _, t_star, x_star, u_exact_fn = get_dataset(n_t, n_x, true_mu, config)

    # Selected time steps to evaluate, every 0.001 seconds
    t_star = jnp.linspace(0, 0.006, 7)

    # Restore model
    model = models.InverseDriftDiffusion(config, t_star, x_star, u_exact_fn)

    ckpt_path = os.path.join(workdir, "ckpt", config.wandb.name)
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params

    obs_t_test = jnp.linspace(0, 0.006, 7)
    obs_x_test = jax.random.uniform(jax.random.PRNGKey(0), (100,), minval=x_star[0], maxval=x_star[-1])
    obs_u_exact = u_exact_fn(obs_t_test, obs_x_test)
    if config.setting.noise_level is not None:
        obs_u = model.add_noise_to_data(obs_u_exact)
    else:
        obs_u = obs_u_exact

    u_pred = model.u_pred_fn(params, model.t_star, model.x_star)
    
    # Scaling back with n_inj
    u_pred *= model.n_inj_scale
    obs_u *= model.n_inj_scale

    # Compute L2 error
    print('Max predicted n:' , jnp.max(u_pred))
    print('Min predicted n:' , jnp.min(u_pred))
   
    
    # Plot results
    fig = plt.figure()
    if config.setting.noise_level is not None:
        plt.scatter(obs_x_test, obs_u[0,:], c="blue", label='Observations')
        plt.scatter(obs_x_test, obs_u[1,:], c="orange", label='Observations')
        plt.scatter(obs_x_test, obs_u[2,:], c="green", label='Observations')
        plt.scatter(obs_x_test, obs_u[3,:], c="red", label='Observations')
        plt.scatter(obs_x_test, obs_u[4,:], c="purple", label='Observations')
        plt.scatter(obs_x_test, obs_u[5,:], c="brown", label='Observations')
        plt.scatter(obs_x_test, obs_u[6,:], c="pink", label='Observations')
    plt.plot(x_star, u_pred[0,:], label='t=0.000')
    plt.plot(x_star, u_pred[1,:], label='t=0.001')
    plt.plot(x_star, u_pred[2,:], label='t=0.002')
    plt.plot(x_star, u_pred[3,:], label='t=0.003')
    plt.plot(x_star, u_pred[4,:], label='t=0.004')
    plt.plot(x_star, u_pred[5,:], label='t=0.005')
    plt.plot(x_star, u_pred[6,:], label='t=0.006')
    plt.grid()
    plt.xlabel("x [m]")
    plt.ylabel("Charge density n(x) [#/m3]")
    plt.title("Charge Density over x for different timesteps")
    plt.legend()
    plt.tight_layout()

    # Save the figure
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    fig_path = os.path.join(save_dir, f"Inverse_drift_diffusion_{step}.png")
    fig.savefig(fig_path, bbox_inches="tight", dpi=800)
    
    # --- final result prints ---
    print('\n--------- SUMMARY ---------\n')
    # print the predicted & final rho values
    mu_pred = jnp.exp(model.state.params['params']['mu_param'][0]) 
    mu_ref = config.setting.true_mu
    rel_error = (mu_pred-mu_ref)/mu_ref

    print(f'Predicted Mu:  {mu_pred}')
    print(f'True Mu:       {mu_ref}')
    print(f'Relative error: {rel_error:.1%}\n')
    print('---------------------------\n')

    

    if step == '':
        # save plot information as csv for later use        
        TT, XX = jnp.meshgrid(t_star, x_star, indexing='ij')

        u_pred = jax.device_get(u_pred)

        TT = jax.device_get(TT)
        XX = jax.device_get(XX)

        u_pred = u_pred.reshape(-1)
        u_ref = u_ref.reshape(-1)
        TT = TT.reshape(-1)
        XX = XX.reshape(-1)
        combined_array = np.column_stack((TT, XX, u_pred, u_ref))
        csv_file_path = "Inverse_Drift_Diffusion.csv"
        header_names = ['t_star', 'x_star', 'n_pred', 'n_ref']
        np.savetxt(csv_file_path, combined_array, delimiter=",", header=",".join(header_names), comments='')