import os

import ml_collections
from math import floor, log
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

from jaxpi.utils import restore_checkpoint
import models
from utils import get_dataset
import pandas as pd


def evaluate(config: ml_collections.ConfigDict, workdir: str, step=""):
    
    eps = 8.85e-12
    true_rho = config.setting.true_rho
    rho_scale = config.setting.rho_scale

    # Problem setup
    r_0 = config.setting.r_0  # inner radius
    r_1 = config.setting.r_1  # outer radius
    n_r = 10_000

    u0 = config.setting.u0 
    u1 = config.setting.u1
    
    # Get  dataset
    u_ref, r_star = get_dataset(r_0, r_1, n_r, true_rho, u0)
    
    ln = jnp.log(r_0 / r_1)
    C_2 = u0 / ln - true_rho * (r_1**2 - r_0**2) / (4 * eps * ln)
    C_1 = true_rho * r_1**2 / (4 * eps) - C_2 * jnp.log(r_1)

    # Restore model
    model = models.InversePoisson(config, u0, u1, r_star, true_rho, rho_scale)
    ckpt_path = os.path.join(workdir, "ckpt", config.wandb.name)
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params

    u_pred = model.u_pred_fn(params, model.r_star)
    e_pred_fn = jax.vmap(lambda params, r: -jax.grad(model.u_net, argnums=1)(params, r), (None, 0))

    #du_dr = jax.grad(model.u_pred_fn) # e = d/dr U
    e_pred = e_pred_fn(params, model.r_star)
    e_ref = -(C_2 / r_star - true_rho * r_star / (2 * eps)) # analytical solution for e
    # Convert them to NumPy arrays for Matplotlib
    r_star_np = jnp.array(r_star)
    u_pred_np = jnp.array(u_pred)
    u_ref_np = jnp.array(u_ref)

    # Create a Matplotlib figure and axis
    fig = plt.figure(figsize=(14, 8))
    plt.subplot(2, 2, 1)
    plt.xlabel('Radius [m]')
    plt.ylabel('Potential V(r)')
    plt.title('Predicted and Analyical Potential')

    # Plot the prediction values as a solid line
    plt.plot(r_star_np, u_pred_np, label='Prediction', color='blue')

    # Plot the analytical solution as a dashed line
    plt.plot(r_star_np, u_ref_np, linestyle='--', label='Analytical Solution', color='red')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    
    # Set x-axis limits to [r_star[0], r_star[-1]]
    plt.xlim(r_star_np[0], r_star_np[-1])

    # plot absolute errors 
    plt.subplot(2, 2, 3)
    plt.xlabel('Radius [m]')
    plt.ylabel('Potenial [V]')
    plt.title('Absolute Potential Error')

    plt.plot(r_star_np, jnp.abs(u_pred_np - u_ref_np) , label='Absolute error', color='red')
    plt.grid()
    plt.xlim(r_star_np[0], r_star_np[-1])
    plt.tight_layout()

    # plot electrical field and rho
    plt.subplot(2, 2, 2)

    plt.xlabel('Radius [m]')
    plt.ylabel('Electric field [V/m]')
    plt.title('Predicted and Analytical Electrical field')

    # Plot the prediction values as a solid line
    plt.plot(r_star_np, e_pred, label='Prediction', color='blue')
    # Plot the analytical solution as a dashed line
    plt.plot(r_star_np, e_ref, linestyle='--', label='Analytical Solution', color='red')
    plt.grid()
    plt.legend()
    plt.xlim(r_star_np[0], r_star_np[-1])
    plt.tight_layout()

    #plot absolute field errors 
    plt.subplot(2, 2, 4)
    plt.xlabel('Radius [m]')
    plt.ylabel('Electrical field [V/m]')
    plt.title('Absolute Electrical field')

    plt.plot(r_star_np, jnp.abs(e_pred - e_ref) , label='Absolute error', color='red')
    plt.grid()
    plt.xlim(r_star_np[0], r_star_np[-1])
    plt.tight_layout()

    
    # Save the figure
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    #fig_path = os.path.join(save_dir, "inverse_poisson.pdf")
    #fig.savefig(fig_path, bbox_inches="tight", dpi=800)
    # save as png for easy copy
    fig_path = os.path.join(save_dir, f"inverse_poisson_{step}.png")
    fig.savefig(fig_path, bbox_inches="tight", dpi=800)

    # To view in colab, run, run: 
    # from IPython.display import Image
    # Image(filename='/content/jaxpi/examples/inverse_case_1/figures/current_sota/inverse_poisson.png')

    # --- final result prints ---
    print('\n--------- SUMMARY ---------\n')
    # print L2 error
    l2_error = model.compute_l2_error(params, u_ref)
    print("L2 error:       {:.3e}".format(l2_error))  

    # print the predicted & final rho values
    rho_pred = model.state.params['params']['rho_param'][0] * config.setting.rho_scale 
    rho_ref = config.setting.true_rho
    rel_error = (rho_pred-rho_ref)/rho_ref
    pred_scale = abs(floor(log(rho_pred, 10)))
    rho_pred = round(rho_pred, pred_scale + 3)
    print(f'Predicted Rho:  {rho_pred}')
    print(f'True Rho:       {rho_ref}')
    print(f'Relative error: {rel_error:.1%}\n')
    if config.setting.guassian_noise_perc is not None:
        print(f'Noise level:    {config.setting.guassian_noise_perc}')
    print('---------------------------\n')

    df = pd.DataFrame({
        'radius': r_star,
        'predicted potetial': u_pred,
        'analytical potential': u_ref,
        'predicted field': e_pred,
        'analytical field': e_ref})
    df.to_csv('inverse_laplace_plot_data.csv', index=False)

    # plot observations
    if config.setting.guassian_noise_perc is not None:
        fig = plt.figure(figsize=(8, 6))
        plt.xlabel('Radius [m]')
        plt.ylabel('Potential V')
        plt.title(f'Noisy observation data (noise level {config.setting.guassian_noise_perc:.0%})')
        plt.scatter(model.obs_r, model.obs_u , label='Observations', color='blue')
        plt.plot(r_star_np, u_ref_np, label='Analytical Solution', color='red')
        plt.grid()
        plt.xlim(r_star_np[0], r_star_np[-1])
        plt.legend()
        plt.tight_layout()

        fig_path = os.path.join(save_dir, "Observations.png")
        fig.savefig(fig_path, bbox_inches="tight", dpi=800)


    
 
