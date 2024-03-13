import os

import ml_collections
from math import floor, log
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np
from jaxpi.utils import restore_checkpoint
import models
from utils import get_dataset



def evaluate(config: ml_collections.ConfigDict, workdir: str, step=''):
    
    eps = 8.85e-12
    rho = 5e-10

    true_offset = config.setting.true_offset

    # Problem setup
    r_0 = config.setting.r_0  # inner radius
    r_1 = config.setting.r_1  # outer radius
    n_r = 10_000
    
    true_r_0 = r_0 + true_offset
    true_r_1 = r_1 + true_offset

    # Get  dataset
    u_ref, r_star = get_dataset(r_0, r_1, n_r, true_offset)

    # Initial condition (TODO: Looks as though this is for t = 0 in their solution, should we have for x = 0)?
    u0 = 1 #u_ref[0]
    u1 = 0 #u_ref[-1]
    
    # Constants for analytical solution
    C_1 = ((4*eps*jnp.log(true_r_1) + rho * r_0**2 * jnp.log(true_r_1) - rho * true_r_1**2 * jnp.log(true_r_0)) /
       (4 * eps * (-jnp.log(true_r_0) + jnp.log(true_r_1))))
    
    C_2 = (-4 * eps - rho*true_r_0**2 + rho * true_r_1**2) / (4 * eps * (-jnp.log(true_r_0) + jnp.log(true_r_1)))

    # Restore model
    model = models.InversePoisson(config, u0, u1, r_star, true_offset)
    ckpt_path = os.path.join(workdir, "ckpt", config.wandb.name)
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params

    u_pred = model.u_pred_fn(params, model.r_star)
    e_pred_fn = jax.vmap(lambda params, r: -jax.grad(model.u_net, argnums=1)(params, r), (None, 0))

    #du_dr = jax.grad(model.u_pred_fn) # e = d/dr U
    e_pred = e_pred_fn(params, model.r_star)
    e_ref = -(C_2 / (r_star + true_offset) - rho * (r_star+true_offset) / (2 * eps)) # analytical solution for e

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

    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # save as png for easy copy
    fig_path = os.path.join(save_dir, f"inverse_poisson_{step}.png")
    fig.savefig(fig_path, bbox_inches="tight", dpi=800)

    # To view in colab, run, run: 
    # from IPython.display import Image
    # Image(filename='/content/jaxpi/examples/inverse_case_1/figures/current_sota/inverse_poisson.png')

    # --- final result prints ---
    if step == '':
        print('\n--------- SUMMARY ---------\n')
        # print L2 error
        l2_error = model.compute_l2_error(params, u_ref)
        print("L2 error:       {:.3e}".format(l2_error))  

        # print the predicted & final rho values
        offset_pred = jnp.exp(model.state.params['params']['offset_param'][0]) 
        offset_ref = config.setting.true_offset
        rel_error = (offset_pred-offset_ref)/offset_ref
        #pred_scale = abs(floor(log(rho_pred, 10)))
        #rho_pred = round(rho_pred, pred_scale + 3)
        print(f'Predicted Offset:  {offset_pred}')
        print(f'True Offset:       {offset_ref}')
        print(f'Relative error: {rel_error:.1%}\n')
        print('---------------------------\n')

        e_pred_np, e_ref_np = jnp.array(e_pred), jnp.array(e_ref)
        combined_array = np.column_stack((r_star_np, u_pred_np, u_ref_np, e_pred_np, e_ref_np))
        csv_file_path = "inverse_case_1_geom.csv"
        header_names = ['r_star', 'u_pred', 'u_ref', 'e_pred', 'e_ref']
        np.savetxt(csv_file_path, combined_array, delimiter=",", header=",".join(header_names), comments='')




    
 
