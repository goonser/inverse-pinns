import os

import ml_collections
import pandas as pd
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

from jaxpi.utils import restore_checkpoint
import models
from utils import get_dataset, get_observations, get_reference_dataset


def evaluate(config: ml_collections.ConfigDict, workdir: str, step=""):
    
    # Problem setup
    n_x = config.setting.n_x    # number of spatial points
    n_scale = config.setting.n_scale

    # Get  dataset
    _, x_star = get_dataset(n_x=n_x)

    # Initial condition (TODO: Looks as though this is for t = 0 in their solution, should we have for x = 0)?
    u_scale = config.setting.u0
    u0 = config.setting.u0
    u1 = config.setting.u1

    # Define domain
    x0 = x_star[0]
    x1 = x_star[-1]

    dom = jnp.array([x0, x1]) 
    # Restore model
    model = models.InversePoisson(config, u0, u1, x_star, n_scale)
    ckpt_path = os.path.join(workdir, "ckpt", config.wandb.name)
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params

    u_pred = model.u_pred_fn(params, model.x_star)
    u_pred *= u_scale
    e_pred_fn = jax.vmap(lambda params, x: -jax.grad(model.u_net, argnums=1)(params, x), (None, 0))

    n_pred = model.n_pred_fn(params, model.x_star)
    
    if config.arch.arch_name == "InverseMlpCaseChargeProfile":
        a = params['params']['n_scale_param'][0]
        n_pred *= a
    else:    
        n_pred *= n_scale   # TODO: check if correct


    #du_dr = jax.grad(model.u_pred_fn) # e = d/dr U
    e_pred = e_pred_fn(params, model.x_star)
    e_pred *= u0
    
    n_values = n_scale * jax.vmap(model.heaviside)(x_star)

    r_pred = model.r_pred_fn(params, model.x_star)**2


    # Create a Matplotlib figure and axis
    fig = plt.figure(figsize=(8, 14))
    plt.subplot(4,1,1)
    plt.xlabel('Distance [m]')
    plt.ylabel('Charge density n(x)')
    plt.title('Charge density')
    plt.plot(x_star, n_pred, label='pred (x)', color='blue')
    plt.plot(x_star, n_values, linestyle='--', label='true n(x)', color='red')
    plt.legend()
    plt.tight_layout()    
    plt.xlim(x_star[0], x_star[-1])
    plt.grid()


    plt.subplot(4, 1, 2)
    plt.xlabel('Distance [m]')
    plt.ylabel('Potential V(x)')
    plt.title('Potential')

    # Plot the prediction
    plt.plot(x_star, u_pred, label='Predicted V(x)', color='blue')

    # Plot original V(x)
    plt.plot(x_star, 1e6*(-x_star + 1), linestyle='--', label='Original V(x)', color='red') 
    plt.grid()
    plt.legend()
    plt.tight_layout()    
    plt.xlim(x_star[0], x_star[-1])

    # plot electrical field
    plt.subplot(4, 1, 3)

    plt.xlabel('Distance [m]')
    plt.ylabel('Electric field [V/m]')
    plt.title('Electrical field')

    # Plot the prediction values as a solid line
    plt.plot(x_star, e_pred, color='blue')
    plt.grid()
    plt.xlim(x_star[0], x_star[-1])
    plt.tight_layout()    

    plt.subplot(4, 1, 4)
    plt.scatter(x_star, r_pred, color='blue', marker='o', s=1, alpha=0.5)  # Use marker='o' for circular markers, adjust 's' for marker size
    plt.yscale('log')
    plt.plot(x_star, jnp.full_like(x_star, jnp.mean(r_pred)), label='Mean', linestyle='--', color='red')

    plt.xlabel('Distance [m]')
    plt.ylabel('Squared Residual Loss')
    plt.title('Squared Residual Loss')
    plt.legend()
    plt.grid()
    plt.xlim(x_star[0], x_star[-1])
    plt.tight_layout()

    # Save the figure
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    fig_path = os.path.join(save_dir, f"laplace_2.5_{step}.png")
    fig.savefig(fig_path, bbox_inches="tight", dpi=800)
    plt.close(fig)
    
    # Save COMSOL comparison
    file_paths = [config.eval.potential_file_path, config.eval.field_file_path]
    has_ref_data = all(path is not None for path in file_paths)
    print()
    if has_ref_data:
        x_ref_star, e_ref, u_ref = get_reference_dataset(config, config.eval.field_file_path, config.eval.potential_file_path)
        
        # get new pred data
        u_ref_pred = model.u_pred_fn(params, x_ref_star)
        u_ref_pred *= config.setting.u0

        n_pred = model.n_pred_fn(params, x_ref_star)
        n_pred *= model.n_scale

        e_pred_fn = jax.vmap(lambda params, x: -jax.grad(model.u_net, argnums=1)(params, x), (None, 0))

        e_pred = e_pred_fn(params, x_ref_star)
        e_pred *= config.setting.u0

        n_values = n_scale * jax.vmap(model.heaviside)(x_ref_star)
        
        # Plot n results
        fig = plt.figure(figsize=(8, 12))
        plt.subplot(3, 1, 1)
        
        plt.plot(x_ref_star, n_pred, label='PINN', color='blue')
        plt.plot(x_ref_star, n_values, label='True', color='red')
        plt.grid()
        plt.xlabel("Distance [m]")
        plt.ylabel(r'Charge density [$\# / \mathrm{m}^3}$]')
        plt.title("Charge density predictions using PINN and COMSOL")
        plt.legend()
        plt.tight_layout()
        plt.xlim(x_ref_star[0], x_ref_star[-1])

        # plot Potential field
        plt.subplot(3, 1, 2)
        
        plt.plot(x_ref_star, u_ref_pred, label='PINN', color='blue')
        plt.plot(x_ref_star, u_ref, label='COMSOL', color='red', linestyle='--')
        plt.xlabel("Distance [m]")
        plt.ylabel("Potential [V]")
        plt.title("Potential predictions using PINN and COMSOL")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.xlim(x_ref_star[0], x_ref_star[-1])

        # plot electrical field
        plt.subplot(3, 1, 3) 
        plt.plot(x_ref_star, e_pred, label='PINN', color='blue')
        plt.plot(x_ref_star, e_ref, label='COMSOL', color='red', linestyle='--')
        plt.xlabel("Distance [m]")
        plt.ylabel("Electric field [V/m]")
        plt.title("Electric field predictions using PINN and COMSOL")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.xlim(x_ref_star[0], x_ref_star[-1])

        # save image
        fig_path = os.path.join(save_dir, f"comp_inv_case_1_5_{step}.png")
        fig.savefig(fig_path, bbox_inches="tight", dpi=800)
        plt.close(fig)

    if step == "":
        # save data to csv file
        csv_path = os.path.join(save_dir, f"inverse_case1p5_plot_data_{step}.csv")
        df = pd.DataFrame({'x': x_ref_star, 
                           'n_pred': n_pred, 
                           'n_true': n_values, 
                           'u_pred': u_ref_pred, 
                           'u_true_comsol': u_ref, 
                           'e_pred': e_pred, 
                           'e_true_comsol': e_ref,})
        df.to_csv(csv_path, index=False)


        # plot observations
        if config.setting.guassian_noise_perc is not None:
            # get clean data
            obs_x, obs_u = get_observations(config)

            fig = plt.figure(figsize=(8, 6))
            plt.xlabel('Radius [m]')
            plt.ylabel('Potential V')
            plt.title(f'Noisy observation data (noise level {config.setting.guassian_noise_perc:.0%})')
            plt.scatter(model.obs_x, model.obs_u , label='Observations', color='blue')
            plt.scatter(obs_x, obs_u, label='Analytical Solution', color='red')
            plt.grid()
            plt.xlim(x_star[0], x_star[-1])
            plt.legend()
            plt.tight_layout()

            fig_path = os.path.join(save_dir, "Observations.png")
            fig.savefig(fig_path, bbox_inches="tight", dpi=800)
            plt.close(fig)