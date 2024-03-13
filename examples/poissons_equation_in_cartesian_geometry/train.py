import os
import time

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

import ml_collections

# from absl import logging
import wandb

from jaxpi.samplers import BaseSampler, UniformSampler, init_sampler
from jaxpi.logging import Logger
from jaxpi.utils import save_checkpoint

import models
from utils import get_dataset, get_reference_dataset

from abc import ABC, abstractmethod
from functools import partial

import jax.numpy as jnp
from jax import random, pmap, local_device_count
from eval import evaluate

from torch.utils.data import Dataset

import numpy as np
import matplotlib.pyplot as plt

class OneDimensionalUniformSampler(BaseSampler):
    def __init__(self, dom, batch_size, rng_key=random.PRNGKey(1234)):
        super().__init__(batch_size, rng_key)
        self.dom = dom
        self.dim = 1

    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key):
        "Generates data containing batch_size samples"
        batch = random.uniform(
            key,
            shape=(self.batch_size, self.dim),
            minval=self.dom[0],
            maxval=self.dom[1],
        )

        return batch



def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    logger = Logger()
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name)

    # Problem setup
    n_x = config.setting.n_x    # number of spatial points (old: 128 TODO: INCREASE A LOT?)
    n_scale = config.setting.n_scale

    # Get  dataset
    _, x_star = get_dataset(n_x=n_x)
    _, _, u_ref = get_reference_dataset(config, config.eval.field_file_path, config.eval.potential_file_path)

    # Initial condition (TODO: Looks as though this is for t = 0 in their solution, should we have for x = 0)?
    u0 = config.setting.u0
    u1 = config.setting.u1


    # Define domain
    x0 = x_star[0]
    x1 = x_star[-1]

    dom = jnp.array([x0, x1]) 

    # Initialize model
    model = models.InversePoisson(config, u0, u1, x_star, n_scale)
    
    # Initialize sampler
    sampler = init_sampler(model, config)
    res_sampler = iter(sampler)

    evaluator = models.InversePoissonEvaluator(config, model)
    # jit warm up
    print("Waiting for JIT...")
    for step in range(config.training.max_steps):
        
        start_time = time.time()
    
        # Update RAD points
        if config.sampler.sampler_name != "random":
            if step % config.sampler.resample_every_steps == 0 and step != 0:
                
                if config.sampler.sampler_name == "rad-cosine": #and step!= config.sampler.resample_every_steps: 
                    #jax.debug.print("Resampling with rad-cosine and passign prev sampler")
                    sampler = init_sampler(model, config, prev = sampler)    
                else:
                    sampler = init_sampler(model, config)

                res_sampler = iter(sampler)
                
                if config.sampler.plot_rad == True:
                    sampler.plot(workdir, step, config.wandb.name)
                

        batch = next(res_sampler)
        
        if config.sampler.plot_batch == True:
            # plot histogram of new batch
            fig = plt.figure(figsize=(8, 8))
            plt.xlabel('Radius [m]')
            plt.ylabel('Count')
            plt.title('Batch histogram')
            plt.hist(batch.flatten(), bins=50, label='Sampled data', color='blue')
            plt.grid()
            plt.legend()
            plt.tight_layout()
            # Save the figure
            save_dir = os.path.join(workdir, "figures", config.wandb.name)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            fig_path = os.path.join(save_dir, f"batch_hist_{step}.png")
            fig.savefig(fig_path, bbox_inches="tight", dpi=800)
            plt.close(fig)


        model.state = model.step(model.state, batch)

        # Update weights
        if config.weighting.scheme in ["grad_norm", "ntk"]:
            if step % config.weighting.update_every_steps == 0:
                model.state = model.update_weights(model.state, batch)

        # Log training metrics, only use host 0 to record results
        if jax.process_index() == 0:
            if step % config.logging.log_every_steps == 0:
                # Get the first replica of the state and batch
                state = jax.device_get(tree_map(lambda x: x[0], model.state))
                batch = jax.device_get(tree_map(lambda x: x[0], batch))
                log_dict = evaluator(state, batch, u_ref)
                if config.arch.arch_name == "InverseMlpCaseChargeProfile":
                    n_scale = (10  ** state.params['params']['n_scale_param'][0]) 
                    log_dict['n_scale_param'] = n_scale
                wandb.log(log_dict, step)
                end_time = time.time()

                logger.log_iter(step, start_time, end_time, log_dict)

        # Saving
        if config.saving.save_every_steps is not None:
            if (step + 1) % config.saving.save_every_steps == 0 or (step + 1) == config.training.max_steps:
                path = os.path.join(workdir, "ckpt", config.wandb.name)
                save_checkpoint(model.state, path, keep=config.saving.num_keep_ckpts)
                if config.saving.plot == True:
                    evaluate(config, workdir, step + 1)
    return model