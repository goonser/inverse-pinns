import os
import time

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

import ml_collections

# from absl import logging
import wandb

from jaxpi.samplers import UniformSampler
from jaxpi.logging import Logger
from jaxpi.utils import save_checkpoint
from eval import evaluate
import models
from utils import get_dataset


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    logger = Logger()
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name)

    # Problem setup
    n_t = 200  # number of time steps 
    n_x = 128  # number of spatial points

    true_mu = config.setting.true_mu

    # Get  dataset
    u_ref, t_star, x_star, u_exact_fn = get_dataset(n_t, n_x, true_mu, config)

    # Define domain
    t0 = t_star[0]
    t1 = t_star[-1]

    x0 = x_star[0]
    x1 = x_star[-1]

    dom = jnp.array([[t0, t1], [x0, x1]])

    # Initialize model
    model = models.InverseDriftDiffusion(config, t_star, x_star, u_exact_fn)
    # Initialize residual sampler
    res_sampler = iter(UniformSampler(dom, config.training.batch_size_per_device))

    evaluator = models.InverseDriftDiffusionEvalutor(config, model)
    # jit warm up
    print("Waiting for JIT...")
    for step in range(config.training.max_steps):
        start_time = time.time()

        batch = next(res_sampler)

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
                mu = state.params['params']['mu_param'][0]
                log_dict['mu_param'] = jnp.exp(mu)

                wandb.log(log_dict, step)
                end_time = time.time()

                logger.log_iter(step, start_time, end_time, log_dict)

        # Saving
        if config.saving.save_every_steps is not None:
            if (step + 1) % config.saving.save_every_steps == 0 or (
                step + 1
            ) == config.training.max_steps:
                path = os.path.join(workdir, "ckpt", config.wandb.name)
                save_checkpoint(model.state, path, keep=config.saving.num_keep_ckpts)
                if config.saving.plot == True:
                    evaluate(config, workdir, step + 1)

    return model
