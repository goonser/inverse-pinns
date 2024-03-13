import os

from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, grad, tree_map
from jax.tree_util import tree_map
from jax.flatten_util import ravel_pytree

import flax 
from flax.training import checkpoints


def flatten_pytree(pytree):
    return ravel_pytree(pytree)[0]


@partial(jit, static_argnums=(0,))
def jacobian_fn(apply_fn, params, *args):
    # apply_fn needs to be a scalar function
    J = grad(apply_fn, argnums=0)(params, *args)
    J, _ = ravel_pytree(J)
    return J


@partial(jit, static_argnums=(0,))
def ntk_fn(apply_fn, params, *args):
    # apply_fn needs to be a scalar function
    J = jacobian_fn(apply_fn, params, *args)
    K = jnp.dot(J, J)
    return K

def save_checkpoint(state, workdir, keep=5, name=None):
    #Use legacy checkpointing in order to run in colab 
    flax.config.update('flax_use_orbax_checkpointing', False)
    
    # Create the workdir if it doesn't exist.
    if not os.path.isdir(workdir):
        os.makedirs(workdir)

    # Save the checkpoint.
    if jax.process_index() == 0:
        # Get the first replica's state and save it.
        state = jax.device_get(tree_map(lambda x: x[0], state))
        step = int(state.step)
        checkpoints.save_checkpoint(workdir, state, step=step, keep=keep)


def restore_checkpoint(state, workdir, step=None):
    #Use legacy checkpointing in order to run in colab
    flax.config.update('flax_use_orbax_checkpointing', False)
    
    state = checkpoints.restore_checkpoint(workdir, state, step=step)
    return state

def save_sequential_checkpoints(config, workdir, model_1, model_2):
    """ Adaption of save_checkpoints to sequential learning of two models """
    #Use legacy checkpointing in order to run in colab 
    flax.config.update('flax_use_orbax_checkpointing', False)

    # determine current combined step
    state_1 = jax.device_get(tree_map(lambda x: x[0], model_1.state))
    step_1  = int(state_1.step)

    state_2 = jax.device_get(tree_map(lambda x: x[0], model_2.state)) 
    step_2  = int(state_2.step)

    step = step_1 + step_2

    # save each model
    for model, state in zip([model_1, model_2], [state_1, state_2]):
        path = os.path.join(workdir, "ckpt", config.wandb.name, model.tag)
        
        # Create the workdir if it doesn't exist.
        if not os.path.isdir(path):
            os.makedirs(path)
        
        # Save the checkpoint.
        if jax.process_index() == 0:
            # Get the first replica's state and save it.
            checkpoints.save_checkpoint(path, state, step=step, keep=config.saving.num_keep_ckpts)