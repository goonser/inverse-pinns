import ml_collections

import jax.numpy as jnp


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.mode = "train"
    
    # Problem setting 
    config.setting = setting = ml_collections.ConfigDict()
    setting.guassian_noise_perc = 0.01
    setting.obs_file = "obs_k_100.dat"
    setting.n_scale = 5e13
    setting.n_x = 12800
    setting.n_obs = 1000
    setting.u0 = 1e6
    setting.u1 = 0
    setting.k = 100
    setting.loss_scale = 1

    # Sampler Config
    config.sampler = sampler = ml_collections.ConfigDict()
    sampler.sampler_name = "random"
    sampler.resample_every_steps = 20_000 # If sampler_name = "rad" Resample new RAD points every 10_000 steps
    sampler.num_rad_points = 100_000
    sampler.plot_rad = False
    sampler.c = 1
    sampler.k = 0.5
    sampler.gamma = 0
    sampler.cosine_lr = 0.9
    sampler.cosine_T = 10
    sampler.plot_batch = False

    # Evaluate 
    config.eval = eval = ml_collections.ConfigDict()
    # COMSOL reference solution files (set None if not available for the current n_inj
    eval.potential_file_path = 'Case1p5_validation_data_U_vs_x_ninj5e13.txt(1).txt'
    eval.field_file_path = 'Case1p5_validation_data_E_vs_x_ninj5e13.txt(1).txt'

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "PINN-Inverse-Case1.5"   
    wandb.name = "default"
    wandb.tag = None

    # Arch
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "InverseMlpCaseChargeProfile"
    arch.num_layers = 4
    arch.layer_size = 64
    arch.out_dim = 2
    arch.activation = "gelu"
    arch.periodicity = ml_collections.ConfigDict({"period": (1.0,), "axis": (1,), "trainable": (False,)})

    arch.fourier_emb = ml_collections.ConfigDict({"embed_scale": 10.0, "embed_dim": 256})
    arch.reparam = ml_collections.ConfigDict({"type": "weight_fact", "mean": 1.0, "stddev": 0.1})

    # Optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "Adam"
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.learning_rate = 1e-3
    optim.decay_rate = 0.9
    optim.decay_steps = 2000
    optim.grad_accum_steps = 0
    optim.weight_decay = 1e-2   # L2 Regularization strength

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 150_000
    training.batch_size_per_device = 8192

    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = None #"grad_norm"
    weighting.init_weights = ml_collections.ConfigDict({"res": 1.0, "observ": 1.0})
    weighting.momentum = 0.9
    weighting.update_every_steps = 1000

    weighting.use_causal = False 
    weighting.causal_tol = 1.0
    weighting.num_chunks = 32

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 1_000
    logging.log_errors = True
    logging.log_losses = True
    logging.log_weights = True
    logging.log_grads = False
    logging.log_ntk = False
    logging.log_preds = False

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_every_steps = 25_000
    saving.num_keep_ckpts = 1
    saving.plot = True

    # # Input shape for initializing Flax models
    config.input_dim = 1

    # Integer for PRNG random seed.
    config.seed = 43

    return config
