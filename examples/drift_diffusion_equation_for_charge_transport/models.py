from functools import partial
import jax
import jax.numpy as jnp
from jax import lax, jit, grad, vmap
from jaxpi.models import ForwardIVP
from jaxpi.evaluator import BaseEvaluator
from jaxpi.utils import ntk_fn, flatten_pytree

from utils import get_dataset

from matplotlib import pyplot as plt


class InverseDriftDiffusion(ForwardIVP):
    def __init__(self, config, t_star, x_star, u_exact_fn):
        super().__init__(config)
        
        #rescale n_inj and n0 to [0,1]
        n_inj = 1.0
        n_0 = config.setting.n_0 / config.setting.n_inj
        self.n_inj_scale = config.setting.n_inj
        self.n_t_obs = config.setting.n_t_obs
        self.n_x_obs = config.setting.n_x_obs

        # constants
        self.true_mu = config.setting.true_mu
        self.E_ext = config.setting.E_ext
        self.Temp = 293
        self.q = 1.602e-19
        self.kb = 1.38e-23

        # functions
        self.obs_t = jax.random.uniform(jax.random.PRNGKey(0), (self.n_t_obs,), minval=t_star[0], maxval=t_star[-1])
        self.obs_x = jax.random.uniform(jax.random.PRNGKey(0), (self.n_x_obs,), minval=x_star[0], maxval=x_star[-1])
        
        if config.setting.noise_level is not None:
            exact_points = u_exact_fn(self.obs_t, self.obs_x)
            self.obs_u = self.add_noise_to_data(exact_points)
        else:
            self.obs_u = u_exact_fn(self.obs_t, self.obs_x)

        # initial conditions
        self.n_injs = jnp.full_like(t_star, n_inj)
        self.n_0s = jnp.full_like(x_star, n_0)
        
        # domain
        self.t_star = t_star
        self.x_star = x_star

        self.t0 = t_star[0]
        self.t1 = t_star[-1]

        # Predictions over a grid
        self.u_pred_fn = vmap(vmap(self.u_net, (None, None, 0)), (None, 0, None))
        self.r_pred_fn = vmap(vmap(self.r_net, (None, None, 0)), (None, 0, None))

    def add_noise_to_data(self, u_exact):
        noise_level = self.config.setting.noise_level
        u_noise = u_exact + noise_level * jax.random.normal(jax.random.PRNGKey(0), u_exact.shape)
        return u_noise

    def u_net(self, params, t, x):
        z = jnp.stack([t, x])
        u = self.state.apply_fn(params, z)
        return u[0]
    
    #def scaled_u_net(self, params, t, x): 
    #    # scale predictions back up from [0,1] to [0, n_inj]
    #    return self.n_inj_scale * self.u_net(params, t, x)

    def r_net(self, params, t, x):
        # Fetching current value of mu form the parameter set
        mu_n = jnp.exp(params['params']['mu_param'])
        
        W = mu_n * self.E_ext
        Diff = mu_n * self.kb * self.Temp/self.q 

        dn_t = grad(self.u_net, argnums=1)(params, t, x)
        dn_x = grad(self.u_net, argnums=2)(params, t, x)
        dn_xx = grad(grad(self.u_net, argnums=2), argnums=2)(params, t, x)
        return 1/W*dn_t + dn_x - Diff/W*dn_xx

    @partial(jit, static_argnums=(0,))
    def res_and_w(self, params, batch):
        # Sort temporal coordinates for computing  temporal weights
        t_sorted = batch[:, 0].sort()
        # Compute residuals over the full domain
        r_pred = vmap(self.r_net, (None, 0, 0))(params, t_sorted, batch[:, 1])
        # Split residuals into chunks
        r_pred = r_pred.reshape(self.num_chunks, -1)
        l = jnp.mean(r_pred**2, axis=1)
        # Compute temporal weights
        w = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ l)))
        return l, w

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        # Initial loss 
        u_pred = vmap(self.u_net, (None, None, 0))(params, self.t0, self.x_star)
        ics_loss = jnp.mean((self.n_0s[1:] - u_pred[1:]) ** 2) # slicing to exclude x = 0

        # Boundary loss
        x_0 = 0
        u_pred = vmap(self.u_net, (None, 0, None))(params, self.t_star, x_0)
        bcs_loss = jnp.mean((self.n_injs - u_pred) ** 2)

        # Observation 
        obs_u_pred = self.u_pred_fn(params, self.obs_t, self.obs_x) 
        obs_loss = jnp.mean((self.obs_u - obs_u_pred) ** 2)

        # Residual loss
        if self.config.weighting.use_causal == True:
            l, w = self.res_and_w(params, batch)
            res_loss = jnp.mean(l * w)
        else:
            r_pred = vmap(self.r_net, (None, 0, 0))(params, batch[:, 0], batch[:, 1]) 
            res_loss = jnp.mean((r_pred) ** 2)

        loss_dict = {"ics": ics_loss, "bcs": bcs_loss, "res": res_loss, "obs" : obs_loss}
        return loss_dict

    @partial(jit, static_argnums=(0,))
    def compute_diag_ntk(self, params, batch):
        ics_ntk = vmap(ntk_fn, (None, None, None, 0))(
            self.u_net, params, self.t0, self.x_star
        )

        # Consider the effect of causal weights
        if self.config.weighting.use_causal:
            # sort the time step for causal loss
            batch = jnp.array([batch[:, 0].sort(), batch[:, 1]]).T
            res_ntk = vmap(ntk_fn, (None, None, 0, 0))(
                self.r_net, params, batch[:, 0], batch[:, 1]
            )

            res_ntk = res_ntk.reshape(self.num_chunks, -1)  # shape: (num_chunks, -1)
            res_ntk = jnp.mean(
                res_ntk, axis=1
            )  # average convergence rate over each chunk
            _, casual_weights = self.res_and_w(params, batch)
            res_ntk = res_ntk * casual_weights  # multiply by causal weights
        else:
            res_ntk = vmap(ntk_fn, (None, None, 0, 0))(
                self.r_net, params, batch[:, 0], batch[:, 1]
            )

        ntk_dict = {"ics": ics_ntk, "res": res_ntk}

        return ntk_dict

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, u_test):
        u_pred = self.u_pred_fn(params, self.t_star, self.x_star)
        error = jnp.linalg.norm(u_pred - u_test) / jnp.linalg.norm(u_test)
        return error


class InverseDriftDiffusionEvalutor(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, u_ref):
        l2_error = self.model.compute_l2_error(params, u_ref)
        self.log_dict["l2_error"] = l2_error

    def log_preds(self, params):
        u_pred = self.model.u_pred_fn(params, self.model.t_star, self.model.x_star)
        fig = plt.figure(figsize=(6, 5))
        plt.imshow(u_pred.T, cmap="jet")
        self.log_dict["u_pred"] = fig
        plt.close()

    def __call__(self, state, batch, u_ref):
        self.log_dict = super().__call__(state, batch)

        if self.config.weighting.use_causal:
            _, causal_weight = self.model.res_and_w(state.params, batch)
            self.log_dict["cas_weight"] = causal_weight.min()

        if self.config.logging.log_errors:
            self.log_errors(state.params, u_ref)

        if self.config.logging.log_preds:
            self.log_preds(state.params)

        return self.log_dict
