from functools import partial
import jax
import jax.numpy as jnp
from jax import lax, jit, grad, vmap

from jaxpi.models import ForwardIVP
from jaxpi.evaluator import BaseEvaluator
from jaxpi.utils import ntk_fn, flatten_pytree

from matplotlib import pyplot as plt

from utils import get_observations, get_noisy_observations


class InversePoisson(ForwardIVP):
    def __init__(self, config, u0, u1, r_star, true_offset):
        super().__init__(config)

        self.eps = 8.85e-12
        self.rho = 5e-10
        self.true_offset = true_offset

        self.u0 = u0
        self.u1 = u1
        self.r_star = r_star

        self.r0 = r_star[0]
        self.r1 = r_star[-1]
        
        use_clean_data = config.setting.guassian_noise_perc is None or not config.setting.guassian_noise_perc != 0
        if use_clean_data:
            self.obs_r, self.obs_u = get_observations(self.r0, self.r1, self.true_offset, config)
        else: 
            self.obs_r, self.obs_u = get_noisy_observations(self.r0, self.r1, self.true_offset, config)

        #new  
        self.u_pred_fn = vmap(self.u_net, (None, 0))
        self.r_pred_fn = vmap(self.r_net, (None, 0))
        
        
        
    def u_net(self, params, r):
        # params = weights for NN 
        r_reshape = jnp.reshape(r, (1, -1)) # make it a 2d array with just one column to emulate jnp.stack()
        u = self.state.apply_fn(params, r_reshape) # gives r to the neural network's (self.state) forward pass (apply_fn)
        return (self.r1-r)/(self.r1-self.r0) + (r-self.r0)*(self.r1 - r)*u[0] # hard boundary
    

    def r_net(self, params, r):
        du_r = grad(self.u_net, argnums=1)(params, r)
        du_rr = grad(grad(self.u_net, argnums=1), argnums=1)(params, r)
        offset = params['params']['offset_param']
        C = jnp.exp(offset) # ensure C is positive
        return (r+C) * du_rr + du_r + (self.rho/self.eps) * (r+C) 
    
    @partial(jit, static_argnums=(0,))
    def res_and_w(self, params, batch):
        raise NotImplementedError(f"Casual weights not supported yet for 1D Laplace!")

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        # Residual loss
        if self.config.weighting.use_causal == True:
            raise NotImplementedError(f"Casual weights not supported yet for 1D Laplace!")
        else:
            r_pred = vmap(self.r_net, (None, 0))(params, batch[:,0]) 
            res_loss = jnp.mean((r_pred) ** 2)

        # Observation loss
        obs_u_pred = vmap(self.u_net, (None, 0))(params, self.obs_r)
        obs_loss = jnp.mean((self.obs_u - obs_u_pred) ** 2)

        loss_dict = {"res": res_loss, "observ": obs_loss}
        return loss_dict

    @partial(jit, static_argnums=(0,))
    def compute_diag_ntk(self, params, batch):
        #ics_ntk = vmap(ntk_fn, (None, None, 0))(
        #    self.u_net, params, self.r_star
        #)
        inner_bcs_ntk = self.u_net(params, self.r0)
        outer_bcs_ntk = self.u_net(params, self.r1)


        # Consider the effect of causal weights
        if self.config.weighting.use_causal: 
            raise NotImplementedError(f"Casual weights not supported yet for 1D Laplace!")

        else:
            res_ntk = vmap(ntk_fn, (None, None, 0))(
                self.r_net, params, batch[:, 0]
            )
        #ntk_dict = {"ics": ics_ntk, "res": res_ntk}
        ntk_dict = {"inner_bcs": inner_bcs_ntk, "outer_bcs": outer_bcs_ntk, "res": res_ntk}

        return ntk_dict

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, u_test):
        u_pred = self.u_pred_fn(params, self.r_star)
        error = jnp.linalg.norm(u_pred - u_test) / jnp.linalg.norm(u_test)
        return error
    
    def compute_parameter_l2_error(self, params, config):
        r0_pred = jnp.exp(params['params']['offset_param'][0])
        r0_ref = config.setting.true_offset
        error = jnp.abs(r0_pred-r0_ref)/r0_ref
        return error


class InversePoissonEvaluator(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, u_ref):
        l2_error = self.model.compute_l2_error(params, u_ref)
        self.log_dict["l2_error"] = l2_error

    def log_param_errors(self, params, config):
        param_error = self.model.compute_parameter_l2_error(params, config)
        self.log_dict["l2_param_error"] = param_error

    def log_preds(self, params):
        u_pred = self.model.u_pred_fn(params, self.model.r_star)
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
            self.log_param_errors(state.params, self.config)

        if self.config.logging.log_preds:
            self.log_preds(state.params)

        return self.log_dict