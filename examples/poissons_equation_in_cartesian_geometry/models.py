from functools import partial
import jax
import jax.numpy as jnp
from jax import lax, jit, grad, vmap

from jaxpi.models import ForwardIVP
from jaxpi.evaluator import BaseEvaluator
from jaxpi.utils import ntk_fn, flatten_pytree

from utils import get_dataset, get_observations, get_noisy_observations, get_reference_dataset

from matplotlib import pyplot as plt


class InversePoisson(ForwardIVP):
    def __init__(self, config, u0, u1, x_star, n_scale):
        super().__init__(config)

        self.u_scale = u0
        self.u0 = u0 / self.u_scale
        self.u1 = u1 / self.u_scale
        self.x_star = x_star
        self.n_scale = n_scale
        self.loss_scale = config.setting.loss_scale

        self.x0 = x_star[0]
        self.x1 = x_star[-1]
        self.dom = jnp.array([self.x0, self.x1])

        # parameters 
        self.q = 1.602e-19
        self.epsilon = 8.85e-12

        # mappings  
        self.u_pred_fn = vmap(self.u_net, (None, 0))
        self.r_pred_fn = vmap(self.r_net, (None, 0))

        # Number of points to sample for observation loss
        if config.setting.guassian_noise_perc is not None:
            self.obs_x, self.obs_u = get_noisy_observations(config)
        else:    
            self.obs_x, self.obs_u = get_observations(config)

        self.k = config.setting.k   

        #new 
        self.u_pred_fn = vmap(self.u_net, (None, 0))
        self.n_pred_fn = vmap(self.n_net, (None, 0))
        self.r_pred_fn = vmap(self.r_net, (None, 0))

        # Check so that the paths are passed in the config file, if None, not used. 
        if config.eval.potential_file_path is not None and config.eval.field_file_path is not None:
            self.x_ref, self.E_ref, self.u_ref = get_reference_dataset(config, config.eval.field_file_path, config.eval.potential_file_path)
        else:
            if config.logging.log_errors == True:
                print('Missing reference data: Setting log_errors to False')
                config.logging.log_errors = False
                
    def neural_net(self, params, x):
        # params = weights for NN 
        # make it a 2d array with just one column to emulate jnp.stack()
        x_reshape = jnp.reshape(x, (1, -1)) 
        # gives r to the neural network's (self.state) forward pass (apply_fn)
        y = self.state.apply_fn(params, x_reshape) 
        u = y[0] # first output of the neural network
        n = y[1] # second output of the neural network
        return u, n
        
    def u_net(self, params, x):
        u, _ = self.neural_net(params, x)
        return (self.x1-x)/(self.x1-self.x0) + (x-self.x0)*(self.x1 - x)*u # Hard boundary
    
    def n_net(self, params, x):
        _, n = self.neural_net(params, x)
        return n

    def r_net(self, params, x):        
        du_xx = grad(grad(self.u_net, argnums=1), argnums=1)(params, x)
        if self.config.arch.arch_name == "InverseMlpCaseChargeProfile":
            a = params['params']['n_scale_param'][0]
            n = self.n_net(params, x) * (10 ** a)
        else:
            n = self.n_net(params, x) * self.n_scale
        return du_xx * self.u_scale + self.q * n / self.epsilon
    
    def heaviside(self, x):
        # https://en.wikipedia.org/wiki/Heaviside_step_function
        # larger k -> steeper step
        # larger a -> larger positive translation
        k = self.k
        a = 0.5
        return 1 - 1 / (1 + jnp.exp(-2 * k * (x - a)))
    
    @partial(jit, static_argnums=(0,))
    def res_and_w(self, params, batch):
        raise NotImplementedError(f"Casual weights not supported for 1D!")

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch): 

        # Residual loss
        if self.config.weighting.use_causal == True:
            raise NotImplementedError(f"Casual weights not supported for 1D!")
        else:
            r_pred = vmap(self.r_net, (None, 0))(params, batch[:,0])
            r_pred *= self.loss_scale 
            res_loss = jnp.mean((r_pred) ** 2)

        # Observation loss
        obs_u_pred = vmap(self.u_net, (None, 0))(params, self.obs_x)
        obs_loss = jnp.mean((self.loss_scale * (self.obs_u - self.u_scale * obs_u_pred)) ** 2)

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
            raise NotImplementedError(f"Casual weights not supported for 1D Laplace!")

        else:
            res_ntk = vmap(ntk_fn, (None, None, 0))(
                self.r_net, params, batch[:, 0]
            )
        #ntk_dict = {"ics": ics_ntk, "res": res_ntk}
        ntk_dict = {"inner_bcs": inner_bcs_ntk, "outer_bcs": outer_bcs_ntk, "res": res_ntk}

        return ntk_dict

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, _):
        # Compute n_l2 error:
        n_pred = self.n_pred_fn(params, self.x_star)
        n_true = self.heaviside(self.x_star)
        n_error = jnp.linalg.norm(n_pred - n_true) / jnp.linalg.norm(n_true)

        # Compute u_l2 error:
        u_ref = self.u_ref
        u_pred = self.u_pred_fn(params, self.x_ref)
        u_pred *= self.u_scale
        u_error = jnp.linalg.norm(u_pred - u_ref) / jnp.linalg.norm(u_ref)
        return u_error, n_error


class InversePoissonEvaluator(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, u_ref):
        u_l2_error, n_l2_error = self.model.compute_l2_error(params, u_ref)
        self.log_dict["u_l2_error"] = u_l2_error
        self.log_dict["n_l2_error"] = n_l2_error

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

        if self.config.logging.log_preds:
            self.log_preds(state.params)

        return self.log_dict