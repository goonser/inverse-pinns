from functools import partial
from typing import Any, Callable, Sequence, Tuple, Optional, Union, Dict

from flax import linen as nn
from flax.core.frozen_dict import freeze

from jax import random, jit, vmap
import jax.numpy as jnp
from jax.nn.initializers import glorot_normal, normal, zeros, constant
import jax
import time

activation_fn = {
    "relu": nn.relu,
    "gelu": nn.gelu,
    "swish": nn.swish,
    "sigmoid": nn.sigmoid,
    "tanh": jnp.tanh,
    "sin": jnp.sin,
}


def _get_activation(str):
    if str in activation_fn:
        return activation_fn[str]

    else:
        raise NotImplementedError(f"Activation {str} not supported yet!")


def _weight_fact(init_fn, mean, stddev):
    def init(key, shape):
        key1, key2 = random.split(key)
        w = init_fn(key1, shape)
        g = mean + normal(stddev)(key2, (shape[-1],))
        g = jnp.exp(g)
        v = w / g
        return g, v

    return init


class PeriodEmbs(nn.Module):
    period: Tuple[float]  # Periods for different axes
    axis: Tuple[int]  # Axes where the period embeddings are to be applied
    trainable: Tuple[
        bool
    ]  # Specifies whether the period for each axis is trainable or not

    def setup(self):
        # Initialize period parameters as trainable or constant and store them in a flax frozen dict
        period_params = {}
        for idx, is_trainable in enumerate(self.trainable):
            if is_trainable:
                period_params[f"period_{idx}"] = self.param(
                    f"period_{idx}", constant(self.period[idx]), ()
                )
            else:
                period_params[f"period_{idx}"] = self.period[idx]

        self.period_params = freeze(period_params)

    @nn.compact
    def __call__(self, x):

        """
        Apply the period embeddings to the specified axes.
        """
        y = []

        for i, xi in enumerate(x):
            if i in self.axis:
                raise NotImplementedError('Should not be here!!!!')
                idx = self.axis.index(i)
                period = self.period_params[f"period_{idx}"]
                y.extend([jnp.cos(period * xi), jnp.sin(period * xi)])
            else:
                y.append(xi)

        return jnp.hstack(y)


class FourierEmbs(nn.Module):
    embed_scale: float
    embed_dim: int

    @nn.compact
    def __call__(self, x):
        kernel = self.param(
            "kernel", normal(self.embed_scale), (x.shape[-1], self.embed_dim // 2)
        )
        y = jnp.concatenate(
            [jnp.cos(jnp.dot(x, kernel)), jnp.sin(jnp.dot(x, kernel))], axis=-1
        )
        return y


class Dense(nn.Module):
    features: int
    kernel_init: Callable = glorot_normal()
    bias_init: Callable = zeros
    reparam: Union[None, Dict] = None

    @nn.compact
    def __call__(self, x):
        if self.reparam is None:
            kernel = self.param(
                "kernel", self.kernel_init, (x.shape[-1], self.features)
            )

        elif self.reparam["type"] == "weight_fact":
            g, v = self.param(
                "kernel",
                _weight_fact(
                    self.kernel_init,
                    mean=self.reparam["mean"],
                    stddev=self.reparam["stddev"],
                ),
                (x.shape[-1], self.features),
            )
            kernel = g * v

        bias = self.param("bias", self.bias_init, (self.features,))

        y = jnp.dot(x, kernel) + bias

        return y


# TODO: Make it more general, e.g. imposing periodicity for the given axis


class Mlp(nn.Module):
    arch_name: Optional[str] = "Mlp"
    num_layers: int = 4
    layer_size: int = 256
    out_dim: int = 1
    activation: str = "tanh"
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    reparam: Union[None, Dict] = None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        if self.periodicity:
            x = PeriodEmbs(**self.periodicity)(x)
        if self.fourier_emb:
            x = FourierEmbs(**self.fourier_emb)(x)

        for _ in range(self.num_layers):
            x = Dense(features=self.layer_size, reparam=self.reparam)(x)
            x = self.activation_fn(x)

        x = Dense(features=self.out_dim, reparam=self.reparam)(x)
        return x

class MlpDriftDiffusion(Mlp):

    def setup(self):
        super().setup()  # Call the setup method of the parent class

    @nn.compact
    def __call__(self, x):
        if self.periodicity:
            x = PeriodEmbs(**self.periodicity)(x)
        if self.fourier_emb:
            x = FourierEmbs(**self.fourier_emb)(x)

        for _ in range(self.num_layers):
            x = Dense(features=self.layer_size, reparam=self.reparam)(x)
            x = self.activation_fn(x)

        x = Dense(features=self.out_dim, reparam=self.reparam)(x)
        x = nn.sigmoid(x)
        return x


class InverseMlpCaseChargeProfile(Mlp):

    def setup(self):
        super().setup()  # Call the setup method of the parent class

        # Intorducting scale parameter as learnable parameter #TODO make random initailization 
        self.n_scale = self.param('n_scale_param', lambda _: jnp.array([1.0]))
        
    @nn.compact
    def __call__(self, x):
        if self.periodicity:
            x = PeriodEmbs(**self.periodicity)(x)
        if self.fourier_emb:
            x = FourierEmbs(**self.fourier_emb)(x)

        for _ in range(self.num_layers):
            x = Dense(features=self.layer_size, reparam=self.reparam)(x)
            x = self.activation_fn(x)

        x = Dense(features=self.out_dim, reparam=self.reparam)(x)
        x = nn.sigmoid(x)

        return x
    

class InverseMlpOffset(Mlp):
    arch_name: Optional[str] = "InverseMlpOffset"

    def setup(self):
        super().setup()  # Call the setup method of the parent class

        # Additional setup for InverseMlp
        self.offset_param = self.param('offset_param', lambda rng: jax.random.uniform(rng, (1,), minval=jnp.log(0.05), maxval=jnp.log(0.5))) #minval=jnp.log(0.01), maxval=jnp.log(0.1)))
        
class InverseMlpRho(Mlp):
    arch_name: Optional[str] = "InverseMlpRho"

    def setup(self):
        super().setup()  # Call the setup method of the parent class

        # Additional setup for InverseMlp
        self.rho_param = self.param('rho_param', lambda rng: jax.random.uniform(rng, (1,)))


class InverseMlpMu(MlpDriftDiffusion):
    arch_name: Optional[str] = "InverseMlpMu"

    def setup(self):
        super().setup()  # Call the setup method of the parent class

        # Additional setup for InverseMlp
        self.mu_param = self.param('mu_param', lambda rng: jax.random.uniform(rng, (1,), minval=jnp.log(2e-5), maxval=jnp.log(2e-3)))
          

class ModifiedMlp(nn.Module):
    arch_name: Optional[str] = "ModifiedMlp"
    num_layers: int = 4
    layer_size: int = 256
    out_dim: int = 1
    activation: str = "tanh"
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    reparam: Union[None, Dict] = None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        if self.periodicity:
            x = PeriodEmbs(**self.periodicity)(x)

        if self.fourier_emb:
            x = FourierEmbs(**self.fourier_emb)(x)

        u = Dense(features=self.layer_size, reparam=self.reparam)(x)
        v = Dense(features=self.layer_size, reparam=self.reparam)(x)

        u = self.activation_fn(u)
        v = self.activation_fn(v)

        for _ in range(self.num_layers):
            x = Dense(features=self.layer_size, reparam=self.reparam)(x)
            x = self.activation_fn(x)
            x = x * u + (1 - x) * v

        x = Dense(features=self.out_dim, reparam=self.reparam)(x)
        return x


class MlpBlock(nn.Module):
    num_layers: int
    layer_size: int
    out_dim: int
    activation: str
    reparam: Union[None, Dict]
    final_activation: bool

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_layers):
            x = Dense(features=self.layer_size, reparam=self.reparam)(x)
            x = self.activation_fn(x)

        x = Dense(features=self.out_dim, reparam=self.reparam)(x)
        if self.final_activation:
            x = self.activation_fn(x)

        return x


class DeepONet(nn.Module):
    arch_name: Optional[str] = "DeepONet"
    num_branch_layers: int = 4
    num_trunk_layers: int = 4
    layer_size: int = 256
    out_dim: int = 1
    activation: str = "tanh"
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    reparam: Union[None, Dict] = None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, u, x):
        u = MlpBlock(
            num_layers=num_branch_layers,
            layer_size=self.layer_size,
            out_dim=self.layer_size,
            activation=self.activation,
            final_activation=False,
            reparam=self.reparam,
        )(u)

        x = Mlp(
            num_layers=num_trunk_layers,
            layer_size=self.layer_size,
            out_dim=self.layer_size,
            activation=self.activation,
            periodicity=self.periodicity,
            fourier_emb=self.fourier_emb,
            reparam=self.reparam,
        )(x)

        y = u * x
        y = self.activation_fn(y)
        y = Dense(features=self.out_dim, reparam=self.reparam)(y)
        return y
