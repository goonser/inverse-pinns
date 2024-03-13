# JAXPI

This repository is an adaptation of the [JAX-PI repository from Predictive Intelligence Lab](https://github.com/PredictiveIntelligenceLab/jaxpi) for a master's thesis at the Department of Electrical Engineering at Chalmers University of Technology. It offers a comprehensive implementation of physics-informed neural networks (PINNs), seamlessly integrating several advanced network architectures, and training algorithms from [An Expert's Guide to Training Physics-informed Neural Networks
](https://arxiv.org/abs/2308.08468) and adapts these to the domain of inverse problems for discharge physics. 
## Demo 
A demonstration notebook for running the code on Google Colab is available [here](https://colab.research.google.com/drive/1a33Zx5J9NJ3mn8uNzxFKQq_m0DLjST9Q?usp=sharing). The next sections will provide a more thorough description of using the repo. 

## Installation

Please ensure you have Python 3.8 or later installed on your system.
Our code is GPU-only.
We highly recommend using the most recent versions of JAX and JAX-lib, along with compatible CUDA and cuDNN versions.
The code has been tested and confirmed to work with the following versions:

- JAX 0.4.5
- CUDA 11.7
- cuDNN 8.2


Install JAX-PI with the following commands:

``` 
git clone https://github.com/goonser/inverse-pinns.git
cd inverse-pinns/
pip install .
```

## Quickstart

We use [Weights & Biases](https://wandb.ai/site) to log and monitor training metrics. 
Please ensure you have Weights & Biases installed and properly set up with your account before proceeding. 
You can follow the installation guide provided [here](https://docs.wandb.ai/quickstart).

To illustrate how to use our code, we will use the Poisson's equation in cartesian geometry as an example. 
First, navigate to the advection directory within the `examples` folder:

``` 
cd inverse-pinns/examples/poissons_equation_in_cartesian_geometry
``` 
To train the model, run the following command:
```
python3 main.py 
```

Our code automatically supports multi-GPU execution. 
You can specify the GPUs you want to use with the `CUDA_VISIBLE_DEVICES` environment variable. For example, to use the first two GPUs (0 and 1), use the following command:

```
CUDA_VISIBLE_DEVICES=0,1 python3 main.py
```

**Note on Memory Usage**: Different models and examples may require varying amounts of GPU memory. 
If you encounter an out-of-memory error, you can decrease the batch size using the `--config.batch_size_per_device` option.

To evaluate the model's performance, you can switch to evaluation mode with the following command:

```
python main.py --config.mode=eval
```

## Code structure
The code corresponding to each problem is entered in a folder in the examples directory (e.g., examples/poissons_equation_in_coaxial_geometry/). Here is an overview of the contents of each such file:
| Name                                   | Function                                                                                                                      |
|----------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| configs                                | Folder containing all config files                                                                                            |
| config file (e.g., configs/default.py) | Contains training-related configurations such as network architecture, batch size, iterations, and problem-specific variables |
| models.py                               | Specifies the loss functions and core methods. Here is where most changes are done when adapting the code to a new PDE         |
| train.py                               | Specifies the training process                                                                                                |
| eval.py                                | Evaluates the model and creates plots                                                                                         |
| main.py                                | Runs either train.py or eval.py depending on mode                                                                             |

### models.py
The file models.py contains the model describing the PDE and the related losses. The core functions are as follows: 


| Function  | Purpose                                                                                                                                                                                                                      |
|-----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| u_net     | Performs a forward pass of the neural network and outputs a model prediction $u(t,x)$. Here is where hard boundary conditions should be introduced, if any.                                                                  |
| r_net     | Calculates the PDE residual for a given (t,x).                                                                                                                                                                               |
| losses    | Computes the squared initial, boundary, residual, and observation loss (if applicable) across a sampled batch.                                                                                                                   |
| res_and_w | Calculates weights for each sequential segment of the temporal domain if config.weighting.causal_tol = True (i.e., if using modified residual loss to avoid violating causality). Not applicable for non-time-dependent PDEs. |


### configs
The config files contain a larger number of training and problem-specific configurations, variables, and settings. Here is an overview of some of the most frequently modified: 
| Config                | Purpose                                                                                                          |
|-----------------------|------------------------------------------------------------------------------------------------------------------|
| mode                  | `train` for training, `eval` for evaluation                                                                      |
| num_layers            | Depth of NN                                                                                                      |
| layer_size            | Width of NN                                                                                                      |
| activation            | Activation function (e.g., 'tanh' or 'gelu')                                                                     |
| fourier_emb           | Fourier embedding scale and dimension                                                                            |
| reparam               | Random weight factorization                                                                                      |
| max_steps             | Number of iterations before stopping                                                                             |
| batch_size_per_device | Batch size when training (a smaller value than the default 4096 recommended)                                     |
| use_causal            | Use modified PDE residual loss to avoid violating causality. Should be None or False for non-time-dependent PDEs |

## Citation
**TODO ADD BibTex Citation**

Citation for original JAXPI:

    @article{wang2023expert,
      title={An Expert's Guide to Training Physics-informed Neural Networks},
      author={Wang, Sifan and Sankaran, Shyam and Wang, Hanwen and Perdikaris, Paris},
      journal={arXiv preprint arXiv:2308.08468},
      year={2023}
    }
