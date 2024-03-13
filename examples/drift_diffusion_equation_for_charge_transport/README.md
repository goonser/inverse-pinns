# Inverse Drift-Diffusion Equation for unknown parameter ion mobility $\mu$

## Problem Setup

The partial differential equation is defined as 

$$\begin{aligned}
    \begin{cases}
        & \dfrac{\partial n}{\partial t} + \mu E \dfrac{\partial n}{\partial x} - \dfrac{\mu k_b T}{q} \dfrac{\partial n}{\partial x} = 0 \\
        & n(t,x = 0) = n_{inj}, \quad t \geq 0\\
        & n(t = 0, x) = n_0, \quad  x \in (0,1] \\
        & x \in [0, 1] \\
        & t \in [0, 0.007]
    \end{cases}
\end{aligned}$$

for observations of the volume density of positive ions $n(t,x)$. Observations are generated from the approximative solution for $E >> \frac{k_b T}{q}$ that is given by

$$
n(t,x) \approx
\begin{cases} 
& n_{inj}, \quad x \leq \mu E t \\
& n_0,  \quad \text{else}
\end{cases}$$
