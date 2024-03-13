# Inverse Poissons's equation for one-dimensional Cartesian geometry for a stationary, unknown volume density of positive ions $n(x)$

## Problem Setup

The partial differential equation is defined as 

$$\begin{aligned}
    \begin{cases}
        & \dfrac{\partial^2 U}{\partial x^2} +  \dfrac{qn(x;k)}{\partial \epsilon} = 0 \\
        & U(x=x_0) = U_0 \\
        & U(x=x_1) = U_1 \\
        & n(x;k) = n_{inj} \bigl( 1 - \dfrac{1}{1 + \exp(-2 k (x - 0.5))}\bigl) \\
        & x \in [0,1]
    \end{cases}
\end{aligned}$$

for a steepness parameter k and observations of the potential $U(r)$.
