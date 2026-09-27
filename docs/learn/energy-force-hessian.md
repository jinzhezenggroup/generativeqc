# Energy, forces, gradients, and Hessians

For a fixed electronic method, the Born-Oppenheimer energy is a scalar function of the nuclear coordinates $\mathbf R$.

The nuclear gradient components are

$$
g_{A\alpha}
=
\frac{\partial E}{\partial R_{A\alpha}},
$$

where $A$ labels the nucleus and $\alpha\in\{x,y,z\}$. The force has the opposite sign,

$$
F_{A\alpha}
=
-\frac{\partial E}{\partial R_{A\alpha}}
=
-g_{A\alpha}.
$$

A stationary geometry satisfies

$$
\nabla_\mathbf R E = 0.
$$

A geometry optimizer uses first derivatives to search for such structures.

The **Hessian** contains second derivatives of the energy,

$$
H_{A\alpha,B\beta}
=
\frac{\partial^2 E}
{\partial R_{A\alpha}\partial R_{B\beta}},
$$

and describes local curvature. It is used for vibrational analysis, stationary-point characterization, and curvature-aware optimization.

For a displacement direction $\mathbf v$, a Hessian-vector product is

$$
\mathbf H\mathbf v
=
\left.
\frac{d}{dt}
\nabla E(\mathbf R+t\mathbf v)
\right|_{t=0},
$$

so a method can apply local curvature without necessarily forming and storing the full Hessian matrix.

Analytic electronic-structure derivatives generally contain more than explicit integral derivatives: orbital relaxation, overlap/Pulay terms, and method-specific response contributions must be included whenever the stationary structure does not make them vanish.

Do not infer derivative support merely because an energy endpoint exists; support must be documented for the chosen method/backend.

Next: [post-HF methods](post-hf.md).
