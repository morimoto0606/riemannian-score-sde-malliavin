# Malliavin Teacher Implementation

This document fixes the notation and the theoretical interpretation of the
Malliavin teacher in this repository. The implementation is a replacement for
the conditional-score teacher in the upstream DSM pipeline. It does not change
the upstream network, score parameterisation, DSM weighting, optimiser, EMA,
or reverse sampler.

## 1. Theory and notation

Consider a manifold-valued SDE whose solution can be written as

$$
X_t = \Phi_t(X_0,W).
$$

For the numerical teacher, the Brownian path is replaced by a finite collection
of independent standard Gaussian increments $Z=(Z_1,\ldots,Z_N)$, giving the
differentiable endpoint map

$$
F_{X_0,t}(Z)=X_t.
$$

### Flow Jacobian

The symbol

$$
J_{t,s}=\frac{\partial X_t}{\partial X_s}
$$

is reserved for the Jacobian of the stochastic flow between states. It is not
the Malliavin derivative and is not the quantity named `endpoint_jacobian` in
the current implementation.

### Malliavin derivative

The Malliavin derivative in Brownian direction \(\alpha\) at time \(s\) is
written

$$
D_s^\alpha X_t.
$$

For the projected Brownian SDE on \(S^2\), its continuous-time form is

$$
D_s^\alpha X_t
=J_{t,s}\,\sigma(s)P(X_s)e_\alpha,
\qquad
P(x)=I-xx^\top.
$$

Here $J_{t,s}$ is the flow Jacobian, $P(X_s)e_\alpha$ is a tangent noise
direction, and \(\sigma(s)\) is the diffusion coefficient. In the discretised
implementation, the corresponding object is

$$
D_ZX_t=\frac{\partial F_{X_0,t}(Z)}{\partial Z}.
$$

It already includes the GRW step size and diffusion coefficients through the
endpoint map. The code calls this matrix `endpoint_jacobian`; despite that
name, it is a derivative with respect to Gaussian noise, not
\(\partial X_t/\partial X_0\) and not the flow Jacobian $J_{t,s}$.

### Malliavin covariance and covering process

Let $B(X_t)\in\mathbb R^{3\times2}$ be an orthonormal basis of
\(T_{X_t}S^2\), and define the tangent-coordinate noise derivative

$$
A=B(X_t)^\top D_ZX_t.
$$

The discrete Malliavin covariance is

$$
\Gamma=AA^\top.
$$

For endpoint vector fields $V=[V_1,V_2,V_3]$, with

$$
V_j(x)=P(x)e_j,
$$

the regularised minimum-energy covering weight is

$$
U
=A^\top(\Gamma+\lambda I)^{-1}B(X_t)^\top V(X_t).
$$

In the ideal unregularised full-rank case this satisfies $AU=B^\top V$.
The implementation uses a small positive \(\lambda\) for numerical stability.

### Skorokhod operator and score identity

The Skorokhod integral is denoted only by

$$
\delta(U)=D^*U.
$$

For finite-dimensional standard Gaussian noise it is

$$
\delta(U)=U^\top Z-\operatorname{div}_Z U.
$$

Gaussian integration by parts and manifold integration by parts give, for
each endpoint field $V_j$,

$$
V_j\log p_{t\mid0}(X_t\mid X_0)
=\mathbb E\!\left[
-\delta(U_j)-\operatorname{div}V_j(X_t)
\mid X_t,X_0
\right].
$$

Thus one path produces the stochastic directional target

$$
T_j=-D^*U_j-\operatorname{div}V_j.
$$

On $S^2$, \(\operatorname{div}V_j(x)=-2x_j\). The three directional
components are redundant but span the two-dimensional tangent space. The code
reconstructs the final tangent target using the orthonormal basis $B$. Since
\(V=P=BB^\top\), this reconstruction is exactly

$$
T_{\mathrm{vec}}=B(B^\top T),
$$

with no additional scale factor or pseudoinverse.

## 2. Relation to denoising score matching

The Heat and Varadhan teachers directly evaluate an approximation to the
conditional transition score

$$
\nabla_{X_t}\log p_{t\mid0}(X_t\mid X_0).
$$

The Malliavin teacher instead returns a pathwise stochastic estimator. More
precisely, its target depends on the simulated path (or $Z$) as well as
\(X_0,t\):

$$
T=T(X_0,Z,t),
$$

and ideally satisfies

$$
\mathbb E[T\mid X_t,X_0,t]
=\nabla_{X_t}\log p_{t\mid0}(X_t\mid X_0).
$$

The tower property then gives

$$
\mathbb E[T\mid X_t,t]
=\nabla_{X_t}\log p_t(X_t).
$$

Consequently, the population regression objective

$$
\mathbb E\left[
w(t)\lVert s_\theta(X_t,t)-T\rVert^2
\right]
$$

has the same minimiser as regression onto the marginal score. This follows
from the conditional squared-error decomposition

$$
\begin{aligned}
\mathbb E[w(t)\lVert s_\theta-T\rVert^2]
={}&\mathbb E\left[
w(t)\lVert s_\theta-\mathbb E[T\mid X_t,t]\rVert^2
\right]\\
&+\mathbb E\left[
w(t)\lVert T-\mathbb E[T\mid X_t,t]\rVert^2
\right].
\end{aligned}
$$

The second term is independent of \(\theta\). In this repository,
`like_w: false` uses the unchanged upstream DSM weight $w(t)=\sigma(t)^2$.
Therefore the implemented Malliavin loss is

$$
\mathbb E\left[
\sigma(t)^2
\left\lVert
s_\theta(X_t,t)-\bigl(-D^*U-\operatorname{div}V\bigr)
\right\rVert^2
\right],
$$

followed by the tangent-vector reconstruction described above. The target may
have a higher irreducible pathwise variance than the analytic Heat target;
that changes the observed loss floor, but not the ideal population minimiser.

The presence of `y_0` in `sample_and_score(rng, sde, y_0, t)` is therefore
intentional. This is a conditional DSM teacher whose two conditional
expectations recover the marginal score; it is not a teacher that directly
evaluates \(\nabla\log p_t\).

## 3. Theory-to-code mapping

| Theory | Code | Meaning |
|---|---|---|
| $X_0$ | `initial_point` / `y_0` | Initial data point, fixed while differentiating the endpoint map |
| $X_t$ | `endpoint` / `y_t` | GRW endpoint |
| $Z$ | `flat_noise` | Flattened independent standard Gaussian increments |
| $D_ZX_t=\partial X_t/\partial Z$ | `endpoint_jacobian` | Discrete endpoint derivative with respect to Gaussian noise |
| $J_{t,s}=\partial X_t/\partial X_s$ | not explicitly materialised | Flow Jacobian; distinct from `endpoint_jacobian` |
| $B^\top D_ZX_t$ | `tangent_jacobian` | Noise derivative represented in an orthonormal endpoint tangent basis |
| \(\Gamma\) | `covariance` | Discrete Malliavin covariance in tangent coordinates |
| $V_j=P_xe_j$ | `fields` | Tangent-by-construction endpoint vector fields |
| $U$ | `covering` | Regularised minimum-energy covering weights |
| $U^\top Z$ | `gaussian_pairing` | Gaussian pairing in the Skorokhod integral |
| \(\operatorname{div}_Z U\) | `covering_divergence` | Exact-JVP trace or Hutchinson trace estimate |
| \(D^*U=\delta(U)\) | `skorokhod` | `gaussian_pairing - covering_divergence` |
| \(-D^*U-\operatorname{div}V\) | `directional_score` | Pathwise directional score estimator |
| $T_{\mathrm{vec}}\in T_{X_t}S^2$ | `score_target` | Reconstructed tangent vector passed to upstream DSM |

The name `tangent_jacobian` is retained for code stability, but mathematically
it denotes $B^\top D_ZX_t$, not the flow Jacobian $J_{t,s}$.

## 4. Computational structure

`MalliavinTeacher` still requires `initial_point=y_0` because it estimates a
conditional transition score pathwise. Unlike Heat or Varadhan, it does not
need to evaluate an analytic approximation of
\(p_{t\mid0}(X_t\mid X_0)\). Instead it differentiates the simulated endpoint
with respect to its Gaussian increments and forms the Malliavin--Skorokhod
weight.

For $N$ GRW steps on $S^2$, the noise dimension is $3N$.

- `compute_endpoint_jacobian` is the only explicit `jax.jacrev` site. It
  constructs $D_ZX_t$.
- `compute_divergence_exact` evaluates \(\operatorname{div}_Z U\) with one
  basis-direction JVP per noise dimension, without constructing the full
  Jacobian of $U$.
- `compute_divergence_hutchinson` replaces that trace by probe-direction JVPs.
- The batch dimension is handled with `jax.vmap`, and the training step remains
  JIT-compatible.

A future marginal-only formulation might expose an interface such as
`sample_and_score(rng, sde, t)` and internally sample or integrate over the
initial distribution. That would be a different estimator and interface. The
current `sample_and_score(rng, sde, y_0, t)` interface is the correct one for
conditional DSM.

## 5. Implementation cautions

1. `endpoint_jacobian` means \(\partial\text{endpoint}/\partial Z\). It must
   never be interpreted as \(\partial X_t/\partial X_0\) or denoted by the
   flow-Jacobian symbol $J_{t,s}$.
2. `skorokhod` is \(\delta(U)=D^*U\). The symbol \(\delta\) is reserved for the
   Skorokhod operator in this documentation.
3. The target must remain tangent by construction. The implementation uses
   tangent vector fields and an orthonormal tangent basis; it does not train an
   unconstrained ambient vector and project it after the fact.
4. The estimator is exact for the chosen finite-dimensional Gaussian endpoint
   map when the covering equation is solved exactly. GRW discretisation,
   covariance regularisation, and an optional Hutchinson trace introduce their
   respective numerical approximation or variance.
5. A pathwise Malliavin target need not equal the Heat target at each sample.
   The score identities concern conditional expectations.
