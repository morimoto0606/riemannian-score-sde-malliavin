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
The implementation accepts a non-negative \(\lambda\); positive values provide
numerical stability while zero selects the unregularised comparison condition.

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

### SO(3) specialization

For the matrix representation of $SO(3)$, the implementation uses the same
normalized Lie-algebra basis as Geomstats' `random_normal_tangent`.  If
$\widehat e_i$ denotes this Frobenius-orthonormal basis, the endpoint frame is

$$
E_i(R)=R\widehat e_i,\qquad i=1,2,3.
$$

After flattening ambient $3\times3$ matrices, $E(R)$ has shape $9\times3$ and

$$
A=E(R_t)^*D_ZX_t\in\mathbb R^{3\times3N},\qquad
\Gamma=AA^*\in\mathbb R^{3\times3}.
$$

The endpoint fields are chosen as $V_j=E_j$.  Therefore $E^*V=I_3$ and the
regularized covering weights are

$$
U=A^*(\Gamma+\lambda I_3)^{-1}.
$$

The repository's matrix $SO(3)$ uses a bi-invariant metric.  Its left-invariant
orthonormal frame fields preserve the corresponding Riemannian/Haar volume and
have zero divergence.  Consequently the SO(3) directional estimator is simply

$$
T_j=-\delta(U_j),
$$

and the matrix-valued target is $\sum_jT_jE_j(R_t)$.  In particular, the
$S^2$ projected-coordinate formula $\operatorname{div}V_j=-2x_j$ is never used
on $SO(3)$.  Rao--Blackwellization remains an explicitly S2-only option.

### To check
- $V_i, i = 1,2 ,3$は3本あるが，tangent spaceは2次元なので，1つ余分ではないか？

  はい、点ごとには1本分が冗長である。ただし、これは意図的な冗長性で
  あり、3本のうち任意の1本を大域的に削除できるという意味ではない。
  行列としてまとめると

  $$
  V(x)=[V_1(x),V_2(x),V_3(x)]=P(x),
  \qquad \operatorname{rank}V(x)=2,
  $$

  であり、各点で

  $$
  \sum_{i=1}^3 x_iV_i(x)=P(x)x=0
  $$

  という1本の線形関係がある。一方、例えば固定して $V_1,V_2$ だけを
  使うと、$x=e_1$ では $V_1=0$ かつ $V_2=e_2$ となり、spanが1次元へ
  落ちる。どの2本を固定しても同様の退化点がある。これは $S^2$ 上で
  smoothな大域的orthonormal tangent frameを選べないこととも整合する。

  3本の projected coordinate fields は、tangent space上ではtight frameに
  なっている。実際、

  $$
  \sum_{i=1}^3 V_i(x)V_i(x)^\top
  =P(x)P(x)^\top=P(x),
  $$

  なので、$u\in T_xS^2$ に対して右辺は $u$ をそのまま返す。従って、
  3方向の真のdirectional scoreには冗長性があるが、情報の重複による
  scale factorは生じない。

  現在の実装はsingularな $3\times3$ 行列を逆行列にしているわけでは
  ない。$B^\top V=B^\top P=B^\top$ はshape $2\times3$、rank 2であり、
  逆行列を計算するMalliavin covarianceは

  $$
  \Gamma=(B^\top D_ZX_t)(B^\top D_ZX_t)^\top\in\mathbb R^{2\times2}
  $$

  である。最後に3成分のpathwise directional targetを
  $BB^\top=P$ でtangent spaceへ戻す。このとき
  $\operatorname{div}V=(-2x_1,-2x_2,-2x_3)^\top=-2x$ は
  $P(x)x=0$ により消え、

  $$
  P\bigl(-\delta(U)-\operatorname{div}V\bigr)
  =-P\delta(U)
  $$

  となる。したがって、3本を使う目的は大域的に退化しないsmoothな
  spanning familyを得ることであり、最終scoreの自由度は一貫して2で
  ある。局所chartごとに2本を選ぶ実装も原理上は可能だが、chart切替え、
  vector fieldのdivergence、および切替え境界での微分可能性を別途扱う
  必要がある。


--------------------
## 1.2 離散Mallaivin 解析

- 離散実装の $D_ZX_t$ は連続時間の $D_sX_t$ をどの程度近似しているか？

  **jax.jacrev**によるautodiff自体は、離散endpoint map

  $$
  F_N(Z_1,\ldots,Z_N)=X_t^{(N)}
  $$

  に対して、浮動小数点誤差を除けば
  $D_ZX_t^{(N)}=\partial F_N/\partial Z$をexactに計算している。従って、
  有限差分による微分近似誤差があるわけではない。近似なのは、微分対象の
  $F_N$が連続SDEの解写像$\Phi_t(W)$をGRWで離散化したものであることと、
  連続時間のnoise directionを有限個のGaussian incrementsで表している
  ことである。

  まず、両者はshapeとscaleが異なるので、そのまま行列差を取ることは
  できない。物理時間の区間を $I_k=[t_k,t_{k+1})$、長さを
  $\Delta t$ とし、標準Gaussian incrementを

  $$
  Z_k=\frac{W_{t_{k+1}}-W_{t_k}}{\sqrt{\Delta t}}
  $$

  とする。このとき、連続時間Malliavin derivativeをpiecewise-constantな
  Cameron--Martin基底

  $$
  \phi_k(s)=\frac{1_{I_k}(s)}{\sqrt{\Delta t}}
  $$

  へ射影した係数は

  $$
  \left\langle D_\cdot X_t,\phi_k\right\rangle_{L^2}
  =\frac{1}{\sqrt{\Delta t}}
   \int_{I_k}D_sX_t\,ds.
  $$

  `endpoint_jacobian`の第 $k$ block
  $D_{Z_k}X_t^{(N)}=\partial X_t^{(N)}/\partial Z_k$ が近似するのは
  この係数である。従って、pointwiseなMalliavin derivativeに対応する
  piecewise-constant reconstructionは

  $$
  \widehat D_s^{(N)}X_t
  =\frac{D_{Z_k}X_t^{(N)}}{\sqrt{\Delta t}},
  \qquad s\in I_k,
  $$

  である。$D_sX_t$には既に $\sqrt{\beta(s)}$ が含まれ、離散側でも
  endpoint mapのincrementに $\sqrt{\beta(t_k)\Delta t}$ が含まれるため、
  これとは別にbetaで割る必要はない。rawな $D_{Z_k}X_t^{(N)}$ は
  $O(\sqrt{\Delta t})$ なので、これを直接 $D_sX_t$ と比較すると必ず
  scaleの異なる比較になる。

  形式的には、$\Pi_N$を時間方向のpiecewise-constant projectionとすると、
  再構成した離散Malliavin derivativeの誤差は

  $$
  \widehat D^{(N)}X_t-DX_t
  =
  \left(\widehat D^{(N)}X_t-\Pi_NDX_t\right)
  +
  \left(\Pi_NDX_t-DX_t\right)
  $$

  と分けられる。第1項はGRW endpoint mapを微分したことによる
  differentiated-integrator error、第2項は時間方向のprojection errorで
  ある。実装上はさらに、$t_0+\mathrm{eps}$から積分を始めるtime truncation、
  $\beta(t)$のquadrature、covariance regularisation $\lambda$のbias、
  Hutchinson divergenceを使う場合のtrace推定分散が加わる。これらは
  autodiffがexactであっても消えない。

  現在のrepositoryには解析的な連続時間 $D_sX_t$との誤差測定はまだ
  実装されていない。`endpoint_max_abs_error`が確認しているのは、明示的な
  Gaussian-noise endpoint mapとnative upstream GRW endpointが同じ離散
  endpointを返すことであり、$N\to\infty$でのMalliavin derivativeの
  収束ではない。

  数値的に評価する場合は、同じBrownian pathからfine gridとcoarse gridを
  couplingする必要がある。$\Delta t_{\mathrm{coarse}}=m\Delta t_{\mathrm{fine}}$
  のとき、fine Gaussian incrementsから

  $$
  Z_k^{\mathrm{coarse}}
  =\frac{1}{\sqrt m}\sum_{\ell\in I_k}Z_\ell^{\mathrm{fine}}
  $$

  を作る。fine-grid derivativeをcoarse intervalへ平均したreferenceは

  $$
  \overline D_k^{\mathrm{ref}}
  =\frac{1}{m}\sum_{\ell\in I_k}
   \frac{D_{Z_\ell}X_t^{(N_{\mathrm{ref}})}}
        {\sqrt{\Delta t_{\mathrm{fine}}}}
  $$

  となる。coarse側の
  $D_{Z_k}X_t^{(N)}/\sqrt{\Delta t_{\mathrm{coarse}}}$ とこれを比較すれば、
  例えば

  $$
  E_D(N)^2
  =\sum_k\Delta t_{\mathrm{coarse}}
   \left\lVert
   \frac{D_{Z_k}X_t^{(N)}}{\sqrt{\Delta t_{\mathrm{coarse}}}}
   -\overline D_k^{\mathrm{ref}}
   \right\rVert_F^2
  $$

  という離散 $L^2([t_{\mathrm{start}},t])$ errorを測れる。ここで現在の
  upstream GRWでは $t_{\mathrm{start}}=t_0+\mathrm{eps}$ である。
  coarseとfineでendpointが異なる
  ため、intrinsicに比較するなら両者を同じendpoint tangent spaceへ
  parallel transportしてからnormを取る。embedded $S^2\subset\mathbb R^3$
  でambient Frobenius normを使う方法は簡単だが、endpoint errorも含む
  extrinsic errorになる。

  teacherに直接関係する、より安定な収束診断はMalliavin covarianceである。
  ambient representationでの離散共分散は

  $$
  \Gamma_N^{\mathrm{amb}}
  =D_ZX_t^{(N)}(D_ZX_t^{(N)})^*
  =\sum_kD_{Z_k}X_t^{(N)}(D_{Z_k}X_t^{(N)})^*,
  $$

  であり、連続時間の

  $$
  \Gamma^{\mathrm{amb}}
  =\int_{t_{\mathrm{start}}}^tD_sX_t(D_sX_t)^*\,ds
  $$

  のRiemann-sum近似になる。コード中の $2\times2$ `covariance`は
  $B^\top\Gamma_N^{\mathrm{amb}}B$である。この比較では $D_ZX_t$を
  $1/\sqrt{\Delta t}$ で再scaleする必要はない。実験では
  $N=25,50,100,200$などを共通Brownian path上で
  $N_{\mathrm{ref}}=800$または$1600$と比較し、$E_D(N)$、
  covarianceのrelative Frobenius error、固有値、condition numberを
  複数pathと複数$t$について報告するのが適切である。現在のGRWに対する
  収束次数はこの文書では仮定せず、log--log slopeから実測する必要がある。

  ただし、$D_ZX_t$と$\Gamma_N$の収束だけではteacher全体の妥当性確認として
  不十分である。実際にcoveringで使われるのは

  $$
  Q_N=(\Gamma_N+\lambda I)^{-1}B^\top V
  $$

  なので、小さい固有値方向の誤差がinverseによって増幅される可能性がある。
  fine-grid referenceとの

  $$
  \frac{\lVert Q_N-Q_{\mathrm{ref}}\rVert_F}
       {\lVert Q_{\mathrm{ref}}\rVert_F}
  $$

  に加えて、coveringが要求されたendpoint fieldsを再現するかを

  $$
  R_N=\lVert A_NU_N-B^\top V\rVert_F
  $$

  で確認する必要がある。tangent basis $B$はcoarse endpointとfine endpointで
  異なり得るため、比較時には同じtangent spaceへparallel transportするか、
  ambient operatorへ戻して比較する。

  さらに、最終teacherには

  $$
  \delta(U_N)=U_N^\top Z-\operatorname{div}_Z U_N
  $$

  が含まれる。divergenceは$U_N$のnoise derivativeを使うため、
  $D_ZX_t$と$\Gamma_N$が近いだけでは$\delta(U_N)$の収束までは保証されない。
  最終的には

  $$
  T_N=-P(X_t^{(N)})\delta(U_N)
  $$

  についても、coupled fine-grid referenceに対する誤差を調べる必要がある。
  ただし$T_N$は本来pathwise varianceが大きいので、単一pathの差だけでなく、

  - $\sigma(t)T_N$のnorm mean/std
  - coupled pathwise differenceのmean/quantile
  - endpoint近傍で平均したconditional target
  - conditional meanとHeat transition scoreの差

  を複数pathについて評価する。

  KNNによるconditional mean近似に依存しないend-to-end checkとして、
  smoothなtangent test vector field $G$に対するStein residual

  $$
  \mathcal R_G(N)
  =
  \mathbb E\!\left[
  \left\langle T_N,G(X_t^{(N)})\right\rangle
  +\operatorname{div}G(X_t^{(N)})
  \right]
  $$

  も利用できる。正しいscore weightなら$\mathcal R_G(N)=0$であり、複数の
  $G$、$t$、$N$についてresidualがzeroへ収束するかを確認できる。

  従って、$N=100$の現在のteacherを数値的に正当化するために必要な検証は
  少なくとも次の4段階である。

  1. $\widehat D_s^{(N)}X_t$の離散$L^2$自己収束
  2. $\Gamma_N$のrelative error、固有値、condition numberの収束
  3. regularised inverse action $Q_N$とcovering residual $R_N$の収束
  4. 最終teacher $T_N$のconditional meanまたはStein residualの収束

  discretisationとHutchinson errorを混同しないため、最初の収束実験では
  exact divergenceを使うか、少なくとも十分多いprobe数とshared probesで
  divergence noiseを固定する。その後、固定した$N$でprobe数を変えて
  Hutchinson errorを別に評価する。現在確認済みなのは同じ有限$N$での
  native upstream endpointとの一致までであり、上記の連続時間極限に対する
  数値的consistency checkは今後の検証項目である。


## 離散 SDE と Malliavin 微分に関する参考文献

本実装では、連続時間 SDE の Malliavin derivative $D_sX_T$ を直接計算するのではなく、
まず SDE を有限個の Gaussian increments によって離散化し、

$$
X_T^{(N)}
=
F_N(Z_1,\ldots,Z_N),
\qquad
Z_k
=
\frac{W_{t_{k+1}}-W_{t_k}}{\sqrt{\Delta t}},
$$

という有限次元 Gaussian noise から endpoint への写像を構成する。
その上で automatic differentiation により

$$
D_{Z_k}X_T^{(N)}
=
\frac{\partial F_N}{\partial Z_k}
$$

を計算する。

有限次元 Gaussian space 上では Malliavin derivative は Gaussian coordinates
に関する通常の gradient と同一視できるため、これは与えられた離散 scheme
に対する Malliavin derivative を計算していると解釈できる。
したがって automatic differentiation 自体は有限差分による近似ではなく、
離散 endpoint map $F_N$ の derivative を chain rule により計算している。

連続時間の Malliavin derivative との対応では、

$$
\phi_k(s)
=
\frac{\mathbf 1_{[t_k,t_{k+1})}(s)}
{\sqrt{\Delta t}}
$$

を Cameron--Martin 空間の離散基底とすると、

$$
\left\langle D_\cdot X_T,\phi_k\right\rangle_{L^2}
=
\frac{1}{\sqrt{\Delta t}}
\int_{t_k}^{t_{k+1}}D_sX_T\,ds.
$$

したがって

$$
\frac{\partial F_N}{\partial Z_k}
$$

は pointwise な $D_{t_k}X_T$ そのものではなく、
この区間に対応する Malliavin derivative の Gaussian-coordinate coefficient
を近似すると考えるのが適切である。piecewise-constant reconstruction は

$$
\widehat D_s^{(N)}X_T
=
\frac{1}{\sqrt{\Delta t}}
\frac{\partial F_N}{\partial Z_k},
\qquad
s\in[t_k,t_{k+1}),
$$

となる。

従って誤差は automatic differentiation そのものから生じるのではなく、

1. 連続 SDE を Euler scheme や GRW によって近似する誤差
2. Wiener noise を有限個の Gaussian increments に射影する誤差
3. Malliavin derivative を時間方向に piecewise-constant に表現する誤差

に由来する。

### 参考文献

1. **Detemple, J., Garcia, R. and Rindisbacher, M. (2005).**
   *Representation formulas for Malliavin derivatives of diffusion processes.*
   Finance and Stochastics, **9**, 349--367.
   DOI: 10.1007/s00780-004-0151-6.

   Diffusion process の Malliavin derivative の representation と数値計算を扱う。
   Euler discretization に基づく Malliavin derivative の近似との数値比較も行われており、
   離散化した diffusion process の Malliavin derivative を数値的に計算するという
   本研究の考え方に近い。

2. **Bally, V. and Talay, D. (1995).**
   *The Euler scheme for stochastic differential equations:
   error analysis with Malliavin calculus.*
   Mathematics and Computers in Simulation, **38**, 35--41.

   Euler scheme の誤差解析に Malliavin calculus を用いた研究。
   SDE の Euler approximation と Malliavin calculus の関係を理解する上で
   基礎的な文献である。

3. **Bally, V. and Talay, D. (1996).**
   *The law of the Euler scheme for stochastic differential equations:
   II. Convergence rate of the density.*
   Monte Carlo Methods and Applications, **2**, 93--128.

   Euler approximation の確率法則および density の収束を Malliavin calculus
   を用いて解析している。Euler scheme の Malliavin covariance や非退化性を
   利用して離散近似の density convergence を扱う代表的研究である。

4. **León, J. A., Liu, Y. and Tindel, S. (2024).**
   *Euler scheme for SDEs driven by fractional Brownian motions:
   Malliavin differentiability and uniform upper-bound estimates.*
   Stochastic Processes and their Applications, **175**.
   arXiv:2305.10365.

   Fractional Brownian motion 駆動 SDE を対象として、
   Euler scheme 自体の Malliavin differentiability を研究している。
   特に Euler scheme とその Malliavin derivatives に対する、
   step size に一様な pathwise bound を導出している。
   「連続 SDE を離散化した後、その離散 scheme を Malliavin 微分する」
   という本実装の考え方に直接関連する。

5. **León, J. A., Liu, Y. and Tindel, S. (2023).**
   *Euler scheme for SDEs driven by fractional Brownian motions:
   integrability and convergence in law.*
   arXiv:2307.06759.

   Euler scheme とその Malliavin derivatives の integrability を解析し、
   それを用いて Euler scheme の convergence in law を示している。
   離散 scheme の Malliavin derivative と連続時間極限との関係を考える際の
   参考文献となる。

6. **Tzen, B. and Raginsky, M. (2019).**
   *Neural Stochastic Differential Equations:
   Deep Latent Gaussian Models in the Diffusion Limit.*
   arXiv:1905.09883.

   Wiener space 上の stochastic automatic differentiation と
   black-box SDE solver を組み合わせた neural SDE の学習を扱う。
   Malliavin derivative の numerical convergence を直接研究した論文ではないが、
   discretized stochastic flow を automatic differentiation によって微分するという
   現代的な computational viewpoint に関連する。

### 本実装との位置づけ

既存研究から、SDE の numerical scheme 自体に Malliavin derivative を定義し、
その性質や連続時間極限を調べることは正当な数値 Malliavin calculus の枠組みである。

本実装では、この離散 Malliavin derivative を再帰式として陽に実装する代わりに、

$$
F_N:(Z_1,\ldots,Z_N)\longmapsto X_T^{(N)}
$$

という離散 endpoint map を automatic differentiation し、

$$
D^{(N)}X_T^{(N)}
=
\nabla_ZF_N
$$

を計算する。

したがって `jax.jacrev` の利用は、Malliavin derivative を別の微分概念で
置き換えているのではなく、有限次元 Gaussian approximation 上の
Malliavin derivative を automatic differentiation によって計算していると
解釈できる。

一方、今回用いている manifold 上の geodesic random walk (GRW) に対して、

$$
\nabla_ZF_N
\longrightarrow
DX_T
$$

がどのノルムでどの収束次数を持つかについては、
Euler--Maruyama に対する既存結果からそのまま従うとは限らない。
そのため、本研究では coupled fine/coarse Brownian increments を用いて

$$
\widehat D^{(N)}X_T,
\qquad
\Gamma_N,
\qquad
(\Gamma_N+\lambda I)^{-1},
\qquad
T_N
$$

の mesh refinement に対する numerical consistency を別途検証する。


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
