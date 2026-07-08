+++
title = "The Generative Sequel: Wavelet Diffusion Neural Operators"
description = "Part 4 of a series on the Fourier transform: from a PDE to a computational problem — operators, symbols, stiffness — and a toy wavelet diffusion neural operator that samples whole Burgers trajectories."
date = 2026-07-07
[taxonomies]
tags = ["math", "fourier", "wavelets", "jax", "machine-learning", "scientific-computing", "diffusion-models"]
+++

_This is part 4 of the series
([part 1](@/posts/post-1-origins/index.md): Euler's formula and Fourier series\;
[part 2](@/posts/post-2-dft-fft/index.md): the DFT, FFT, and spectral methods\;
[part 3](@/posts/post-3-wavelets-wno/index.md): wavelets and a minimal WNO).
Everything is reproducible from
[the repo](https://github.com/MarioDanielPanuco/Fourier-Transform):
`pixi run -e cuda wdno-train`, then `pixi run figs-post4`._

## Where it sits

Part 3 ended with the deterministic operator-learning picture — a table worth
re-drawing, because the subject of this post is the newest addition:

|          | transform              | middle                                    | output                                   |
| -------- | ---------------------- | ----------------------------------------- | ---------------------------------------- |
| FNO      | FFT                    | learned weights on low modes              | one solution field                       |
| WNO      | DWT                    | learned weights on coarse subbands        | one solution field                       |
| **WDNO** | DWT (space _and_ time) | **diffusion model** over all coefficients | a _distribution_ over whole trajectories |

The **Wavelet Diffusion Neural Operator**
([Hu et al., 2024](https://arxiv.org/abs/2412.04833)) stacks two separate upgrades
on the WNO, and it pays to keep them apart:

1. **Regression → generation.** FNO/WNO map an input function to a single point
   estimate, trained with a relative-L2 loss. WDNO instead learns a conditional
   _distribution_ $p\bigl(\mathcal{W}u_{[0,T]} \mid \mathcal{W}a\bigr)$ — a standard
   [DDPM](https://arxiv.org/abs/2006.11239) with a U-Net denoiser — over the full
   space-time trajectory, conditioned on the wavelet transform of the problem data
   $a$ (initial conditions, forcings). Generation is distinct from regression in two ways:
   it provides calibrated samples of chaotic dynamics, and gradient _guidance_, which
   turns the same trained model into a controller.

2. **Physical space → wavelet space.** The diffusion doesn't run on the raw field.
   The whole trajectory $u(t, x)$ is wavelet-transformed jointly in space and time,
   and the DDPM diffuses and denoises the complete coefficient vector. Diffusion models
   are known to smear abrupt changes and struggle with resolution transfer, while
   wavelet coefficients represent discontinuities _sparsely and locally_ — the hard
   content of the signal is concentrated in a few coefficients instead of spread across
   a global basis.

![The WDNO pipeline: pack the trajectory with a 2D DWT, run a conditional DDPM on the coefficients, inverse-transform the sample](01-wdno-pipeline.svg)

One disambiguation before anything else, because the name invites it: the
"diffusion" in WDNO is the **generative process** — noise gradually removed by a
learned denoiser. WDNO is a solver/controller
architecture, not a method for parabolic PDEs. We validate this claim via the application of WDNOs to non-diffusion, non-parabolic equations: 1D advection (hyperbolic), 1D Burgers (shock-forming), 1D _compressible_
Navier–Stokes, 2D incompressible flow, and the ERA5 weather record. What the method actually wants from a problem is structural:
trajectories on a regular grid, with sharp, localized, multiscale features, e.g, fronts,
shocks, filaments, etc, where a wavelet basis is sparse and a global basis isn't.
On smooth, slowly-varying dynamics it's accuracy is least differentiated from a
plain FNO\; on equations governed by shock-dominated dynamics, WDNOs stand out as the more advantageous method.

## From equation to computation

It's worth outlining the _analytical_ pipeline that
turns an equation into something a computer (or a neural operator) can chew on —
this is the machinery that underlies the generation of every training sample in
parts 3 and 4.

**The physical scenario.** Consider momentum transport in a one-dimensional fluid:
a velocity field $u(t, x)$ on a periodic domain that _advects itself_ (each parcel
is carried along at its own speed, so fast fluid overtakes slow fluid) while
molecular viscosity diffuses the differences away. The simplest mathematical model
of this competition is the **viscous Burgers equation**, the standard 1D caricature
of the Navier–Stokes momentum balance:

$$
\underbrace{\frac{\partial u}{\partial t}}\_{\text{evolution}}
\\;+\\; \underbrace{u\\, \frac{\partial u}{\partial x}}\_{\text{self-advection}}
\\;=\\; \underbrace{\nu\\, \frac{\partial^2 u}{\partial x^2}}\_{\text{viscous smoothing}},
\qquad x \in [0, 1),\\; \nu = 0.01 .
$$

Self-advection steepens smooth profiles into near-discontinuous fronts\; viscosity
arrests the steepening at a width set by $\nu$. That contest is what makes Burgers
the canonical test problem for shock-capturing methods — and for wavelet-based
operators.

The workflow from this model to a computation is three steps.

**Step 1 — split into operator and nonlinearity.** Move everything except the time
derivative to the right-hand side and sort the terms by linearity:

$$
\frac{\partial u}{\partial t}
= \underbrace{\nu\\, \partial_x^2 u}\_{\mathcal{L}u}
\\; \underbrace{-\\; u\\, \partial_x u}\_{\mathcal{N}(u)}
\qquad\text{i.e.}\qquad
u_t = \mathcal{L}u + \mathcal{N}(u),
$$

where $\mathcal{L}$ collects every _linear, constant-coefficient_ term — here
$\mathcal{L} = \nu\\,\partial_x^2$ — and $\mathcal{N}$ is whatever remains, here
$\mathcal{N}(u) = -u\\,u_x$. What kind of object is $\mathcal{L}$? An
**operator on function space**: it eats a function and returns a function, linearly
— a differential operator acting on periodic functions (densely defined on a
[Sobolev subspace](https://en.wikipedia.org/wiki/Sobolev_space) of $L^2$).
An infinite-dimensional matrix, in the same sense that part
2's DFT was a finite one.

**Step 2 — diagonalize the operator.** On a periodic domain the Fourier modes
$e^{ikx}$ are eigenfunctions of _every_ constant-coefficient differential operator
— the only fact this series has ever really used. Each derivative brings down one
factor of $ik$:

$$
\partial_x e^{ikx} = ik\\, e^{ikx}
\quad\Longrightarrow\quad
\mathcal{L} = \sum_j a_j\\, \partial_x^j
\\;\text{ acts as }\\;
\mathcal{L}\\,e^{ikx} = \ell(k)\\,e^{ikx},
\qquad
\ell(k) = \sum_j a_j (ik)^j .
$$

For the Burgers operator the computation is one line — $\mathcal{L} =
\nu\\,\partial_x^2$ has the single coefficient $a_2 = \nu$, so

$$
\ell(k) = \nu\\,(ik)^2 = -\nu k^2
$$

— real and negative: every mode decays, faster with frequency, which is the
spectral fingerprint of diffusion. The function $\ell(k)$ is called the **symbol**
of the operator. In Fourier coordinates, applying $\mathcal{L}$ is multiplication
of coefficient $k$ by the number $\ell(k)$: the operator has become a diagonal
matrix, exactly the move that solved the heat equation in part 1.

**Step 2b — the nonlinearity has no symbol.** The same trick fails on
$\mathcal{N}(u) = -u\\,u_x$: it is not linear, so it has no eigenbasis to share.
Two identities rescue it. First, the product rule rewrites it in _conservative
form_,

$$
u\\, u_x = \tfrac{1}{2}\\, \partial_x\bigl(u^2\bigr)
\quad\Longrightarrow\quad
\mathcal{N}(u) = -\tfrac{1}{2}\\, \partial_x\bigl(u^2\bigr),
$$

reducing the problem to "square the field, then differentiate." Second, each piece
is computed in the domain where it is cheap. Squaring in Fourier coordinates would
be a full convolution of coefficient sequences (part 2's convolution theorem read
in reverse, $\mathcal{O}(n^2)$)\; squaring in physical space is pointwise. So the
**pseudo-spectral** evaluation hops between domains:

$$
\widehat{\mathcal{N}(v)} = -\tfrac{1}{2}\\, ik \cdot
\mathcal{F}\Bigl[\bigl(\mathcal{F}^{-1}\hat{v}\bigr)^2\Bigr] \cdot m(k),
$$

inverse FFT, square, FFT, multiply by $ik$. The mask $m(k)$ is the **2/3 rule**:
the square of an $n$-mode signal contains $2n$ modes, and the unrepresentable half
folds back onto the grid as aliases (part 2's folding ruler)\; zeroing the top
third of the spectrum removes the corruption exactly for a quadratic term. This
five-operation recipe is the entire cost of nonlinearity in a spectral method.

The symbol and the nonlinearity together are all the solver ever needs to know
about the physics — which is why the repo's solver can treat the equation as data:

```python
@dataclass(frozen=True)
class Equation:
    name: str
    linear: Callable[[Grid], Array]              # grid -> l(k): the symbol
    nonlinear: Callable[[Array, Grid], Array] | None

burgers = Equation("burgers", lambda g: -nu * g.k**2, quadratic(-0.5))
kdv     = Equation("kdv",     lambda g: 1j * g.k**3,  quadratic(-3.0))
ks      = Equation("ks",      lambda g: g.k**2 - g.k**4, quadratic(-0.5))
```

| equation             | PDE                                   | symbol $\ell(k)$ | nonlinearity                   |
| -------------------- | ------------------------------------- | ---------------- | ------------------------------ |
| heat                 | $u_t = \nu u_{xx}$                    | $-\nu k^2$       | —                              |
| advection–diffusion  | $u_t + c u_x = \nu u_{xx}$            | $-ick - \nu k^2$ | —                              |
| Burgers              | $u_t + u u_x = \nu u_{xx}$            | $-\nu k^2$       | $-\tfrac{1}{2}\partial_x(u^2)$ |
| KdV                  | $u_t + 6u u_x + u_{xxx} = 0$          | $ik^3$           | $-3\\,\partial_x(u^2)$         |
| Kuramoto–Sivashinsky | $u_t + u u_x + u_{xx} + u_{xxxx} = 0$ | $k^2 - k^4$      | $-\tfrac{1}{2}\partial_x(u^2)$ |

**Step 3 — discretize and integrate.** Sampling $x$ on $n$ grid points truncates
the Fourier series to $n$ modes, and the PDE collapses into a system of ordinary
differential equations, one per coefficient — the
[method of lines](https://en.wikipedia.org/wiki/Method_of_lines). For Burgers:

$$
\frac{d\hat{v}_k}{dt} = -\nu k^2\\, \hat{v}_k
\\;-\\; \tfrac{1}{2}\\, ik\\, \widehat{\bigl(v^2\bigr)}_k ,
\qquad k = 0, \ldots, n/2 .
$$

One practical obstruction remains, and it is visible in the symbol itself:
[**stiffness**](https://en.wikipedia.org/wiki/Stiff_equation). Because $|\ell(k)|$
grows like $k^2$ for diffusion — $k^3$ for KdV's dispersion, $k^4$ for
Kuramoto–Sivashinsky — the highest resolved mode forces any fully explicit
time-stepper into steps of order $\Delta x^p$, absurdly small precisely when the
grid is fine. The remedy is an
[integrating factor](https://en.wikipedia.org/wiki/Exponential_integrator): the
linear part alone has the _exact_ solution $\hat{v}_k(t) = e^{\ell(k) t}\\,
\hat{v}_k(0)$, so a change of variables absorbs the stiff term analytically and
leaves only the benign nonlinear term to be stepped numerically. The repo uses an
integrating-factor RK4\; for a linear equation every stage vanishes and the scheme
degenerates to the exact propagator, which doubles as the solver's self-test
(`python -m ftx.spectral`).

That is the entire classical pipeline: **symbol + nonlinearity + exponential
integrator**. Here it is running four different physical regimes with the same
thirty lines — transport, shock formation, solitons overtaking each other,
spatiotemporal chaos — by swapping nothing but the `Equation` value:

![Four space-time heatmaps from the same integrator: advection-diffusion, Burgers, KdV solitons, and Kuramoto-Sivashinsky chaos](04-pde-gallery.png)

**Posing the learning problem.** A neural operator re-poses this pipeline as data.
The WNO of part 3 learned the _endpoint map_ $u(\cdot, 0) \mapsto u(\cdot, T)$: the
solver generated input–output pairs, and the operator amortized the 500 time-steps
between them into one forward pass. The WDNO changes the target: learn the
_distribution over whole trajectories_ $u(t, x)$, represented in wavelet
coefficients, conditioned on the wavelet transform of what you know (here, the
initial state). Same solver, same data — a different, strictly more ambitious
question asked of it.

## Simulation and control in one model

**Simulation** is conditional sampling: draw wavelet coefficients from the learned
conditional distribution and inverse-transform,

$$
\mathcal{W}u \sim p(\\,\cdot \mid \mathcal{W}a\\,),
\qquad
u = \mathcal{W}^{-1}\bigl(\mathcal{W}u\bigr).
$$

**Control** is the payoff of being generative. Suppose 1D Burgers with a forcing
term $f(t, x)$ that we are free to choose, and a target state $u^{\star}$ to be
reached at time $T$. A standard quadratic objective penalizes both the miss and the
actuation effort:

$$
J = \int_D \bigl| u(T, x) - u^{\star}(x) \bigr|^2\\, dx
\\;+\\; \alpha \int_{[0,T] \times D} \bigl| f(t, x) \bigr|^2\\, dt\\, dx .
$$

Guided sampling folds this objective into generation. Write $x^{(k)}$ for the noisy
wavelet-coefficient iterate at reverse-diffusion step $k$ and $c$ for the
conditioning\; each step subtracts the objective's gradient alongside the learned
score:

$$
x^{(k-1)} = x^{(k)}
\\;-\\; \eta\\,\Bigl(\epsilon_\theta\bigl(x^{(k)}, c, k\bigr)
\\;+\\; \lambda\\, \nabla J\bigl(\hat{x}_0^{(k)}\bigr)\Bigr) + \xi_k ,
$$

where $\epsilon_\theta$ is the trained denoiser, $\hat{x}_0^{(k)}$ is the standard
DDPM estimate of the clean sample at step $k$, $\lambda$ is the guidance weight,
and $\xi_k$ is the sampler's noise. The sampler _is_ the planner: no separate
policy network, no differentiating through a solver. The same weights simulate and
control.

## Multi-resolution training

The paper's second contribution attacks resolution generalization head-on instead
of hoping the operator inherits it. Build training pairs by downsampling —
$(N, N/2), (N/2, N/4), \ldots$, no extra fine-grid solves — and train _two_
diffusion models: a base model at the coarsest grid, and a super-resolution model
for

$$
p\bigl(\mathcal{W}u_{\mathrm{high}} \\;\big|\\; \mathcal{W}u_{\mathrm{low}},\\,
\mathcal{W}a_{\mathrm{high}}\bigr).
$$

At inference, sample coarse, then apply the
super-resolution model as many rungs up as you like — including resolutions never
seen in training. The wavelet representation is what makes the rung-to-rung map
_local_ (each fine coefficient depends on a neighborhood of coarse ones)\; their
ablation shows the same scheme in raw space-time degrades as super-resolution steps
stack.

## What the paper reports

Five systems, against raw-space DDPM, FNO, MWT, CNN, OFormer and others
(numbers transcribed from v3 of the paper — re-verify against the
[released code](https://github.com/AI4Science-WestlakeU/wdno) before quoting):

| system                              | WDNO   | best competitor | note                           |
| ----------------------------------- | ------ | --------------- | ------------------------------ |
| 1D advection, simulation (MSE)      | 2.9e-5 | DDPM 4.2e-5     | smooth — modest gap            |
| 1D Burgers, simulation (MSE)        | 1.4e-4 | DDPM 1.3e-4     | smooth-ish — a wash            |
| 1D compressible Navier–Stokes (MSE) | 0.22   | DDPM 5.52       | **25× — shocks are the story** |
| 2D incompressible fluid (MSE)       | 0.0023 | DDPM 0.016      | 7×                             |
| ERA5 weather (MSE)\*                | 12.83  | FNO 14.39       | real data                      |
| 2D smoke control (objective $J$)    | 0.068  | DDPM 0.312      | **78% less leakage**           |

<small>\* ERA5 is ECMWF's global atmospheric reanalysis (hourly estimates on a
0.25° latitude–longitude grid, 1979–present). The paper uses its **temperature
field**, with the task of predicting the next 20 hours of evolution from the
preceding 12 hours of states.</small>

The pattern is the series' through-line wearing a new coat: where the solution is
smooth, a global basis is fine and wavelets buy little (the Burgers row)\; where the
state carries fronts and shocks, the local basis dominates (the compressible NS
row). Their Fourier-domain ablation — the identical diffusion pipeline with an FFT
in place of the DWT — is "significantly inferior" on the shock-heavy system.

## Crux

[`src/ftx/wdno/`](https://github.com/MarioDanielPanuco/Fourier-Transform) implements
the simulation in a few hundred lines of JAX Python, grown out of
part 3's WNO. The pipeline, per file:

- **`data.py`** — the same pseudo-spectral Burgers solver (now the shared
  `ftx.spectral` module), but keeping the whole rollout: $M$ trajectories of
  shape $(N_t \times N_x)$, where $N_t$ counts saved time frames and $N_x$ is the
  saved spatial resolution. For this post, $M = 2{,}064$ trajectories of shape
  $32 \times 64$.

- **`wavelets2d.py`** — the transform. The 1D Haar DWT is the simplest orthonormal
  wavelet transform: it replaces each adjacent pair of samples by its normalized
  average and difference,

$$
a_i = \frac{u_{2i} + u_{2i+1}}{\sqrt{2}},
\qquad
d_i = \frac{u_{2i} - u_{2i+1}}{\sqrt{2}},
$$

halving the resolution while retaining exact invertibility: $(a, d)$ carries the
same information as $u$, reorganized into a coarse approximation plus the detail
needed to reconstruct it. The 2D transform is separable — apply the 1D transform
along $t$, then along $x$ — so one level yields four subbands: the approximation
(both axes averaged), two mixed subbands (detail in $t$, average in $x$, and vice
versa), and the diagonal detail. Recursing on the approximation for $L$ levels
gives part 3's multiresolution pyramid, now in two dimensions. Two design choices
deserve justification. _Why transform $(t, x)$ jointly?_ Because the structures
that make PDE trajectories hard — a shock line moving through the plane — are
localized in space **and** time together\; 2D wavelet atoms are localized in both
coordinates at every scale, so the shock line touches only a few coefficients per
level. _Why Haar?_ With periodic wrapping, Haar halves each axis exactly (no
boundary padding), so a level-2 decomposition of a $32 \times 64$ trajectory
packs into the classic nested layout of exactly $32 \times 64$ — the denoiser can
be an ordinary image U-Net over the coefficient plane. (The paper uses smoother
biorthogonal bases, `bior2.4`/`bior1.3`\; Haar keeps the toy's shape bookkeeping
trivial at some cost in coefficient sparsity.)

![A Burgers trajectory heatmap next to its packed 2D Haar coefficient image](02-packed-haar.png)

- **`unet.py`** — a small NHWC U-Net (806k parameters): two downsamplings, residual
  blocks with GroupNorm, sinusoidal timestep embeddings.
- **`diffusion.py`** — vanilla DDPM: 300 steps, linear β schedule, noise-prediction
  loss, ancestral sampler under `jax.lax.scan`.
- **`train.py`** — packs every trajectory once, rescales the coefficients, and runs
  standard DDPM training conditioned on the initial state:

```text
# preparation
W    <- dwt2_packed(trajectories)                  # (M, 32, 64) coefficient images
C    <- dwt2_packed(u0 tiled across the Nt frames) # conditioning images
s    <- std(W, axis=0), clipped away from 0        # per-coefficient scale map
W, C <- W / s,  C / s

# training loop
repeat:
    x0, c <- random minibatch from (W, C)
    k     ~  Uniform{0, ..., 299}                  # noise level
    eps   ~  N(0, I)
    xk    <- sqrt(abar_k) * x0 + sqrt(1 - abar_k) * eps
    loss  <- || UNet(xk, c, k) - eps ||^2          # predict the injected noise
    adam update on UNet parameters

# sampling (simulation)
x ~ N(0, I)
for k = 299 down to 0:
    x <- ancestral DDPM step using UNet(x, c, k)
return idwt2_packed(x * s)
```

The per-coefficient normalization (`s` above, equivalently per-subband for the
packed layout) deserves its own paragraph, because at first sight it appears to
fight the compression story — if the wavelet transform is valuable for
concentrating the signal in few coefficients, why flatten the scales back out? The
resolution: the concentration is real, and it is exactly _why_ normalization is
needed. An orthonormal Haar step maps a locally smooth pair to
$a = (u_{2i} + u_{2i+1})/\sqrt{2} \approx \sqrt{2}\\, u$ — the approximation
coefficient _grows_ by $\sqrt{2}$ per averaging, per axis. After two levels in two
dimensions the coarse block carries coefficients roughly $4\times$ the raw field's
scale, while the fine-detail coefficients of a smooth region sit near zero. For
_compression_ (part 3, or an SVD truncation) that disparity is the entire point:
keep the few large coefficients, discard the rest. For _generation_ it is a
hazard: the DDPM corrupts every coefficient with noise of the same unit scale, so
without rescaling, a given noise level $k$ barely perturbs the coarse block while
having long since drowned the fine subbands — the denoiser would learn the coarse
structure and write off the details (precisely the shock information the wavelet
basis was chosen to preserve) as noise. Dividing by the per-coefficient standard
deviation equalizes the signal-to-noise schedule across subbands\; the scale map is
reapplied before the inverse transform. Compression discards small coefficients\;
generation must _model_ them — the two uses pull in opposite directions, and the
normalization is what reconciles the shared representation with the second use.

![Training curve of the WDNO](03-wdno-loss.png)

Sixteen held-out initial conditions, one DDPM sample each (300 denoising steps, ~2 s
total for all sixteen):

![WDNO samples against the pseudo-spectral solver as space-time heatmaps, with the error concentrated along the shock line](05-wdno-samples.png)

Mean whole-trajectory rel-L2 ≈ **18%**, with the error visibly concentrated along
the moving shock — the sample nails the global transport and diffuses about the
exact front position, which is the honest failure mode for a generative model this
size. Two caveats before comparing that to the WNO's 9.7%: this metric covers the
_whole trajectory_ (32 frames, easy early ones and hard late ones) versus the WNO's
single endpoint\; and one diffusion sample is a draw from a distribution, not a
posterior mean — averaging several samples per input lowers the number.

## Benchmarks: what the GPU actually buys

All numbers in this section are measured on 1D viscous Burgers — the same system
as everything above — on one machine (Ryzen 9800X3D, RTX 5080, JAX on WSL2), via
`pixi run [-e cuda] wno-bench` and `wdno-bench`. The WNO benchmark trains the
endpoint operator on the 256-point grid at batch 32\; the WDNO benchmark trains
the U-Net denoiser on $32 \times 64$ packed coefficient images at batch 16.
Throughput is measured without per-step host synchronization (see below).

| benchmark (1D Burgers)         | CPU   | RTX 5080 | speedup |
| ------------------------------ | ----- | -------- | ------- |
| WNO training, steps/s          | 35.8  | 199.2    | 5.6×    |
| WNO inference, inputs/s        | 2,954 | 116,735  | ~40×    |
| WDNO (U-Net) training, steps/s | 9.5   | 183.9    | 19×     |
| WDNO sampling, trajectories/s  | 2.0   | 3.8      | 1.9×    |

Three observations, each of which cost us a wrong belief.

First, **instrument before you conclude**. An earlier draft of part 3 claimed the
WNO trained in a "dead heat" between CPU and GPU. That claim rested on two
artifacts. The training script's loss history is appended as a Python float every
step — a device-to-host synchronization that stalls the GPU pipeline each
iteration (of the GPU's 104 s end-to-end training time, roughly 44 s is this
stall\; the same run at benchmark throughput would take ~60 s). And the training
figure's "on CPU" label turned out to be hardcoded — the original run it described
was almost certainly a GPU run. Both are fixed: metrics now record
`jax.default_backend()`, and the honest end-to-end comparison is **104 s (GPU)
versus 339 s (CPU)** for the full 12,000 steps.

Second, **arithmetic intensity, not parameter count, sets the speedup**. The
WNO's layer is a chain of small sequential DWT convolutions and per-subband
einsums — dozens of kernel launches that each do little arithmetic, sensitive to
launch latency (which WSL2 does not help). It still earns 5.6× in training and
~40× in batched inference, where the whole test set is one launch-amortized
forward pass. The WDNO's U-Net is wide convolutions — real work per launch — and
reaches 19× in training. Its _sampling_, by contrast, gains only 1.9×: 300
strictly sequential denoiser calls at batch 16 leave the GPU underfed\; batching
more trajectories per sampling pass would recover most of the gap.

Third, **the equation shapes the balance sheet**. These numbers are for a small 1D
system. Moving to 2D fields (the paper's smoke-control setting) multiplies the
work per kernel and widens every GPU margin\; adding wavelet decomposition levels
or longer filters does the opposite, multiplying small sequential kernels.
Trajectory _generation_ itself — `lax.scan` over thousands of RK4 steps — is
sequential in the same way diffusion sampling is, and benefits from the GPU mainly
through `vmap` over many initial conditions at once. Where each equation in the roadmap below lands on this
spectrum is one of the things to measure as it is added, alongside accuracy.

## Where to take it

The point of making the solver equation-agnostic is that every next experiment is
now a few lines: pass a different symbol and nonlinearity, regenerate trajectories,
retrain. A roadmap in rough order of what each problem would prove:

- **KdV** — solitons and dispersive shocks\; sharp coherent structures that travel
  and interact, ideal for validating super-resolution on localized features.
- **Kuramoto–Sivashinsky** — spatiotemporal chaos with cellular structure\; does a
  generative solver capture chaotic-but-structured statistics?
- **Compressible Euler (Sod tube)** — genuine discontinuities: contacts,
  rarefactions, shocks\; the regime where the paper's 25× lives.
- **FitzHugh–Nagumo** — traveling excitation waves with steep fronts\; the guided
  control story maps naturally onto spiral-wave suppression (defibrillation).
- **Nonlinear Schrödinger** — optical solitons and rogue-wave statistics\; the
  signals crossover.

The first two already ship in `ftx.spectral` (the gallery above is them). Each
addition will follow the template this post established for Burgers: physical
scenario → PDE → symbol and nonlinearity → discretization → trained operator →
benchmark. What's missing versus the paper — and the natural next build — is
**control**: add a forcing channel to the dataset and the guided update above\;
1D Burgers control is exactly the paper's first benchmark.

The through-line of the whole series, one last time: _find the basis that makes
your operator simple, act there, come back._ Euler's formula supplied the basis\;
the FFT made the round trip cheap\; wavelets rebuilt the basis for a world with
edges\; neural operators learned the middle from data\; and the WDNO makes the
middle a distribution you can sample and steer.

### Further reading

- Hu et al., ["Wavelet Diffusion Neural Operator"](https://arxiv.org/abs/2412.04833)
  (2024) — the paper this post follows\; [code](https://github.com/AI4Science-WestlakeU/wdno).
- Ho, Jain & Abbeel, ["Denoising Diffusion Probabilistic Models"](https://arxiv.org/abs/2006.11239)
  (2020) — the DDPM machinery used verbatim here.
- Kovachki et al., ["Neural Operator: Learning Maps Between Function Spaces"](https://arxiv.org/abs/2108.08481)
  (2021) — the general theory behind parts 3 and 4.
- Kassam & Trefethen, ["Fourth-order time-stepping for stiff PDEs"](https://doi.org/10.1137/S1064827502410633)
  (2005) — the classic on exponential integrators for exactly the KdV/KS setups above.
- Mallat, _A Wavelet Tour of Signal Processing_ — still the reference for everything
  the DWT does here.
