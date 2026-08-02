+++
title= "~/"
template = "homepage.html"
+++

> When the present determines the future,
> but the approximate present does not approximately determine the future.
> <cite>Edward Lorenz</cite>

## About Me

Hello there! I'm Mario Daniel Panuco. I hold a Master's in Scientific Computing
and Applied Mathematics from UC Santa Cruz, which I came to by way of a Computer
Science Engineering undergrad.

<!-- I started out my academic journey in CSE. I picked it because of my long enamorment — or maybe fixation — with making computers do cool things. In my adolescence, for example, I was keen on video games and other digital media. But it was never the entertainment of the video game itself that took my attention. It was the development of it: the act of running a process, of putting a complex assembly of rock together, of vertically acute design integration between hardware, software, and ultimately the user. -->

<!-- This could be the origin of the first time I cogitated on the idea of computational physics: -->
<!-- what the contemporary constraints were in large physics simulations, -->
<!-- and what new problems scientists could take on in the future as certain computational boundaries were crossed. -->

As of late I've been drawn further into **Scientific Machine Learning (SciML)**, specifically
neural operators as surrogates for PDE solution maps. What holds my attention are the
numerical questions the surrogates inherit: how approximation error scales with
discretization and with sample count, what the universal-approximation for operators
guarantee and at what cost in width, depth, and retained modes, whether
autoregressive rollout is stable, and how to report accuracy on chaotic systems.
Where pointwise error is uninformative past a few Lyapunov times, the useful insight is in
how long a surrogate stayed on the attractor rather than the one-step residual.

My current work is on neural operators for chaotic physical systems. Kuramoto–Sivashinsky is
the testbed I keep returning to:

I'm particularly interested in:

- **Operator learning for multiphysics inverse problems** — computational imaging and wave
  reconstruction; neural operators standing in as forward maps inside an inference loop.
- **Inverse electrophysiology** — reconstructing cortical sources from implanted electrode
  arrays, and Mori–Zwanzig coarse-graining of the array's forward model.
- **Digital twins and control** — MHD field control for tokamak plasmas; uncertainty
  quantification for climate and ocean twins, where the twin is only worth something to a
  domain expert if it carries a calibrated error bar.
- **Standing interests** in computational fluid dynamics and computational genomics.

Elsewhere: [/projects](@/projects/_index.md) — a graph-diffusion risk-scoring
prototype, an HTTP server in C, and a genetic-algorithm-trained neural network in
Rust — and [/teaching](@/teaching.md) for my teaching experience.

## Contact

[mdpanuco@gmail.com](mailto:mdpanuco@gmail.com)
