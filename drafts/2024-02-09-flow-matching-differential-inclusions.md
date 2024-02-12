---
layout: post
title:  Flow matching and differential inclusions
date:   2024-02-09 10:00
description: An interesting connection between flow matching, differential inclusions, hybrid dynamical systems, and discontinuous dynamical systems..
tags: flow-matching, generative-models, control-theory
---


Generative modeling is a fundamental concept in machine learning, where one typically wants to create a model that can generate samples from some complex distribution.
For example, one might want to generate pictures of golden retrievers given a large dataset of golden retriever pictures.
In this case, the dataset of golden retriever pictures is theoretically considered to be a set of samples from the distribution of all possible golden retriever pictures.
The generative model then learns to generate samples from this distribution.

In recent years, diffusion models have become the state-of-the-art model for generative modeling of images.
These types of models rely on continuous-time dynamics in the form of a stochastic differential equation (SDE):

\begin{equation}\label{eq:sde}
    \mathrm{d} \mathbf{x} = \mathbf{f}(\mathbf{x}, t) \mathrm{d}t + \mathbf{g}(\mathbf{x}, t) \mathrm{d}\mathbf{w},
\end{equation}

where $$\mathbf{x} \in \mathbb{R}^n$$ is the state of the system, $$\mathbf{f}: \mathbb{R}^n \times \mathbb{R}_{\geq 0} \to \mathbb{R}^n$$ is the drift term, $$\mathbf{g}: \mathbb{R}^n \times \mathbb{R}_{\geq 0} \to \mathbb{R}^{n \times n}$$ is the diffusion term, and $$\mathbf{w} \in \mathbb{R}^n$$ is a Wiener process.

The dynamics of the probability density function (PDF) $$p: \mathbb{R}^n \times \mathbb{R}_{\geq 0} \to \mathbb{R}_{\geq 0}$$ of the state of the system is described by the Fokker-Planck-Kolmogorov (FPK) equation:

\begin{equation}\label{eq:fokker-planck}
    \frac{\partial p(\mathbf{x}, t)}{\partial t} = -\nabla_\mathbf{x} \cdot \left( \mathbf{f}(\mathbf{x}, t) p(\mathbf{x}, t) \right) + \frac{1}{2} \Delta_\mathbf{x} \left( \mathbf{g}(\mathbf{x}, t) \mathbf{g}(\mathbf{x}, t)^\top p(\mathbf{x}, t) \right).
\end{equation}

Song et al. and Mautsa et al. showed that the FPK equation can be written in the form of a continuity equation:
\begin{equation}\label{eq:continuity}
    \frac{\partial p(\mathbf{x}, t)}{\partial t} = -\nabla_\mathbf{x} \cdot \left(\mathbf{v}(\mathbf{x}, t) p(\mathbf{x}, t)\right) 
\end{equation}
where the vector field $$\mathbf{v}: \mathbb{R}^n \times \mathbb{R}_{\geq 0} \to \mathbb{R}^n$$ is given by
\begin{equation}\label{eq:fpk-vector-field}
    \mathbf{v}(\mathbf{x}, t) = \mathbf{f}(\mathbf{x}, t) - \frac{1}{2} \nabla_\mathbf{x} \cdot \left(\mathbf{g}(\mathbf{x}, t) \mathbf{g}(\mathbf{x}, t)^\top\right) - \frac{1}{2} \mathbf{g}(x, t) \mathbf{g}(\mathbf{x}, t)^\top \nabla_\mathbf{x} \log p(\mathbf{x}, t),
\end{equation}
where $$\nabla_\mathbf{x} \log p(\mathbf{x}, t)$$ is well-known "score function" the one tries to learn in diffusion models.

Recently, **flow matching** has been proposed as an alternative to diffusion models for generative modeling, where instead of assuming the structure of the FPK vector field (\ref{eq:fpk-vector-field}), one considers any curl-free vector field $$\mathbf{v}$$. Why curl-free? Becuase, according to the Helmholtz decomposition theorem, any vector field can be decomposed into a curl-free and a divergence-free component. The divergence-free component would not affect the behavior of the PDF in the continuity equation (\ref{eq:continuity}), thus there are an infinite number of divergence-free components that can be added to the vector field $$\mathbf{v}$$ with the same behavior.

Flow matching relies on the concept of continuous normalizing flows (CNFs), which are composed of a vector field $$\mathbf{v}: \mathbb{R}^n \times \mathbb{R}_{\geq 0} \to \mathbb{R}^n$$, a flow map $$\mathbf{\psi}: \mathbb{R}^n \times \mathbb{R}_{\geq 0} \to \mathbb{R}^n$$, and a PDF $$p: \mathbb{R}^n \times \mathbb{R}_{\geq 0} \to \mathbb{R}_{\geq 0}$$, such that $$(\mathbf{v}, \mathbf{\psi})$$ define an ordinary differential equation (ODE)

\begin{equation}\label{eq:ode}
    \frac{\mathrm{d} \mathbf{\psi}(\mathbf{x}, t)}{\mathrm{d} t} = \mathbf{v}(\mathbf{\psi}(\mathbf{x}, t), t),
\end{equation}

and $$(\mathbf{v}, p)$$ define a continuity equation

\begin{equation}\label{eq:continuity-2}
    \frac{\partial p(\mathbf{x}, t)}{\partial t} = -\nabla_\mathbf{x} \cdot \left(\mathbf{v}(\mathbf{x}, t) p(\mathbf{x}, t)\right).
\end{equation}

## Continuous Normalizing Flow (CNF)

A **CNF** is essentially a triplet $$(\mathbf{v}, \mathbf{\psi}, p)$$ composed of a vector field $$\mathbf{v}: \mathbb{R}^n \times \mathbb{R}_{\geq 0} \to \mathbb{R}^n$$, a flow map $$\mathbf{\psi}: \mathbb{R}^n   \times \mathbb{R}_{\geq 0} \to \mathbb{R}^n$$, and a PDF $$p: \mathbb{R}^n \times \mathbb{R}_{\geq 0} \to \mathbb{R}_{\geq 0}$$, such that $$(\mathbf{v}, \mathbf{\psi})$$ define an ordinary differential equation (ODE) and $$(\mathbf{v}, p)$$ define a continuity equation:

$$
  \begin{aligned}
    \frac{\mathrm{d} \mathbf{\psi}(\mathbf{x}, t)}{\mathrm{d} t} &= \mathbf{v}(\mathbf{\psi}(\mathbf{x}, t), t) \qquad \text{(ODE)}\\
    \frac{\partial p(\mathbf{x}, t)}{\partial t} &= -\nabla_\mathbf{x} \cdot \left(\mathbf{v}(\mathbf{x}, t) p(\mathbf{x}, t)\right) \qquad \text{(Continuity equation)}
  \end{aligned}
$$

### Conditional CNF

In flow matching, one defines a **conditional CNF** $$(\mathbf{v}(\cdot \mid \mathbf{x}_1), \mathbf{\psi}(\cdot \mid \mathbf{x}_1), p(\cdot \mid \mathbf{x}_1))$$ such that

$$
  \begin{aligned}
    \frac{\mathrm{d} \mathbf{\psi}(\mathbf{x}, t \mid \mathbf{x}_1)}{\mathrm{d} t} &= \mathbf{v}(\mathbf{\psi}(\mathbf{x}, t \mid \mathbf{x}_1), t \mid \mathbf{x}_1) \qquad \text{(ODE)}\\
    \frac{\partial p(\mathbf{x}, t \mid \mathbf{x}_1)}{\partial t} &= -\nabla_\mathbf{x} \cdot \left(\mathbf{v}(\mathbf{x}, t \mid \mathbf{x}_1) p(\mathbf{x}, t \mid \mathbf{x}_1)\right) \qquad \text{(Continuity equation)} \\
    \exists T \in \mathbb{R}_{\geq 0} \quad &\text{s.t.} \quad \lim_{t \to T} p(\mathbf{x}, t \mid \mathbf{x}_1) \approx \delta(\mathbf{x} - \mathbf{x}_1) \quad \text{(Data hitting)}
  \end{aligned}
$$

### Marginal CNF

We then assume that there is a **marginal CNF** $$(\mathbf{v}, \mathbf{\psi}, p)$$ such that its PDF is a mixture of the conditional CNF's PDF:

$$
  \begin{aligned}
      \frac{\mathrm{d} \mathbf{\psi}(\mathbf{x}, t)}{\mathrm{d} t} &= \mathbf{v}(\mathbf{\psi}(\mathbf{x}, t), t) \qquad \text{(ODE)}\\
    \frac{\partial p(\mathbf{x}, t)}{\partial t} &= -\nabla_\mathbf{x} \cdot \left(\mathbf{v}(\mathbf{x}, t) p(\mathbf{x}, t)\right) \qquad \text{(Continuity equation)} \\
    p(\mathbf{x}, t) &\approx \int p(\mathbf{x} \mid \mathbf{x}_1) q_1(\mathbf{x}_1) \mathrm{d} \mathbf{x}_1 \qquad \text{(Marginal PDF)} \\
    \mathbf{v}(\mathbf{x}, t) &\approx \frac{1}{p(\mathbf{x}, t)} \int \mathbf{v}(\mathbf{x}, t \mid \mathbf{x}_1) p(\mathbf{x}, t \mid \mathbf{x}_1) q_1(\mathbf{x}_1) \mathrm{d} \mathbf{x}_1 \qquad \text{(Marginal VF)}
  \end{aligned}
$$
