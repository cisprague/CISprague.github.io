---
layout: post
title:  Flow matching and differential inclusions
date:   2024-02-09 10:00
description: An interesting connection between flow matching, differential inclusions, hybrid dynamical systems, and discontinuous dynamical systems..
tags: flow-matching, generative-models, control-theory
---


[Generative modeling](https://en.wikipedia.org/wiki/Generative_model) is a fundamental concept in machine learning, where you typically want to create a model that can generate samples from some complex distribution (e.g. golden retreiver images).
Recently, a new type of generative modelling framework, called [flow matching](https://arxiv.org/abs/2210.02747), has been proposed as an alternative to [diffusion models](https://arxiv.org/abs/2011.13456) that enjoys fast training and sampling.
Flow matching relies on the framework of [continuous normalizing flows (CNFs)](https://arxiv.org/abs/1806.07366), where you learn a model of a time-dependent [vector field](https://en.wikipedia.org/wiki/Vector_field) $$\mathbf{v}: \mathbb{R}^n \times \mathbb{R}_{\geq 0} \to \mathbb{R}^n$$ and then use it to transport samples from a simple distribution $$q_0$$ (e.g. a [normal distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution)) to samples of a more complex distribution $$q_1$$ (e.g. golden retreiver images), like I have shown below with my fantastic artistic skills...

<figure style="text-align: center;">
  <img src="../../../assets/img/transport.jpg" alt="Probability distributions" title="Probability distributions" width="90%">
  <figcaption><strong>Figure</strong>: The transportation of a simple distribution \(q_0\) to a more complex distribution \(q_1\).</figcaption>
</figure>

Theoretically, this transportation of samples obeys the well-known [continuity equation](https://en.wikipedia.org/wiki/Continuity_equation) from physics:

$$
  \frac{\partial p(\mathbf{x}, t)}{\partial t} = -\nabla_\mathbf{x} \cdot \left(\mathbf{v}(\mathbf{x}, t) p(\mathbf{x}, t)\right), \qquad \text{(Continuity)}
$$

where $$p : \mathbb{R}^n \times \mathbb{R}_{\geq0} \to \mathbb{R}_{\geq 0}$$ is a time-dependent "density" and $$\nabla_\mathbf{x} \cdot$$ is the [divergence operator](https://en.wikipedia.org/wiki/Divergence). This equation essentially says that the total mass of the system does not change over time.
In our case, this "mass" is just the total probability of the space, which aligns with the property of [probability density functions (PDFs)](https://en.wikipedia.org/wiki/Probability_density_function) that $$\int_{\mathcal{X}} p(\mathbf{x}, t) \mathrm{d}\mathbf{x} = 1$$. See another fantastic art piece below for an illustration.

<figure style="text-align: center;">
  <img src="../../../assets/img/continuity.jpg" alt="Probability distributions" title="Probability distributions" width="90%">
  <figcaption><strong>Figure</strong>: The continuity equation preserves the total mass of the PDF!</figcaption>
</figure>

So if the vector field and PDF $$(\mathbf{v}, p)$$ satisfy the continuity equation, then we can say that $$\mathbf{v}$$ generates $$p$$.
This also means that the well-known push-forward equation is satisfied

$$
  p(\mathbf{x}, t) = p(\mathbf{\phi}^{-1}(\mathbf{x}, t), 0) \det \left(\nabla_\mathbf{x} \mathbf{\phi}^{-1}(\mathbf{x}, t)) \right), \qquad \text{(Push-forward)}
$$

where $$\mathbf{\phi} : \mathbb{R}^n \times \mathbb{R}_{\geq 0} \to \mathbb{R}^n$$ is a flow map (or integral curve) of $$\mathbf{v}$$ starting from $$\mathbf{x}$$, defining an ordinary differential equation (ODE):

$$
  \frac{\mathrm{d} \mathbf{\phi}(\mathbf{x}, t)}{\mathrm{d} t} = \mathbf{v}(\mathbf{\phi}(\mathbf{x}, t), t), \qquad \text{(ODE)}
$$

$$\mathbf{\phi}^{-1}(\mathbf{x}, t)$$ is its inverse with respect to $$\mathbf{x}$$, and $$\nabla_\mathbf{x} \mathbf{\phi}^{-1}(\mathbf{x}, t)$$ is the Jacobian matrix  with respect to $$\mathbf{x}$$ of its inverse.
Essentially, this is just a theoretical justification saying that we can sample $$\mathbf{x}_0 \sim q_1$$ and then compute $$\mathbf{x}_1 = \mathbf{\phi}(\mathbf{x}_0, T)$$ through numerical integration of $$\mathbf{v}$$ starting from $$\mathbf{x}_0$$ to get a sample from the complex distribution $$\mathbf{x}_1 \sim q_1$$.
See another drawing below!

<figure style="text-align: center;">
  <img src="../../../assets/img/flow_map.jpg" width="90%">
  <figcaption><strong>Figure</strong>: The flow map from \(t = 0\) to \(t = T\).</figcaption>
</figure>

Okay, but there is one big problem here: how do we actually learn a model of such a vector field  $$\mathbf{v}$$ if we only have samples from the simple and complex PDFs, $$q_0$$ and $$q_1$$?
Well, the [flow matching (FM)](https://arxiv.org/abs/2210.02747) authors proposed learning from intermediate samples of a conditional PDF $$p(\mathbf{x}, t \mid \mathbf{x}_1)$$ that converges to a concentrated PDF (i.e. a [Dirac delta distribution](https://en.wikipedia.org/wiki/Dirac_delta_function) $$\delta$$) around each data sample $$\mathbf{x}_1 \sim q_1$$ such that it locally emulates the desired PDF (see drawing below), i.e.

$$
\exists T \in \mathbb{R}_{\geq 0} \quad \text{s.t.} \quad \lim_{t \to T} p(\mathbf{x}, t \mid \mathbf{x}_1) \approx \delta(\mathbf{x} - \mathbf{x}_1). \qquad \text{(Emulation)}
$$

<figure style="text-align: center;">
  <img src="../../../assets/img/conditional_pdf.jpg" width="90%">
  <figcaption><strong>Figure</strong>: \(q_0\) being transported to \(\delta(\mathbf{x} - \mathbf{x}_1)\) for each \(\mathbf{x}_1 \sim q_1\).</figcaption>
</figure>

Just like before, the conditional PDF $$p(\mathbf{x}, t \mid \mathbf{x}_1)$$ also has a vector field $$\mathbf{v}(\mathbf{x}, t \mid \mathbf{x}_1)$$ that generates it, so it also has a continuity equation:

$$
  \frac{\partial p(\mathbf{x}, t \mid \mathbf{x}_1)}{\partial t} = -\nabla_\mathbf{x} \cdot \left(\mathbf{v}(\mathbf{x}, t \mid \mathbf{x}_1) p(\mathbf{x}, t \mid \mathbf{x}_1)\right). \qquad \text{(Conditional continuity)}
$$


The FM authors then make the assumption that the desired PDF can be constructed by a "mixture" of the conditional PDFs:

$$
  p(\mathbf{x}, t) = \int_{\mathcal{X}_1} p(\mathbf{x} \mid \mathbf{x}_1) q_1(\mathbf{x}_1) \mathrm{d} \mathbf{x}_1 = \underset{\substack{\mathbf{x}_1 \sim q_1}}{\mathbb{E}} \left[p(\mathbf{x}, t \mid \mathbf{x}_1)\right], \qquad \text{(Marginal PDF)}
$$

where we call the desired PDF $$p(\mathbf{x}, t)$$ a marginal PDF. 
With this assumption, they are then able to identify a **marginal vector field (VF)** using the identities in parantheses:

$$
\begin{aligned}
  \frac{\partial p(\mathbf{x}, t)}{\partial t} &= \frac{\partial}{\partial t} \int_{\mathcal{X}_1} p(\mathbf{x} \mid \mathbf{x}_1) q_1(\mathbf{x}_1) \mathrm{d} \mathbf{x}_1 \qquad \small{\text{(Marginal PDF)}} \\
  &= \int_{\mathcal{X}_1} \frac{\partial p(\mathbf{x} \mid \mathbf{x}_1)}{\partial t} q_1(\mathbf{x}_1) \mathrm{d} \mathbf{x}_1 \qquad \small{(\text{Leibniz rule})} \\
  &= \int_{\mathcal{X}_1} -\nabla_\mathbf{x} \cdot \left(\mathbf{v}(\mathbf{x}, t \mid \mathbf{x}_1) p(\mathbf{x}, t \mid \mathbf{x}_1)\right) q_1(\mathbf{x}_1) \mathrm{d} \mathbf{x}_1 \qquad \small{(\text{Conditional continuity})} \\
  &= -\nabla_\mathbf{x} \cdot \left(\int_{\mathcal{X}_1} \mathbf{v}(\mathbf{x}, t \mid \mathbf{x}_1) p(\mathbf{x}, t \mid \mathbf{x}_1) q_1(\mathbf{x}_1) \mathrm{d} \mathbf{x}_1\right) \qquad \small{(\text{Leibniz rule})} \\
  &= -\nabla_\mathbf{x} \cdot \left(\mathbf{v}(\mathbf{x}, t) p(\mathbf{x}, t)\right) \qquad \small{(\text{Marginal VF})}\\
  \\
  \implies \mathbf{v}(\mathbf{x}, t) &= \frac{1}{p(\mathbf{x}, t)} \int_{\mathcal{X}_1} \mathbf{v}(\mathbf{x}, t \mid \mathbf{x}_1) p(\mathbf{x}, t \mid \mathbf{x}_1) q_1(\mathbf{x}_1) \mathrm{d} \mathbf{x}_1. \qquad \text{(Marginal VF)}
\end{aligned}
$$

Based on marginal VF, the FM authors showed that we can actually train to match the conditional VFs:

$$
  \begin{aligned}
  L(\theta) &= \underset{\substack{t \sim \mathcal{U}[0, T] \\ \mathbf{x} \sim p(\mathbf{x}, t)}}{\mathbb{E}} \lVert \mathbf{v}_\theta(\mathbf{x}, t) - \mathbf{v}(\mathbf{x}, t) \rVert^2 \qquad \text{(FM)}\\
  &= \underset{\substack{t \sim \mathcal{U}[0, T] \\ \mathbf{x}_1 \sim q_1(\mathbf{x}_1)\\ \mathbf{x} \sim p(\mathbf{x}, t \mid \mathbf{x}_1)}}{\mathbb{E}} \lVert \mathbf{v}_\theta(\mathbf{x}, t) - \mathbf{v}(\mathbf{x}, t \mid \mathbf{x}_1) \rVert^2 + \text{const.} \qquad \text{(CFM)}
  \end{aligned}
$$

where the FM loss matches the NN VF to the unknown desired VF $$\mathbf{v}(\mathbf{x}, t)$$ (what we originally wanted to do) and the CFM loss matches the NN VF to the conditional VFs $$\mathbf{v}(\mathbf{x}, t \mid \mathbf{x}_1)$$.
Since these losses are equal up to a constant, minimising them to convergence should, in theory, result in the same NN VF $$\mathbf{v}_\theta(\mathbf{x}, t)$$.
Check Theorem 1 and Theorem 2 of the [FM paper](https://arxiv.org/abs/2210.02747) to see the original proof of the marginal VF and CFM loss equivalence.

The interesting thing about the marginal VF $$\mathbf{v}(\mathbf{x}, t)$$ is that, similar to the marginal PDF $$p(\mathbf{x}, t \mid \mathbf{x}_1)$$, it is a mixture of conditinonal VFs $$\mathbf{v}(\mathbf{x}, t \mid \mathbf{x}_1)$$.
What does this "mixture" actually mean?
In the case of the marginal PDF, this mixture is just the marginalisation of the conditional PDFs over all samples of the complex distribution $$\mathbf{x}_1 \sim q_1$$.
But, for the marginal VF, it is a bit less clear, but qualitateively it is some weighted combination of the conditinonal VFs. Let's take a look at the terms in the expression for the marginal VF other than the conditinonal VF:

$$
  \frac{1}{p(\mathbf{x}, t)} \int_{\mathcal{X}_1} p(\mathbf{x}, t \mid \mathbf{x}_1) q(\mathbf{x}_1) \mathrm{d}\mathbf{x}_1 = \frac{p(\mathbf{x}, t)}{p(\mathbf{x}, t)} = 1.
$$

This means that, in fact, the marginal VF is a [convex combination](https://en.wikipedia.org/wiki/Convex_combination) of the conditional VF, where the weights are all positive (PDF property) and the weights sum to $$1$$.
Since the marginal VF admits a flow map, we then have a [differential inclusion](https://en.wikipedia.org/wiki/Differential_inclusion):

$$
  \frac{\mathrm{d} \mathbf{\phi}(\mathbf{x}, t)}{\mathrm{d} t} \in \mathrm{co} \left\{\mathbf{v}(\mathbf{\phi}(\mathbf{x}, t), t \mid \mathbf{x}_1) \mid \mathbf{x}_1 \sim q_1 \right\}, \qquad \text{(Differential Inclusion)}
$$

where $$\mathrm{co}$$ is the [convex hull operator](https://en.wikipedia.org/wiki/Convex_hull), which gives the set of all possible convex combinations.
Take a look at the drawing below.

<figure style="text-align: center;">
  <img src="../../../assets/img/marginal_vf.jpg" width="90%">
  <figcaption><strong>Figure</strong>: The marginal VF lies in some convex combination of the conditional VFs (red).</figcaption>
</figure>

Differential inclusions were introduced in the 1960s by [Aleksei Filippov](https://en.wikipedia.org/wiki/Aleksei_Filippov_(mathematician)) as a way to characterise solution to ODEs with discontinuous vector fields (see Filippov's [book](https://link.springer.com/book/10.1007/978-94-015-7793-9)).
They are an integral part of discontinuous dynamical systems (DDSs), where several VFs interface on a partitioned domain (see my drawing below).
I recommend Jorge Cortes' [article](https://ieeexplore.ieee.org/abstract/document/4518905) for a complete description.

<figure style="text-align: center;">
  <img src="../../../assets/img/dds.jpg" width="90%">
  <figcaption><strong>Figure</strong>: A DDS with \(\mathbf{v}(\mathbf{x}, t \mid \mathbf{x}_i)\) over \(\Omega_i \subset \mathbb{R}^n\) and \(\mathbf{v}(\mathbf{x}, t \mid \mathbf{x}_j)\) over \(\Omega_j \subset \mathbb{R}^n\), such that \(\Omega_i \cap \Omega_j = \emptyset\) and \(\Omega_i \cup \Omega_j = \mathbb{R}^n\). A Filippov solution will involve a convex combination of the VFs on the switching boundary \(\partial \Omega_i \cup \partial \Omega_j\).</figcaption>
</figure>

DDSs with differential inclusions are commonplace in [hybrid dynamical systems (HDSs)](https://en.wikipedia.org/wiki/Hybrid_system), such as switched systems or [behavior trees (BTs)](https://arxiv.org/abs/2109.01575) (shamless plug to my PhD research).
I recommend Daniel Liberzon's [lecture](http://liberzon.csl.illinois.edu/teaching/Liberzon-LectureNotes.pdf) on switched systems.

Switched systems are particularly relevant to the marginal VF $$\mathbf{v}(\mathbf{x}, t)$$ discussed above.
In switched systems, there is a "switching signal" the indicates which VF to use (the conditional VFs $$\mathbf{v}(\mathbf{x}, t \mid \mathbf{x}_1)$$ in our case).
This signal may be state-dependent (i.e. dependent on $$\mathbb{R}^n$$) or time-dependent.
If it is only state-dependent, then we end up with a system that looks like the picture above, where the conditional VFs $$\mathbf{v}(\mathbf{x}, t \mid \mathbf{x}_1)$$ are assigned to each partition of the domain.
In the time-dependent case, the switching signal defines a schedule of switching times, i.e. which intervals of time to use a particular conditional VF.
A crucial problem is determining when the switched system will be stable (i.e. all flow maps converge to some desired state).
E.g. we would want all flow maps $$\mathbf{\phi}(\mathbf{x}, t)$$ of the marginal VF $$\mathbf{v}(\mathbf{x}, t)$$ to converge to the support $$\mathrm{supp}(q_1)$$ of the complex distribution $$q_1$$.

Let's say we are considering the time-dependent case, but we do not actually know the switching times.
Then it suffices to consider showing "stability under arbitrary switching", which is essentially showing stability of the differential inclusion.
This is exactly the case in FM!

<!-- Several interesting works from the control theory literature are relevant here:
- Considering that the conditional VF $$\mathbf{v}(\mathbf{x}, t \mid \mathbf{x}_1)$$ used in FM are linear, and each have their own equilibria, the [paper](https://ieeexplore.ieee.org/abstract/document/4434822) of Mastellone et al is relevant. There they characterize stability of a linear VF $$\mathbf{v}() -->



<!-- 

# Old

Well, the authors of [flow matching](https://arxiv.org/abs/2210.02747) (FM) proposed something called *conditional* flow matching (CFM) loss:

$$
  \begin{aligned}
  L(\theta) &= \underset{\substack{t \sim \mathcal{U}[0, T] \\ \mathbf{x} \sim p(\mathbf{x}, t)}}{\mathbb{E}} \lVert \mathbf{v}_\theta(\mathbf{x}, t) - \mathbf{v}(\mathbf{x}, t) \rVert^2 \qquad \text{(FM)}\\
  &= \underset{\substack{t \sim \mathcal{U}[0, T] \\ \mathbf{x}_1 \sim q_1(\mathbf{x}_1)\\ \mathbf{x} \sim p(\mathbf{x}, t \mid \mathbf{x}_1)}}{\mathbb{E}} \lVert \mathbf{v}_\theta(\mathbf{x}, t) - \mathbf{v}(\mathbf{x}, t \mid \mathbf{x}_1) \rVert^2 + \text{const.} \qquad \text{(CFM)}
  \end{aligned}
$$

where we assume that such an underlying vector field $$\mathbf{v}(\mathbf{x}, t)$$ does exist, we construct a conditional vector field $$\mathbf{v}(\mathbf{x}, t \mid \mathbf{x}_1)$$ for each data sample $$\mathbf{x}_1 \sim q_1$$ to locally emulate the underlying vector field , and we then train a neural-network (NN) vector field $$\mathbf{v}_\theta(\mathbf{x}, t)$$ to match it.
The FM authors suggested that we should want $$\mathbf{v}(\mathbf{x}, t \mid \mathbf{x}_1)$$ to generate a conditional PDF $$p(\mathbf{x}, t \mid \mathbf{x}_1)$$ such that it converges to a concentrated distribution around the data sample (see the drawing below), i.e.





The key thing here is that, since the loss functions are equal up to a constant, their gradients with respect to the NN parameters $$\theta$$ will be the same! So, training with either to convergence should (in theory) lead to the same result.
Now, how do we know this is actually true?

Well, since the conditional vector field $$\mathbf{v}(\mathbf{x}, t \mid \mathbf{x}_1)$$ generates a conditional PDF $$p(\mathbf{x}, t \mid \mathbf{x}_1)$$, they must satisfy their own conintuity equation:

$$
  \frac{\partial p(\mathbf{x}, t \mid \mathbf{x}_1)}{\partial t} = -\nabla_\mathbf{x} \cdot \left(\mathbf{v}(\mathbf{x}, t \mid \mathbf{x}_1) p(\mathbf{x}, t \mid \mathbf{x}_1)\right). \qquad \text{(Continuity equation)}
$$


# Okld

Since both losses are equal up to a constant, their gradients with respect to the NN parameters will be the same! So, training with both loss function to convergence will give us approximately the same result.
Now, how do we really know this is true?



Now, how do we locally emulate $$\mathbf{v}(\mathbf{x}, t)$$ with $$\mathbf{v}(\mathbf{x}, t \mid \mathbf{x}_1)$$ for each $$\mathbf{x}_1 \sim q_1$$?




Well, Lipman et al. proposed something called conditional flow matching, where we construct a conditional CNF to emulate the kind flow that we would desire for each data sample $$\mathbf{x} \sim \mathcal{D}$$.


which also means that we can just sample from the simple distribution $$\mathbf{x}_0 \sim q_1$$ and then integrate the vector field 

Note that the vector field $$\mathbf{v}$$ and $


So then, assuming we have the vector field $$\mathbf{v}$$ we can just sample from the simple distribution $$\mathbf{x}_0 \sim q_0$$ and then numerically integrate $$\mathbf{v}$$ from $$\mathbf{x}_0$$


 and then compute $$x_1 \sim q_1$$ by numerically integrating $$\mathbf{v}$$ from $$\mathbf{x}_0$$


Unfortunately, we do not have $$\mathbf{v}$$; if we did then the job would be done!
So, Lipman et al. proposed the conditional flow matching (CFM) objective function

$$
  \begin{aligned}
  L(\theta) &= \underset{\substack{t \sim \mathcal{U}[0, T] \\ \mathbf{x} \sim p(\mathbf{x}, t)}}{\mathbb{E}} \lVert \mathbf{v}_\theta(\mathbf{x}, t) - \mathbf{v}(\mathbf{x}, t) \rVert^2 \qquad \text{(FM)}\\
  &= \underset{\substack{t \sim \mathcal{U}[0, T] \\ \mathbf{x}_1 \sim q_1(\mathbf{x}_1)\\ \mathbf{x} \sim p(\mathbf{x}, t \mid \mathbf{x}_1)}}{\mathbb{E}} \lVert \mathbf{v}_\theta(\mathbf{x}, t) - \mathbf{v}(\mathbf{x}, t \mid \mathbf{x}_1) \rVert^2 + \text{const.} \qquad \text{(CFM)}
  \end{aligned}
$$

where $$\mathbf{x}_1 \in \mathcal{D}$$ is a sample from the dataset, $$\mathbf{v}(\cdot \mid \mathbf{x}_1)$$ is a conditional vector field, and $$p(\cdot \mid \mathbf{x}_1)$$ is a conditional PDF. Together, they satisfy their own continuity equation:

$$
\begin{aligned}
  \frac{\partial p(\mathbf{x}, t \mid \mathbf{x}_1)}{\partial t} &= -\nabla_\mathbf{x} \cdot \left(\mathbf{v}(\mathbf{x}, t \mid \mathbf{x}_1) p(\mathbf{x}, t \mid \mathbf{x}_1)\right). \qquad \text{(Continuity equation)} \\
  \exists T \in \mathbb{R}_{\geq 0} \quad &\text{s.t.} \quad \lim_{t \to T} p(\mathbf{x}, t \mid \mathbf{x}_1) \approx \delta(\mathbf{x} - \mathbf{x}_1) \quad \text{(Data hitting)}
\end{aligned}
$$.



A key insight necessary to prove the CFM objective function is the concept of the **marginal vector field** $$\mathbf{v}$$, which is the vector field that generates the marginal PDF $$p(\mathbf{x}, t)$$ from the conditional PDF $$p(\mathbf{x}, t \mid \mathbf{x}_1)$$.
The first assumption is that the marginal PDF is a mixture of the conditional PDFs:

$$
  p(\mathbf{x}, t) \approx \int_{\mathcal{X}_1} p(\mathbf{x} \mid \mathbf{x}_1) q_1(\mathbf{x}_1) \mathrm{d} \mathbf{x}_1. \qquad \text{(Marginal PDF)}
$$

In the flow matching paper they use this assumption to find the conditional vector field $$\mathbf{v}(\mathbf{x}, t \mid \mathbf{x}_1)$$:

$$
\begin{aligned}
  \frac{\partial p(\mathbf{x}, t)}{\partial t} &= \frac{\partial}{\partial t} \int_{\mathcal{X}_1} p(\mathbf{x} \mid \mathbf{x}_1) q_1(\mathbf{x}_1) \mathrm{d} \mathbf{x}_1 \\
  &= \int_{\mathcal{X}_1} \frac{\partial p(\mathbf{x} \mid \mathbf{x}_1)}{\partial t} q_1(\mathbf{x}_1) \mathrm{d} \mathbf{x}_1 \\
  &= \int_{\mathcal{X}_1} -\nabla_\mathbf{x} \cdot \left(\mathbf{v}(\mathbf{x}, t \mid \mathbf{x}_1) p(\mathbf{x}, t \mid \mathbf{x}_1)\right) q_1(\mathbf{x}_1) \mathrm{d} \mathbf{x}_1 \\
  &= -\nabla_\mathbf{x} \cdot \left(\int_{\mathcal{X}_1} \mathbf{v}(\mathbf{x}, t \mid \mathbf{x}_1) p(\mathbf{x}, t \mid \mathbf{x}_1) q_1(\mathbf{x}_1) \mathrm{d} \mathbf{x}_1\right) \\
  &= -\nabla_\mathbf{x} \cdot \left(\mathbf{v}(\mathbf{x}, t) p(\mathbf{x}, t)\right) \\
  \\
  \implies \mathbf{v}(\mathbf{x}, t) &\approx \frac{1}{p(\mathbf{x}, t)} \int_{\mathcal{X}_1} \mathbf{v}(\mathbf{x}, t \mid \mathbf{x}_1) p(\mathbf{x}, t \mid \mathbf{x}_1) q_1(\mathbf{x}_1) \mathrm{d} \mathbf{x}_1. \qquad \text{(Marginal VF)}
\end{aligned}
$$

Notice that

$$
\frac{1}{p(\mathbf{x}, t)} \int_{\mathcal{X}_1} p(\mathbf{x}, t \mid \mathbf{x}_1) q_1(\mathbf{x}_1) \mathrm{d} \mathbf{x}_1 = \frac{p(\mathbf{x}, t)}{p(\mathbf{x}, t)} = 1,
$$

thus, we have that

$$
\frac{\mathrm{d} \mathbf{\psi}(\mathbf{x}, t)}{\mathrm{d} t} = \mathbf{v}(\mathbf{\psi}(\mathbf{x}, t), t) \in \mathrm{co}\left\{\mathbf{v}(\mathbf{x}, t \mid \mathbf{x}_1) \mid \mathbf{x}_1 \in \mathcal{X}_1\right\},
$$





# Old


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
$$ -->
