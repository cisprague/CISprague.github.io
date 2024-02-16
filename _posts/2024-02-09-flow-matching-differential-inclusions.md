---
layout: post
title:  Flow matching and differential inclusions
date:   2024-02-09 10:00
description: An interesting connection between flow matching, differential inclusions, hybrid dynamical systems, and discontinuous dynamical systems..
tags: flow-matching, generative-models, control-theory
---


[Generative modeling](https://en.wikipedia.org/wiki/Generative_model) is a fundamental concept in machine learning, where you typically want to create a model that can generate samples from some complex distribution.
Recently, a new type of generative modeling framework, called [flow matching](https://arxiv.org/abs/2210.02747), has been proposed as an alternative to [diffusion models](https://arxiv.org/abs/2011.13456).
Flow matching relies on the framework of [continuous normalizing flows (CNFs)](https://arxiv.org/abs/1806.07366), where you learn a model of a time-dependent [vector field (VF)](https://en.wikipedia.org/wiki/Vector_field) $$\mathbf{v}: \mathbb{R}^n \times \mathbb{R}_{\geq 0} \to \mathbb{R}^n$$ and then use it to transport samples from a simple distribution $$q_0$$ (e.g. a [normal distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution)) to samples of a more complex distribution $$q_1$$ (e.g. golden retriever images), like I have shown below with my fantastic artistic skills...

<figure style="text-align: center;">
  <img src="../../../assets/img/transport.jpg" alt="Probability distributions" title="Probability distributions" width="90%">
  <figcaption><strong>Figure</strong>: The transportation of a simple distribution \(q_0\) to a more complex distribution \(q_1\).</figcaption>
</figure>

Theoretically, this transportation of samples obeys the well-known [continuity equation](https://en.wikipedia.org/wiki/Continuity_equation) from physics:

<div class="math-container">
$$
  \frac{\partial p(\mathbf{x}, t)}{\partial t} = -\nabla_\mathbf{x} \cdot \left(\mathbf{v}(\mathbf{x}, t) p(\mathbf{x}, t)\right)
$$
</div>

where $$p : \mathbb{R}^n \times \mathbb{R}_{\geq0} \to \mathbb{R}_{\geq 0}$$ is a time-dependent "density" and $$\nabla_\mathbf{x} \cdot$$ is the [divergence operator](https://en.wikipedia.org/wiki/Divergence). This equation essentially says that the total mass of the system does not change over time.
In our case, this "mass" is just the total probability of the space, which is $$\int_{\mathcal{X}} p(\mathbf{x}, t) \mathrm{d}\mathbf{x} = 1$$ for [probability density functions (PDFs)](https://en.wikipedia.org/wiki/Probability_density_function). See another fantastic art piece below for an illustration.

<figure style="text-align: center;">
  <img src="../../../assets/img/continuity.jpg" width="90%">
  <figcaption><strong>Figure</strong>: The continuity equation preserves the total mass of the PDF!</figcaption>
</figure>

So, if the VF and PDF $$(\mathbf{v}, p)$$ satisfy the continuity equation, then we can say that $$\mathbf{v}$$ generates $$p$$.
This also means that the well-known change-of-variables equation is satisfied (see Chapter 1 of Villani's [book on optimal transport](https://link.springer.com/book/10.1007/978-3-540-71050-9) for details):

<div class="math-container">
$$
  p(\mathbf{x}, t) = p(\mathbf{\phi}^{-1}(\mathbf{x}, t), 0) \det \left(\nabla_\mathbf{x} \mathbf{\phi}^{-1}(\mathbf{x}, t) \right)
$$
</div>

where $$\mathbf{\phi}: \mathbb{R}^n \times \mathbb{R}_{\geq 0} \to \mathbb{R}^n$$ is a [flow map](https://en.wikipedia.org/wiki/Flow_(mathematics)) (or [integral curve](https://en.wikipedia.org/wiki/Integral_curve)) of $$\mathbf{v}$$ starting from $$\mathbf{x}$$, defining an [ordinary differential equation (ODE)](https://en.wikipedia.org/wiki/Ordinary_differential_equation)

<div class="math-container">
$$
  \frac{\mathrm{d} \mathbf{\phi}(\mathbf{x}, t)}{\mathrm{d} t} = \mathbf{v}(\mathbf{\phi}(\mathbf{x}, t), t);
$$
</div>

$$\mathbf{\phi}^{-1}(\mathbf{x}, t)$$ is its inverse with respect to $$\mathbf{x}$$; and $$\nabla_\mathbf{x} \mathbf{\phi}^{-1}(\mathbf{x}, t)$$ is the Jacobian matrix  with respect to $$\mathbf{x}$$ of its inverse.
Essentially, this is just a theoretical justification saying that we can sample $$\mathbf{x}_0 \sim q_1$$ and then compute $$\mathbf{x}_1 = \mathbf{\phi}(\mathbf{x}_0, T)$$ through numerical integration of $$\mathbf{v}$$ starting from $$\mathbf{x}_0$$ to get a sample from the complex distribution $$\mathbf{x}_1 \sim q_1$$.
See another drawing below!

<figure style="text-align: center;">
  <img src="../../../assets/img/flow_map.jpg" width="90%">
  <figcaption><strong>Figure</strong>: A flow map \(\mathbf{\phi}(\mathbf{x}, t)\) from \(t = 0\) to \(t = T\).</figcaption>
</figure>

Okay, but there is one big problem here: how do we actually learn a model of such a vector field  $$\mathbf{v}$$ if we only have samples from the simple and complex PDFs, $$q_0$$ and $$q_1$$?
Well, the [flow matching (FM)](https://arxiv.org/abs/2210.02747) authors proposed learning from intermediate samples of a conditional PDF $$p(\mathbf{x}, t \mid \mathbf{x}_1)$$ that converges to a concentrated PDF (i.e. a [Dirac delta distribution](https://en.wikipedia.org/wiki/Dirac_delta_function) $$\delta$$) around each data sample $$\mathbf{x}_1 \sim q_1$$ such that it locally emulates the desired PDF (see drawing below). I.e. we design $$p(\mathbf{x}, t \mid \mathbf{x}_1)$$ such that, for some time $$T \in \mathbb{R}_{\geq 0}$$, we have:

<div class="math-container">
$$
\lim_{t \to T} p(\mathbf{x}, t \mid \mathbf{x}_1) \approx \delta(\mathbf{x} - \mathbf{x}_1).
$$
</div>

<figure style="text-align: center;">
  <img src="../../../assets/img/conditional_pdf.jpg" width="90%">
  <figcaption><strong>Figure</strong>: \(q_0\) being transported to \(\delta(\mathbf{x} - \mathbf{x}_1)\) for each \(\mathbf{x}_1 \sim q_1\).</figcaption>
</figure>

Just like before, the conditional PDF $$p(\mathbf{x}, t \mid \mathbf{x}_1)$$ also has a vector field $$\mathbf{v}(\mathbf{x}, t \mid \mathbf{x}_1)$$ that generates it, so it also has a continuity equation:

<div class="math-container">
$$
  \frac{\partial p(\mathbf{x}, t \mid \mathbf{x}_1)}{\partial t} = -\nabla_\mathbf{x} \cdot \left(\mathbf{v}(\mathbf{x}, t \mid \mathbf{x}_1) p(\mathbf{x}, t \mid \mathbf{x}_1)\right).
$$
</div>


The FM authors then make the assumption that the desired PDF can be constructed by a "[mixture](https://en.wikipedia.org/wiki/Mixture_distribution)" of the conditional PDFs:

<div class="math-container">
$$
  p(\mathbf{x}, t) = \int_{\mathcal{X}_1} p(\mathbf{x} \mid \mathbf{x}_1) q_1(\mathbf{x}_1) \mathrm{d} \mathbf{x}_1, 
$$
</div>

where the desired PDF $$p(\mathbf{x}, t)$$ can be interpreted as the "marginal PDF". 
With this assumption, they then identify a **marginal VF** by using both the marginal and conditional continuity equations:

<div class="math-container">
$$
\begin{aligned}
  \frac{\partial p(\mathbf{x}, t)}{\partial t} &= \frac{\partial}{\partial t} \int_{\mathcal{X}_1} p(\mathbf{x} \mid \mathbf{x}_1) q_1(\mathbf{x}_1) \mathrm{d} \mathbf{x}_1  \\
  &= \int_{\mathcal{X}_1} \frac{\partial p(\mathbf{x} \mid \mathbf{x}_1)}{\partial t} q_1(\mathbf{x}_1) \mathrm{d} \mathbf{x}_1  \\
  &= \int_{\mathcal{X}_1} -\nabla_\mathbf{x} \cdot \left(\mathbf{v}(\mathbf{x}, t \mid \mathbf{x}_1) p(\mathbf{x}, t \mid \mathbf{x}_1)\right) q_1(\mathbf{x}_1) \mathrm{d} \mathbf{x}_1 \\
  &= -\nabla_\mathbf{x} \cdot \left(\int_{\mathcal{X}_1} \mathbf{v}(\mathbf{x}, t \mid \mathbf{x}_1) p(\mathbf{x}, t \mid \mathbf{x}_1) q_1(\mathbf{x}_1) \mathrm{d} \mathbf{x}_1\right)  \\
  &= -\nabla_\mathbf{x} \cdot \left(\mathbf{v}(\mathbf{x}, t) p(\mathbf{x}, t)\right) \\
  \\
  \implies \mathbf{v}(\mathbf{x}, t) &= \frac{1}{p(\mathbf{x}, t)} \int_{\mathcal{X}_1} \mathbf{v}(\mathbf{x}, t \mid \mathbf{x}_1) p(\mathbf{x}, t \mid \mathbf{x}_1) q_1(\mathbf{x}_1) \mathrm{d} \mathbf{x}_1. 
\end{aligned}
$$
</div>


Based on the marginal VF $$\mathbf{v}(\mathbf{x}, t)$$, the FM authors showed that we can train an NN VF $$\mathbf{v}_\theta$$ to match the conditional VFs:

<div class="math-container">
$$
  \begin{aligned}
  L_\text{FM}(\theta) &= \underset{\substack{t \sim \mathcal{U}[0, T] \\ \mathbf{x} \sim p(\mathbf{x}, t)}}{\mathbb{E}} \lVert \mathbf{v}_\theta(\mathbf{x}, t) - \mathbf{v}(\mathbf{x}, t) \rVert^2\\
  L_\text{CFM}(\theta) &= \underset{\substack{t \sim \mathcal{U}[0, T] \\ \mathbf{x}_1 \sim q_1(\mathbf{x}_1)\\ \mathbf{x} \sim p(\mathbf{x}, t \mid \mathbf{x}_1)}}{\mathbb{E}} \lVert \mathbf{v}_\theta(\mathbf{x}, t) - \mathbf{v}(\mathbf{x}, t \mid \mathbf{x}_1) \rVert^2 \\
  \nabla_\theta L_\text{FM}(\theta) &= \nabla_\theta L_\text{CFM}(\theta)
  \end{aligned}
$$
</div>

where $$L_\text{FM}$$ matches the NN VF to the unknown desired VF $$\mathbf{v}(\mathbf{x}, t)$$ (what we originally wanted to do) and $$L_\text{CFM}$$ matches the NN VF to the conditional VFs $$\mathbf{v}(\mathbf{x}, t \mid \mathbf{x}_1)$$.
Since their gradients are equal, minimizing them should, in theory, result in the same NN VF $$\mathbf{v}_\theta(\mathbf{x}, t)$$.
Check Theorem 1 and Theorem 2 of the [FM paper](https://arxiv.org/abs/2210.02747) to see the original proof of the marginal VF and CFM loss equivalence.

The interesting thing about the marginal VF $$\mathbf{v}(\mathbf{x}, t)$$ is that, similar to the marginal PDF $$p(\mathbf{x}, t)$$, it is a mixture of conditinonal VFs $$\mathbf{v}(\mathbf{x}, t \mid \mathbf{x}_1)$$.
What does this "[mixture](https://en.wikipedia.org/wiki/Mixture_distribution)" actually mean?
In the case of the marginal PDF $$p(\mathbf{x}, t)$$, this mixture is just the marginalization of the conditional PDFs $$p(\mathbf{x}, t \mid \mathbf{x}_1)$$ over all samples of the complex distribution $$\mathbf{x}_1 \sim q_1$$.
But, for the marginal VF $$\mathbf{v}(\mathbf{x}, t)$$, it is a bit less clear, but qualitatively it must be some weighted combination of the conditional VFs $$\mathbf{v}(\mathbf{x}, t \mid \mathbf{x}_1)$$. Let's take a look at the terms in the marginal VF $$\mathbf{v}(\mathbf{x}, t)$$ expression other than the conditional VF:

<div class="math-container">
$$
  \frac{1}{p(\mathbf{x}, t)} \int_{\mathcal{X}_1} p(\mathbf{x}, t \mid \mathbf{x}_1) q(\mathbf{x}_1) \mathrm{d}\mathbf{x}_1 = \frac{p(\mathbf{x}, t)}{p(\mathbf{x}, t)} = 1.
$$
</div>

This means that, in fact, the marginal VF is a [convex combination](https://en.wikipedia.org/wiki/Convex_combination) of the conditional VFs, where the weights are all positive (PDFs are always positive) and sum to $$1$$.
Since the marginal VF $$\mathbf{v}(\mathbf{x}, t)$$ admits a flow map $$\mathbf{\phi}(\mathbf{x}, t)$$,  we then have a [differential inclusion](https://en.wikipedia.org/wiki/Differential_inclusion):

<div class="math-container">
$$
  \frac{\mathrm{d} \mathbf{\phi}(\mathbf{x}, t)}{\mathrm{d} t} \in \mathrm{co} \left\{\mathbf{v}(\mathbf{\phi}(\mathbf{x}, t), t \mid \mathbf{x}_1) \mid \mathbf{x}_1 \sim q_1 \right\},
$$
</div>

where $$\mathrm{co}$$ is the [convex hull operator](https://en.wikipedia.org/wiki/Convex_hull), which gives the set of all possible convex combinations.
Take a look at the red vectors in the drawing below; the set of all positively weighted averages (convex combination) of these vectors is the convex hull.

<figure style="text-align: center;">
  <img src="../../../assets/img/marginal_vf.jpg" width="90%">
  <figcaption><strong>Figure</strong>: The marginal VF lies in some convex combination of the conditional VFs (red).</figcaption>
</figure>

Differential inclusions were introduced in the 1960s by [Filippov (Филиппов)](https://en.wikipedia.org/wiki/Aleksei_Filippov_(mathematician)) as a way to characterize solutions to ODEs with discontinuous VFs (see Filippov's [book on differential inclusions](https://link.springer.com/book/10.1007/978-94-015-7793-9)).
They are an integral part of discontinuous dynamical systems (DDSs), where several VFs interface on a partitioned domain (see my drawing below).
I recommend Cortes' [article on DDSs](https://ieeexplore.ieee.org/abstract/document/4518905) for a complete description.

DDSs with differential inclusions are commonplace in [hybrid dynamical systems (HDSs)](https://en.wikipedia.org/wiki/Hybrid_system), such as switched systems or [behavior trees (BTs)](https://arxiv.org/abs/2109.01575) (shameless plug to my PhD research).
For a complete description, I recommend this [article on HDSs](https://ieeexplore.ieee.org/document/4806347) by Goebel, Sanfelice, and Teel; and this [article on switched systems](http://liberzon.csl.illinois.edu/teaching/Liberzon-LectureNotes.pdf) by Liberzon.

Switched systems are of particular relevance to the marginal VF $$\mathbf{v}(\mathbf{x}, t)$$ discussed above.
In switched systems, there is a "switching signal" $$\sigma$$ that indicates which VF to use (the conditional VFs $$\mathbf{v}(\mathbf{x}, t \mid \mathbf{x}_1)$$ in our case).
This signal may be state-dependent $$\sigma: \mathbb{R}^n \to \mathbb{N}$$ or time-dependent $$\sigma: \mathbb{R}_{\geq 0} \to \mathbb{N}$$, where $$\mathbb{N}$$ (natural numbers) contains the index set of the individual VFs (or "subsystems").
If we adapt this to work with the conditional VFs above, the switching signal would map like $$\sigma: \mathbb{R}^n \to \mathrm{supp}(q_1)$$ for the state-dependent case and $$\sigma: \mathbb{R}_{\geq 0} \to \mathrm{supp}(q_1)$$ for the time-dependent case, where $$\mathrm{supp}(q_1)$$ is the support of the complex PDF (i.e. where there is non-zero probability).

If the switching signal is only state-dependent, then we end up with a DDS that looks like the picture below, where the conditional VFs $$\mathbf{v}(\mathbf{x}, t \mid \mathbf{x}_1)$$ are assigned to each partition of the domain, i.e.:

<div class="math-container">
$$
 \frac{\mathrm{d} \mathbf{\phi}(\mathbf{x}, t)}{\mathrm{d} t} = \mathbf{v}(\mathbf{\phi}(\mathbf{x}, t), t \mid \sigma(\mathbf{\phi}(\mathbf{x}, t))).
$$
</div>

<figure style="text-align: center;">
  <img src="../../../assets/img/dds.jpg" width="90%">
  <figcaption><strong>Figure</strong>: A DDS with \(\mathbf{v}(\mathbf{x}, t \mid \mathbf{x}_i)\) over \(\Omega_i \subset \mathbb{R}^n\) and \(\mathbf{v}(\mathbf{x}, t \mid \mathbf{x}_j)\) over \(\Omega_j \subset \mathbb{R}^n\), such that \(\Omega_i \cap \Omega_j = \emptyset\) and \(\Omega_i \cup \Omega_j = \mathbb{R}^n\). A Filippov solution will involve a convex combination of the VFs on the switching boundary \(\partial \Omega_i \cup \partial \Omega_j\).</figcaption>
</figure>


In the time-dependent case, the switching signal defines a schedule of switching times, i.e. which intervals of time to use a particular conditional VF, i.e.

<div class="math-container">
$$
 \frac{\mathrm{d} \mathbf{\phi}(\mathbf{x}, t)}{\mathrm{d} t} = \mathbf{v}(\mathbf{\phi}(\mathbf{x}, t), t \mid \sigma(t)).
$$
</div>

Here the switching signal $$\sigma(t)$$ can be viewed as an open-loop control policy that we design or that we do not know (could come from external disturbances).
A crucial problem in switched systems is determining whether the system will be stable to some desired state (i.e. converges to the state and stays there).
In our case, we would want the flow map to be stable to samples of the complex distribution.

Now let's assume that we do not know the switching signal. In this case, it suffices to show "**stability under arbitrary switching**" (see chapter 4 of Liberzon's [article on switched systems](http://liberzon.csl.illinois.edu/teaching/Liberzon-LectureNotes.pdf)), which essentially shows the stability of the differential inclusion.
If we can prove that all convex combinations of the conditional VFs $$\mathbf{v}(\mathbf{x}, t \mid \mathbf{x}_1)$$ are stable, then we can prove that the marginal VF $$\mathbf{v}(\mathbf{x}, t)$$ is stable.
See the drawing below, where there is a convex combination of two 2D linear VFs. Every flow map $$\mathbf{\phi}(\mathbf{x}, t)$$ of every convex combination of these VFs will converge exponentially to the manifold in blue, which we can imagine as the support of the complex PDF $$q_1$$.

<figure style="text-align: center;">
  <img src="../../../assets/img/convex_vf.jpg" width="90%">
  <figcaption><strong>Figure</strong>: A convex combination of two 2D linear VFs is shown in red. The average combination is shown in red. All convex combinations will be stable to the manifold in blue. Imagine that this manifold is the support of the complex PDF. If \(\alpha = 0\) the VF will point to the left, if \(\alpha = 1\) the VF will point down.
  </figcaption>
</figure>

Now, why would we care about stability in generative models?
Well, of course we would want the flow maps $$\mathbf{\phi}(\mathbf{x}, t)$$ of the VF $$\mathbf{v}(\mathbf{x}, t)$$ to converge to samples of the complex distribution $$\mathbf{x}_1 \sim q_1$$.
But, in some applications, it may also be desirable to have it so that the flow maps $$\mathbf{\phi}(\mathbf{x}, t)$$ stay stable to the samples $$\mathbf{x}_1 \sim q_1$$.
For instance, in the context of structural biology, we may want to use a generative model the predict how a given [ligand](https://en.wikipedia.org/wiki/Ligand) (e.g. [serotonin](https://en.wikipedia.org/wiki/Serotonin)) binds to a given [receptor](https://en.wikipedia.org/wiki/Receptor_(biochemistry)) (e.g. the [serotonin receptor](https://en.wikipedia.org/wiki/5-HT_receptor)).
It is well-known in structural biology that molecular binding configurations represent minima of a "[free energy landscape](https://en.wikipedia.org/wiki/Folding_funnel)".
It is also well-known in control theory that energy can often be used as an effective [Lyapunov function](https://en.wikipedia.org/wiki/Lyapunov_function) $$V: \mathbb{R}^n \times \mathbb{R}_{\geq 0} \to \mathbb{R}_{\geq 0}$$, which is just a scalar function that can be used to certify that the VF $$\mathbf{v}(\mathbf{x}, t)$$ is stable within some region $$\mathcal{B}$$.
To be a Lyapunov function on a region $$\mathcal{B} \subset \mathbb{R}^n$$, we need to have the following for all $$(\mathbf{x}, t) \in \mathcal{B} \times \mathbb{R}_{\geq 0}$$:

<div class="math-container">
$$
\frac{\partial V(\mathbf{x}, t)}{\partial t} + \nabla_\mathbf{x}V(\mathbf{x}, t) \mathbf{v}(\mathbf{x}, t) \leq 0.
$$
</div>

Of course, if the marginal VF $$\mathbf{v}(\mathbf{x}, t)$$ just follows the negative gradient of this function, i.e. $$\mathbf{v}(\mathbf{x}, t) = -\nabla_\mathbf{x} V(\mathbf{x}, t)$$ (a gradient flow), then the second term will be negative.
The hard bit is ensuring that the first term with the time derivative is negative, which could normally be achieved by making the marginal VF $$\mathbf{v}(\mathbf{x}, t)$$ time-independent.
However, even if we make the conditional VFs time-independent, i.e. $$\mathbf{v}(\mathbf{x} \mid \mathbf{x}_1)$$, the marginal VF $$\mathbf{v}(\mathbf{x}, t)$$ will still be time-dependent due to the dependence on the conditional $$p(\mathbf{x}, t \mid \mathbf{x}_1)$$ and marginal PDF $$p(\mathbf{x}, t)$$:

<div class="math-container">
$$
\mathbf{v}(\mathbf{x}, t) = \frac{1}{p(\mathbf{x}, t)} \int_{\mathcal{X}_1} \mathbf{v}(\mathbf{x} \mid \mathbf{x}_1) p(\mathbf{x}, t \mid \mathbf{x}_1) q_1(\mathbf{x}_1) \mathrm{d} \mathbf{x}_1.
$$
</div>

Assume for the moment, though, that there does exist a free-energy function $$V(\mathbf{x}, t)$$ satisfying the Lyapunov condition.
Then, in the context of ligand-receptor binding, we could have something like in the drawing below, where the marginal VF $$\mathbf{v}(\mathbf{x}, t)$$ follows the negative gradient of a mixture of Lyapunov functions.


<figure style="text-align: center;">
  <img src="../../../assets/img/ligand.jpg" width="90%">
  <figcaption><strong>Figure</strong>: Energy descent in the context of ligand-receptor binding.
  </figcaption>
</figure>

This interpretation is useful, as it is often assumed in structural biology that data follows a [Boltzmann-like distribution](https://en.wikipedia.org/wiki/Boltzmann_distribution): 

<div class="math-container">
$$
  p(\mathbf{x}, t) = \frac{\exp(-V(\mathbf{x}, t))}{z(t)}.
$$
</div>

If this is true, and our flow maps $$\mathbf{\phi}(\mathbf{x}, t)$$ are following the negative gradient of the energy function $$V(\mathbf{x}, t)$$, then it is easy to see that they also follow the gradient of log probability $$\log(p(\mathbf{x}, t))$$ (shown in the drawing below):

<div class="math-container">
$$
\nabla_\mathbf{x} \log \left(p(\mathbf{x}, t) \right) = -\nabla_\mathbf{x} V(\mathbf{x}, t).
$$
</div>



<figure style="text-align: center;">
  <img src="../../../assets/img/gradient_flow.jpg" width="90%">
  <figcaption><strong>Figure</strong>: The correspondence between energy descent and log probability ascent.
  </figcaption>
</figure>


Now, why is this interpretation useful?
It is well-known that ligands can bind to receptors in different ways.
E.g., in the context of drugs, there are [orthosteric and allosteric sites](https://en.wikipedia.org/wiki/Allosteric_modulator) where drugs can bind.
Orthosteric sites are where endogenous drugs ([agonists](https://en.wikipedia.org/wiki/Agonist)) bind; e.g. serotonin is the endogenous agonist of the serotonin receptor.
Allosteric sites are sites other than the orthosteric site, and they are of increasing interest because they allow for specific [allosteric modulation](https://en.wikipedia.org/wiki/Allosteric_modulator).
The problem, however, is that allosteric binding data is not as common as orthosteric binding data, so NN models would most likely be biased toward orthosteric sites.
It is well-known in machine learning that [inductive biases](https://en.wikipedia.org/wiki/Inductive_bias) can help learning performance when there is a lack of data.
Energy could be a useful inductive bias to "bake" into the latent representation of models that are used to generate data corresponding to energy minima.

However, as mentioned before, the time-dependence of the marginal VF $$\mathbf{v}(\mathbf{x}, t)$$ makes it difficult to ensure that the flow maps $$\mathbf{\phi}(\mathbf{x}, t)$$ are stable to the complex distribution $$q_1$$.
In our new [preprint](https://arxiv.org/abs/2402.05774), we show how to make the marginal VF $$\mathbf{v}(\mathbf{x}, t)$$ time-independent to allow for the type of stability we just discussed! If you would like to cite this blog post, please use the following BibTeX entry.

```
@article{sprague2024stable,
  title={Stable Autonomous Flow Matching},
  author={Sprague, Christopher Iliffe and Elofsson, Arne and Azizpour, Hossein},
  journal={arXiv preprint arXiv:2402.05774},
  year={2024}
}
```

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