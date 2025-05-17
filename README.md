# CAUSALGEN

CAUSALGEN is a Bayesian framework for estimating causal effects in time series data with heterogeneous response dynamics across structured groups. It was developed to address limitations in conventional tools like CausalImpact, particularly in settings where treatment effects vary by segment and unfold over time in complex, non-additive ways.

At its core, CAUSALGEN combines a shared latent state process, segment-specific trend deviations, and multiplicative event effects into a coherent generative model. The structure is modular and extensible, making it suitable for scenarios involving geographic regions, user segments (e.g., casual vs. hardcore players), or any other partition where partial pooling is desirable. All inference is conducted in STAN, with the model designed to be flexible rather than plug-and-play — intended for researchers and applied modelers who need fine-grained control over assumptions.

The model uses a multiplicative structural time series formulation. This allows components to interact nonlinearly, capturing state-dependent variance and avoiding some of the brittleness that comes with additive decompositions. Segment-specific local trends are learned via hierarchical priors, enabling group-level flexibility while preserving statistical efficiency. A shared AR(3) latent state governs underlying behavioral dynamics across all groups, and known interventions (e.g., product launches, holidays) can be incorporated through a set of multiplicative event terms. All components are forecasted forward after the intervention point; post-treatment observations are excluded from the fitting process, ensuring causal validity through strict separation of observed and counterfactual data.

This approach is well suited for domains like gaming, media, and digital platforms, where interventions affect structured populations differently, and where assumptions like parallel trends or constant treatment effects often fail. For example, you might use CAUSALGEN to evaluate a pricing experiment across regions with different usage baselines, or to estimate the impact of a live event on distinct engagement segments.

The current implementation includes:

 - A hierarchical local linear trend model with segment-specific initial levels and slopes

 - A shared AR(3) latent state driving multiplicative dynamics across all segments

 - Optional inclusion of covarites that scale the outcome over time

 - Forward simulation of post-intervention periods without introducing new shocks

 - Segment-specific observation noise models


You’ll find the main STAN model in /stan/causalgen.stan. We plan to release a set of R and Python utilities for fitting and diagnostics in upcoming updates.

This project is still evolving. Feedback, extensions, and adaptations are welcome. CAUSALGEN is meant to serve as a robust and customizable starting point for practitioners working at the intersection of forecasting and causal inference — especially when one-size-fits-all tools fall short.


# RESULTS

When testing on real data in a predictive capacity, CausalGen performed between 7-12% better than Google's causal_impact in terms of the midpoint forecast. Importantly, the credible intervals for CausalGen were 25-35% narrower, which aligned with observed data at 95% coverage. 
