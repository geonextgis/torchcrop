# Usage

To use torchcrop in a project:

```python
import torchcrop
```

## Hybrid modeling: neural residual corrections

`torchcrop` lets you attach **optional** neural residual corrections to
individual mechanistic quantities through a constraint-aware framework
(`torchcrop.nn.hybrid`). Each correction target is described by a
`ResidualSpec` whose `constraint` selects a projection that keeps the learned
term inside the quantity's natural geometry, so corrections preserve sign,
the $[0, 1]$ range of stress factors, and the partition-of-unity of the
biomass split:

| Constraint | Target kind | Projection | Guarantee |
| --- | --- | --- | --- |
| `rate_factor` | non-negative scalar / flux / rate constant | $base \cdot e^{\delta}$ | stays $\ge 0$ |
| `transfer` | flux between two conserved pools | $base \cdot e^{\delta}$ (one number, both legs) | mass conserved |
| `unit_interval` | factor in $(0, 1)$ | $\sigma(\operatorname{logit}(base) + \delta)$ | stays in $(0, 1)$ |
| `simplex` | fractions summing to $1$ | $\operatorname{softmax}(\log base + \delta)$ | $\sum = 1$ exactly |

The raw correction is the bounded $\delta = \text{scale} \cdot \tanh(\text{MLP}(\text{context}))$,
and each head's final layer is **zero-initialised**, so a model built with
residual specs but untrained reproduces the pure mechanistic trajectory.

### Enabling corrections

```python
import torch.nn.functional as F
from torchcrop import Lintul5Model
from torchcrop.nn import ResidualSpec, default_slots

# Option A: hand-pick the slots tied to your observables (recommended).
model = Lintul5Model(residual_specs=[
    ResidualSpec(
        "photosynthesis.gtotal", "rate_factor",
        context=("lai", "dvs", "davtmp", "tranrf", "nstress"), scale=0.15,
    ),
])

# Option B: the full recommended catalogue (only if every pathway is observable).
model = Lintul5Model(residual_specs=default_slots())
```

All residual parameters live in `model.parameters()`, so training is standard
PyTorch. Add the residual-magnitude penalty to the loss to anchor corrections
toward zero:

```python
out = model(weather)
loss = F.mse_loss(out.yield_, observed_yield) + 1e-3 * model.hybrid.penalty()
loss.backward()
```

Supported slot names: `"photosynthesis.gtotal"`, `"water.tranrf"`,
`"partitioning.aboveground"`, `"partitioning.fr"`, `"leaf.rdr"`,
`"phenology.dvs_rate"`.

!!! warning "Choosing which slots to enable matters more than the list itself"
    Enable **only** the slots whose pathway is constrained by an observable in
    your calibration data — for example, `"photosynthesis.gtotal"` if you
    observe biomass or yield, `"water.tranrf"` if you observe soil moisture or
    transpiration. Turning on **all** slots at once invites *identifiability*
    and *compensation* problems: several residuals — and the mechanistic
    parameters they shadow, such as `gtotal` versus a learnable RUE — become
    degenerate, so the optimiser can fit the data while learning physically
    meaningless corrections.

    A robust workflow:

    1. Enable a **minimal** set of slots, each tied to an observable.
    2. Calibrate the **mechanistic** parameters first, then add residuals.
    3. Regularise with `model.hybrid.penalty()` to keep corrections small.
