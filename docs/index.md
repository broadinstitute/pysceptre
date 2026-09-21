# pysceptre

Standalone Python port of the statistical engine behind
[`sceptre`](https://github.com/Katsevich-Lab/sceptre)'s discovery analysis for
single-cell CRISPR screens.

```python
from pysceptre import run_discovery_analysis
```

Three validated sceptre paths -- **discovery analysis**, the **calibration
check** and the **power check** -- on the complement control group with CRT
resampling, for high-MOI screens, each validated against the R package rather
than against itself.

Alongside them, **`compute_power`** estimates in closed form what a
screen *could* have detected, per pair and without simulation. It answers a
different question, it is a port of PerturbPlan rather than of sceptre, and it
is validated against that package instead.

What is deliberately *not* implemented, and which paths are exercised rather
than validated, is in [Scope and limitations](scope.md); read it before
assuming a path works.

--8<-- "README.md:why"

## Where to go next

| | |
|---|---|
| [Installation](installation.md) | `uv` and `pip`, the `fast` and `io` extras, tested Python versions |
| [User guide](guide.md) | a run end to end, and how to read the result frame |
| [Tutorials](tutorials.md) | a worked example from synthetic inputs |
| [API reference](api.md) | generated from the source |
| [Scope and limitations](scope.md) | the paths that are not implemented |
| [Design decisions](design.md) | why the port differs from R where it does |
