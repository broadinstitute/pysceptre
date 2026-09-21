# pysceptre

Standalone Python port of the statistical engine behind
[`sceptre`](https://github.com/Katsevich-Lab/sceptre)'s discovery analysis for
single-cell CRISPR screens.

```python
from pysceptre import run_discovery_analysis
```

Three validated analysis paths — **discovery analysis**, the **calibration
check** and the **power check** — on the complement control group with CRT
resampling, for high-MOI screens. Every result is validated against the R
package rather than against itself. What is deliberately *not* implemented is
listed in [Scope and limitations](scope.md); read it before assuming a path
works.

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
