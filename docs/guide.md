# User guide

--8<-- "README.md:quickstart"

## The other two analyses

The calibration check and the power check take the same inputs and return the
same shape of frame. They differ in where their pairs come from, and that
difference shows up in the result:

| | pairs | QC failures |
|---|---|---|
| `run_discovery_analysis` | you supply them | reported in-band: `pass_qc = False`, `p_value = NaN` |
| `run_calibration_check` | constructed from non-targeting gRNAs | cannot occur — pairs are chosen to pass, so there is no `pass_qc` column |
| `run_power_check` | you supply positive controls | reported in-band, like discovery |

The asymmetry is deliberate. A positive control is a claim about one specific
pair, so silently dropping the ones with too few cells would overstate power;
a calibration pair is interchangeable, so one that would fail is simply never
drawn. See [Design decisions](design.md#pairwise-qc-appears-in-two-different-shapes).

Signatures and parameters for all three are in the
[API reference](api.md).

--8<-- "README.md:reproducibility"

--8<-- "README.md:performance"
