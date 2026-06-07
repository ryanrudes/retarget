# Result Schema

`RetargetingResult` is a strict, pickle-free checkpoint. The NPZ contains
`manifest_json` plus numeric arrays:

- `qpos`
- optional `cost`
- optional `human_joints`
- optional `robot_link_positions`
- optional `object_sample_points`

The manifest contains typed run, solver, mesh, playback, warning, and
provenance fields. Robot and motion vocabularies preserve their qualified enum
class identity. Solver reports preserve the qualified registry enum member for
both requested and resolved backends.

```python
from retarget.results import RetargetingResult

result.save_npz("result.npz")
restored = RetargetingResult.load_npz("result.npz")
```

`load_npz()` accepts only the current schema. Object arrays, compatibility keys,
and old constructors are intentionally unsupported.

Use `RetargetingResult.resampled(fps)` when comparing or exporting runs on a
common timeline. If `RetargetingProblem.output_fps` is set, the problem is
resampled before optimization and the structured run report records both input
and output rates.

`EvaluationReport` stores scalar metrics, units, structured details, warnings,
and the result/problem identity used during evaluation. Individual metric
failures produce a partial report so batch analysis can continue without hiding
which metric failed.
