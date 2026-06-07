# Scenes And Task Kinds

Scene construction belongs to the retargeting recipe. The observation supplies
target-independent objects, terrain, contacts, and landmarks; the recipe
chooses how those facts enter a robot problem.

| Task kind | Typical scene |
|-----------|---------------|
| `robot_only` | Ground and robot motion |
| `object_interaction` | Ground plus observed or configured objects |
| `climbing` | Terrain, holds, or climbing structures |

## Robot Only

```python
from retarget import RobotOnlySceneRecipe

scene_recipe = RobotOnlySceneRecipe(
    ground_range=(-1.0, 1.0),
    ground_size=15,
)
```

Pass this to `RoleRetargetingRecipe`. See
`examples/basic_robot_only.py` for the complete experiment.

## Object Interaction

For a configured object:

```python
from retarget import (
    ObjectSpec,
    ObjectTrajectory,
    SceneSpec,
    StaticSceneRecipe,
)

object_spec = ObjectSpec(
    name="box",
    sample_points=box_points,
    trajectory=ObjectTrajectory.identity(frame_count, fps=fps),
)
scene_recipe = StaticSceneRecipe(
    SceneSpec.object_interaction(object_spec)
)
```

For captured objects, an `ObservationRecipe` should instead create
`ObservedObject` with geometry and a `PoseTrack`. The adaptation recipe converts
that track into the runtime `ObjectTrajectory`.

See `examples/object_interaction.py` and the skateboarding recipes.

## Climbing

```python
from retarget import SceneSpec, StaticSceneRecipe, TerrainSpec

scene_recipe = StaticSceneRecipe(
    SceneSpec.climbing(
        terrain=TerrainSpec(name="holds", sample_points=hold_points)
    )
)
```

`HolosomaClimbObservationRecipe` demonstrates the captured-object version, and
`HolosomaClimbRetargetingRecipe` demonstrates robot-specific geometry policy.

## Semantic Contacts

`SemanticContactSequence` remains independent of robot links. Only a
retargeting recipe resolves each `ContactSubject` through robot roles into a
runtime `ContactPlan`.

This split is important when reusing one observation for robots with different
foot, hand, or collision-link layouts.
