# Introduction

Retargeting experiments begin with observations, not synchronized files and not
robot-specific dictionaries.

```text
sensor/video/pose sources
-> native recordings
-> observation recipe
-> SceneObservation
-> retargeting recipe + RobotSpec
-> RetargetingProblem
-> RetargetingResult
```

## Native Data

Every modality keeps its own timeline and frame:

```python
from retarget import GvhmrOutputSource, ViconRecordingSource
from retarget.recipes.skateboarding import GVHMR_SCHEMA, VICON_SCHEMA

mocap = ViconRecordingSource(vicon_path, VICON_SCHEMA, name=demo)
human_pose = GvhmrOutputSource(
    gvhmr_path,
    GVHMR_SCHEMA,
    fps=59.942,
    name=demo,
)
```

Loading these sources produces `MocapRecording` and `HumanPoseRecording`.
Nothing claims they are synchronized. Tracks carry enum roles, validity masks,
and provenance.

## Observation Processing

An `ObservationRecipe` declares the modalities and fusion policy:

```python
from retarget.recipes.skateboarding import SkateboardingObservationRecipe

observation_recipe = SkateboardingObservationRecipe(
    mocap=mocap,
    human_pose=human_pose,
)
observation = observation_recipe.observe()
```

For skateboarding this estimates a `ClockTransform`, crops to valid overlap,
resamples each track according to its type, converts frames, registers the actor
spatially, reconstructs the board, and derives semantic contacts. Failed quality
gates raise `AlignmentError` with an `AlignmentReport`.

`SceneObservation` is independent of the target robot. It can be inspected or
reused:

```python
for report in observation.alignment_reports:
    print(report.strategy, report.score, report.accepted)

observation.save_npz("inspection_checkpoint.npz")
```

The save call is optional and explicit. The standard path remains in memory.

## Robot Adaptation

`RobotSpec` declares a robot-role vocabulary and binds roles to concrete joints,
links, and link groups. An adaptation recipe maps semantic observation roles to
those robot roles:

```python
from retarget.recipes.skateboarding import SkateboardingRetargetingRecipe

recipe = SkateboardingRetargetingRecipe()
problem = recipe.build_problem(observation, robot)
```

Only this stage creates `ContactPlan`, `LinkTargetPlan`, source-to-robot
mappings, objectives, and constraints. Semantic contacts never contain robot
link names.

## Experiment Orchestration

`RetargetingExperiment` composes both stages:

```python
from retarget import RetargetingExperiment

experiment = RetargetingExperiment(
    observation=observation_recipe,
    recipe=recipe,
    robot=robot,
)
result = experiment.run()
```

Pass an existing `SceneObservation` instead of `observation_recipe` to reuse
capture processing with another robot or profile.

Holosoma climbing follows the same hierarchy with one mocap source and no
temporal fusion:

```python
from retarget.recipes.holosoma import (
    HolosomaClimbObservationRecipe,
    HolosomaClimbRetargetingRecipe,
)
```

It is a strict subset of the architecture, not a separate adapter system.

## Declarative Frontend

Run configs serialize the same object graph:

```toml
[observation]
kind = "skateboarding"
vicon = "/data/vicon/demo"
gvhmr = "/data/gvhmr/demo"
video_fps = 59.942

[recipe]
kind = "skateboarding"
```

`RetargetingRunConfig.build_experiment()` returns the public experiment object.
There is no synchronization command or staged archive required by the CLI.
