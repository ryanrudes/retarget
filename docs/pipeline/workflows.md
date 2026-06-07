# End-To-End Workflows

## In Memory

`examples/in_memory_capture.py` creates a typed `SceneObservation` and runs it
through `RetargetingExperiment` without any intermediate file.

## Skateboarding

```python
experiment = RetargetingExperiment(
    observation=SkateboardingObservationRecipe(
        mocap=ViconRecordingSource(vicon_path, VICON_SCHEMA, name=demo),
        human_pose=GvhmrOutputSource(
            gvhmr_path,
            GVHMR_SCHEMA,
            fps=video_fps,
            name=demo,
        ),
    ),
    recipe=SkateboardingRetargetingRecipe(),
    robot=robot,
    retargeter=retargeter,
)
result = experiment.run()
```

The observation recipe estimates clock alignment from typed foot cues, selects
the shared timeline, resamples each track by its own interpolation policy,
converts frames, performs rigid point registration, reconstructs the board, and
derives semantic contacts. The adaptation recipe resolves those semantics
through robot roles.

Raw video can replace an existing pose output with `VideoPoseSource` and a
`HumanPoseEstimator`. `GvhmrEstimator` is an optional implementation that uses
a temporary workspace.

## Holosoma

Holosoma climbing uses the same hierarchy with one mocap source and no temporal
fusion:

```python
experiment = RetargetingExperiment(
    observation=HolosomaClimbObservationRecipe.from_fixture(root),
    recipe=HolosomaClimbRetargetingRecipe(holosoma_root=root),
    robot=g1_spherehand_robot(root),
    retargeter_factory=mujoco_retargeter,
)
```

It is a specialization of the same observation, adaptation, compilation, and
execution interfaces.
