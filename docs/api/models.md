# Data models

Typed specs and results passed through the pipeline.

## Results

::: retarget.results.spec.RetargetingResult

::: retarget.results.spec.EvaluationReport

::: retarget.results.spec.EvaluationManifest

::: retarget.results.spec.EvaluationRecord

## Scene & motion

::: retarget.scene.spec.SceneSpec

::: retarget.scene.spec.ObjectSpec

::: retarget.scene.spec.ObjectTrajectory

::: retarget.scene.spec.TerrainSpec

::: retarget.motion.spec.MotionSequence
    options:
      members:
        - name
        - joint_names
        - joint_positions
        - fps
        - root_poses
        - metadata
        - contacts

::: retarget.motion.spec.MotionFormatSpec
    options:
      members:
        - joint_names

::: retarget.motion.loaders.load_motion

## Robots

::: retarget.robots.spec.RobotSpec

::: retarget.robots.spec.JointLimit

::: retarget.robots.spec.QposLayout

## Mesh

::: retarget.mesh.interaction.InteractionMeshBuilder

::: retarget.mesh.interaction.InteractionMeshSpec

::: retarget.mesh.interaction.MeshTopology
