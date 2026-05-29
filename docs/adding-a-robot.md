# Add A Robot

Create a `RobotSpec` with joint names, height, limits, contact links, optional asset paths, and mappings for both actuated joints and retargeted links.

`default_joint_mapping` maps motion joints to actuated robot joints for nominal tracking or simple IK-style terms. `default_link_mapping` maps motion joints to robot body/link points used by the interaction mesh.

```python
from retarget.robots import RobotSpec, robots

@robots.register("my_robot")
def my_robot() -> RobotSpec:
    return RobotSpec(
        name="my_robot",
        dof=2,
        height_m=1.0,
        joint_names=("hip", "knee"),
        link_names=("pelvis", "left_foot"),
        contact_links=("left_foot",),
        joint_limits={"hip": (-1.0, 1.0), "knee": (-2.0, 0.0)},
    )
```

For asset-backed families, implement `RobotProvider` and register it in `robot_providers` so the provider can resolve local manifests or optional downloads while the high-level API still receives a validated `RobotSpec`.

Built-in `g1_like` and `t1_like` specs are readable humanoid templates with named joints, contact links, default motion mappings, and conservative limits. They are useful for experiments with the simple backend or as starting points for local asset-backed specs; for real simulator runs, load a project-specific spec that points at your URDF or MJCF.

When using the MuJoCo backend, limited hinge and slide ranges are read from the MJCF model. `joint_limits` in the `RobotSpec` act as explicit overrides, so you can tighten simulator ranges or patch an asset without editing the upstream file.

Robot specs can also live outside Python code:

```toml
name = "my_robot"
dof = 2
height_m = 1.0
joint_names = ["hip", "knee"]
link_names = ["pelvis", "left_foot"]
contact_links = ["left_foot"]

[joint_limits]
hip = [-1.0, 1.0]
knee = [-2.0, 0.0]
```

Load one directly:

```python
from retarget.robots import robot_providers

robot = robot_providers.get("file").load("my_robot", path="examples/custom_robot.toml")
```

Or install a robot directory/file into an asset store and resolve it by name:

```bash
retarget assets import /path/to/robot_dir --name my_robot --kind robot
```

```python
robot = robot_providers.get("asset_store").load("my_robot", store=".retarget_assets")
```

For directories, the asset-store provider looks for `robot.toml`, `robot.yaml`, `robot.yml`, or `robot.json` unless `spec_filename` is supplied.

CLI run specs use the same provider system:

```toml
robot = "my_robot"
robot_provider = "file"

[robot_options]
path = "examples/custom_robot.toml"
```

For an asset store:

```toml
robot = "my_robot"
robot_provider = "asset_store"

[robot_options]
store = ".retarget_assets"
```

Relative provider paths are resolved against the run config file.
