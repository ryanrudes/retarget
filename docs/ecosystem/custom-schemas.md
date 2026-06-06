# Custom schemas for a new problem

Use this when your experiment has different Vicon body names, markers, SMPL-X usage, or contact types than the skate reference. The pattern is always: **define enums once**, bundle them in schemas, register on `SyncClip.load`, then read through typed accessors.

Reference implementation: `motion_sync/schemas/skateboarding.py` in the [motion_sync](https://github.com/ryanrudes/motion_sync) repository.

## 1. Rigid bodies (`StrEnum`)

Values must match `vicon__body_names` in `synced.npz` **exactly**.

```python
from enum import StrEnum

class Bodies(StrEnum):
    LEFT_SHOE = "Left_Shoe"
    RIGHT_SHOE = "Right_Shoe"
    TOOL = "My_Tool_Body"
```

Register alone (no markers on clip):

```python
clip = SyncClip.load(path).register_bodies(Bodies)
track = clip.body(Bodies.LEFT_SHOE)
positions = track.positions  # (frames, 3), Z-up
```

## 2. Markers (one `StrEnum` per body)

Logical names (e.g. `HEEL`) can repeat across feet; **values** are unique Vicon strings.

```python
class LeftShoeMarkers(StrEnum):
    HEEL = "Unlabeled20171"
    TOE = "Unlabeled18658"
```

Bundle in `MocapSchema`:

```python
from motion_sync.mocap_schema import MocapSchema

MY_MOCAP = MocapSchema(
    bodies=Bodies,
    body_markers={
        Bodies.LEFT_SHOE: LeftShoeMarkers,
        Bodies.RIGHT_SHOE: RightShoeMarkers,
    },
)
```

`register_mocap` validates that marker enums **partition** all marker names on the clip. Requires marker tracks in `synced.npz` (from paired `vicon.npz` at load).

```python
clip = SyncClip.load(path, mocap=MY_MOCAP)
clip.marker(LeftShoeMarkers.HEEL)
```

## 3. Video / SMPL-X joints (`VideoSchema`)

GVHMR FK stores **full** SMPL-X joint indices in `video__joints`. Map logical names to indices and to retarget’s 20-joint **core** order if you export to `smplx` format.

```python
from enum import StrEnum
from motion_sync.video_schema import VideoSchema, SMPLX_CORE_JOINT_NAMES

class CoreJoints(StrEnum):
  PELVIS = "Pelvis"
  L_FOOT = "L_Foot"
  # ... same names and order as retarget MotionFormat.SMPLX ...

MY_VIDEO = VideoSchema.smplx_core(CoreJoints)
```

`VideoSchema.smplx_core` checks enum order against `SMPLX_CORE_JOINT_NAMES` and applies standard GVHMR index table.

```python
clip = clip.register_video(MY_VIDEO)
clip.joint(CoreJoints.L_FOOT).positions       # (T, 3), Y-up
core = clip.core_joint_positions()            # (T, 20, 3) for fuse / export
```

## 4. Contact types

### Categorical (multi-state per subject)

Subclass `CategoricalContact`, set `layer_id` and `State` (`IntEnum`), implement `detect` and `read`.

```python
from enum import IntEnum
from motion_sync.contacts.categorical import CategoricalContact, CategoricalContactData

class SupportState(IntEnum):
    AIR = 0
    GROUND = 1

class FootSupport(CategoricalContact[SupportState, FootSupportData]):
    layer_id = "foot_support"
    State = SupportState
    left: BodyT
    right: BodyT

    def detect(self, clip, config=None):
        # call contact_detection or your own logic
        ...
        return self.build_layer(subjects=(...), states=array, metadata={...})

    def read(self, clip, layer) -> FootSupportData:
        self._validate_layer(layer)
        ...
```

On disk, categorical layers use lowercase label strings (`air`, `ground`, …) from enum member names.

### Binary (mask per subject)

Subclass `BinaryContact` and use `build_layer(subjects=..., mask=...)`. See `ShoeBoardGrip` for a **derived** layer built from an existing categorical layer.

## 5. `ContactSchema` and `ClipSession`

```python
from motion_sync.contact_registration import ContactSchema
from motion_sync.session import ClipSession

MY_FOOT_SUPPORT = FootSupport(left=Bodies.LEFT_SHOE, right=Bodies.RIGHT_SHOE, board=Bodies.TOOL)
MY_CONTACTS = ContactSchema(types=(MY_FOOT_SUPPORT,))

MY_SESSION = ClipSession(mocap=MY_MOCAP, contacts=MY_CONTACTS, video=MY_VIDEO)
```

One-shot load:

```python
clip = SyncClip.load("output/synced/my_demo", session=MY_SESSION)
```

## 6. Detection, freshness, and CLI

```python
if not clip.contact_is_fresh(MY_FOOT_SUPPORT):
    clip = clip.detect(MY_FOOT_SUPPORT).save("output/synced/my_demo")
data = clip.contact(MY_FOOT_SUPPORT)
```

CLI (after registering a Typer command or using foot-support as template):

```bash
uv run motion-sync detect foot-support output/synced/my_demo --force
```

Metadata (`source_frame_count`, `time_fingerprint`) lets `contact_is_fresh` skip redundant work after re-sync.

## 7. Vendor `configs/motion_sync.yaml`

Keep **upstream sensor/session** strings in the vendor YAML, not in retarget semantics:

- `bodies.<name>.markers` — Vicon marker lists for rigid-body fitting
- `time_sync_solver.smplx_joints` — maps Vicon body paths to SMPL-X joint names for foot-speed sync

Python enums must stay consistent with those strings at the `motion_sync` boundary. Retargeting recipes should consume typed objects (`MotionSequence`, `SceneSpec`, `ContactPlan`, `PreparedRetargetingInputs`) after that boundary.

## 8. Wire into retarget

`retarget` core does not import your `ClipSession`. Integration adapter responsibilities:

1. Load `SyncClip` with your session.
2. Run detectors; save clip.
3. Convert `core_joint_positions()` to Z-up if needed.
4. Build `MotionSequence`, `SceneSpec`, `ContactPlan`, and `LinkTargetPlan`.
5. Return a `PreparedRetargetingInputs` bundle.
6. Add a typed `RetargetingSource` / `RetargetingRecipe` when the adapter should be available from Python and `retarget run`.

See [End-to-end pipeline](pipeline.md) and [Introduction](../introduction.md).

## Checklist for a new project

- [ ] `Bodies` values match synced `vicon__body_names`
- [ ] Marker enums partition clip marker names (if using markers)
- [ ] `VideoSchema` indices fit `video__joints` width from FK
- [ ] Contact `layer_id` is unique; labels match `State` enum
- [ ] `ClipSession` passed to `SyncClip.load` in notebooks and CLIs
- [ ] Export script produces retarget motion format + contact joint names
- [ ] Run config `format`, `contact_joints`, and scene paths aligned with export
