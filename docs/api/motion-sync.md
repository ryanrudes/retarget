# motion_sync API

Typed multimodal trials: Vicon + video/SMPL-X on one timeline, optional contact layers.

Requires the [motion_sync](https://github.com/ryanrudes/motion_sync) package on `PYTHONPATH` (sibling clone or editable install). See [Workspace setup](../ecosystem/workspace-setup.md).

## Core types

::: motion_sync.SyncClip
    options:
      members:
        - load
        - save
        - register_mocap
        - register_bodies
        - register_video
        - register_contacts
        - body
        - marker
        - joint
        - core_joint_positions
        - contact
        - detect
        - contact_is_fresh
        - has_contact
        - export_vicon_bodies
        - frame_count
        - time_s

::: motion_sync.session.ClipSession

::: motion_sync.mocap_schema.MocapSchema

::: motion_sync.mocap_schema.validate_body_enum

::: motion_sync.video_schema.VideoSchema

## Tracks and metadata

NPZ-backed tracks and sync metadata exposed on loaded clips (`clip.vicon`, `clip.video`, `clip.metadata`) and through typed accessors (`clip.body`, `clip.joint`, `clip.markers`).

::: motion_sync.JointTrack

::: motion_sync.MarkerTracks

::: motion_sync.RigidBodyTrack

::: motion_sync.RigidBodyPose

::: motion_sync.SyncMetadata

::: motion_sync.ViconMocap

::: motion_sync.VideoSmplx

::: motion_sync.AxisConvention

::: motion_sync.QuaternionOrder

## Contacts

::: motion_sync.contact_registration.ContactSchema

::: motion_sync.contact_layer.ContactLayer

::: motion_sync.contact_registration.ContactType

::: motion_sync.contacts.categorical.CategoricalContact

::: motion_sync.contacts.binary.BinaryContact

::: motion_sync.contacts.foot_support.FootSupport

::: motion_sync.contacts.foot_support.FootSupportData

::: motion_sync.contact_metadata.stamp_detection_metadata

## Body marker plots

::: motion_sync.draw_body_markers_frame

::: motion_sync.plot_body_markers

## Reference session (skate)

Concrete enums and bundled session live in `motion_sync.schemas.skateboarding` (matches `configs/motion_sync.yaml` in the motion_sync repo).

::: motion_sync.schemas.skateboarding.Bodies

::: motion_sync.schemas.skateboarding.SmplxCoreJoints

Per-body marker enums (`LeftShoeMarkers`, `RightShoeMarkers`, `SkateboardMarkers`) are defined in the same module.

Bundled schemas and session:

- `SKATE_MOCAP` — `MocapSchema[Bodies]` with shoe and board marker enums
- `SKATE_VIDEO` — `VideoSchema.smplx_core(SmplxCoreJoints)`
- `SKATE_FOOT_SUPPORT` — `FootSupport` left/right/board body mapping
- `SKATE_SESSION` — `ClipSession(mocap=…, contacts=…, video=…)` for load/detect defaults

::: motion_sync.schemas.skateboarding.SKATE_SESSION

## Storage (advanced)

::: motion_sync.ViconRecording

Internal NPZ layout is not public; use `SyncClip.load` / `save`.

## See also

- [Custom schemas](../ecosystem/custom-schemas.md) — define your own `Bodies`, `MocapSchema`, and `ClipSession`
- [End-to-end pipeline (skateboarding)](../ecosystem/pipeline.md) — sync, foot-support detect, retarget
