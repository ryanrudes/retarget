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

::: motion_sync.video_schema.VideoSchema

## Contacts

::: motion_sync.contact_layer.ContactLayer

::: motion_sync.contact_registration.ContactType

::: motion_sync.contacts.categorical.CategoricalContact

::: motion_sync.contacts.binary.BinaryContact

::: motion_sync.contacts.foot_support.FootSupport

::: motion_sync.contacts.foot_support.FootSupportData

::: motion_sync.contact_metadata.stamp_detection_metadata

## Reference session (skate)

Concrete enums and bundled session live in `motion_sync.schemas.skateboarding`:

- `Bodies`, `LeftShoeMarkers`, `SmplxCoreJoints`
- `SKATE_MOCAP`, `SKATE_VIDEO`, `SKATE_FOOT_SUPPORT`, `SKATE_SESSION`

## Storage (advanced)

::: motion_sync.ViconRecording

Internal NPZ layout is not public; use `SyncClip.load` / `save`.
