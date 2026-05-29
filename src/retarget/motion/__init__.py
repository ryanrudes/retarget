"""Motion specs, registries, and loaders."""

from retarget.motion.contact import infer_contact_by_velocity
from retarget.motion.loaders import CsvMotionLoader, JsonMotionLoader, NpyMotionLoader, NpzMotionLoader, load_motion
from retarget.motion.registry import motion_formats, motion_loaders
from retarget.motion.spec import MotionFormatSpec, MotionSequence

__all__ = [
    "CsvMotionLoader",
    "JsonMotionLoader",
    "MotionFormatSpec",
    "MotionSequence",
    "NpyMotionLoader",
    "NpzMotionLoader",
    "infer_contact_by_velocity",
    "load_motion",
    "motion_formats",
    "motion_loaders",
]
