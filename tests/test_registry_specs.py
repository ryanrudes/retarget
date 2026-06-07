from enum import StrEnum

import pytest

from retarget.core.enums import (
    HumanoidRobotRole,
    RegistryKind,
    Robot,
    RobotGeometry,
    RobotJoint,
    RobotLink,
    RobotRole,
)
from retarget.core.registry import Registry
from retarget.robots import RobotSpec, robots
from retarget.scene import ObjectSpec, SceneSpec, TerrainSpec
from tests.typed_fixtures import (
    FixtureRobotGeometry,
    FixtureRobotJoint,
    FixtureRobotLink,
    FixtureRobotRole,
    SameValueRobotJoint,
    fixture_robot,
)


class NumberKey(RegistryKind):
    ONE = "one"
    TWO = "two"


class SameValueNumberKey(RegistryKind):
    ONE = "one"


def test_registry_programmatic_lookup_requires_exact_enum_identity():
    registry = Registry[RegistryKind, int]("number", RegistryKind)
    registry.register(NumberKey.ONE, 1)

    assert registry.get(NumberKey.ONE) == 1
    assert registry.get_serialized("one") == 1
    assert registry.key_from_serialized("one") is NumberKey.ONE
    with pytest.raises(KeyError):
        registry.get(SameValueNumberKey.ONE)
    with pytest.raises(KeyError):
        registry.register(SameValueNumberKey.ONE, 2)
    with pytest.raises(TypeError):
        registry.get("one")  # type: ignore[arg-type]


def test_builtin_robot_specs_preserve_concrete_vocabularies():
    robot = robots.get(Robot.SYNTHETIC_HUMANOID)

    assert issubclass(robot.vocabulary.joints, RobotJoint)
    assert issubclass(robot.vocabulary.links, RobotLink)
    assert issubclass(robot.vocabulary.geometries, RobotGeometry)
    assert robot.vocabulary.roles is HumanoidRobotRole
    assert all(type(joint) is robot.vocabulary.joints for joint in robot.joints)
    assert all(type(link) is robot.vocabulary.links for link in robot.links)


def test_robot_spec_rejects_equal_member_from_wrong_vocabulary():
    robot = fixture_robot()

    with pytest.raises(TypeError):
        robot.joint_index(SameValueRobotJoint.ROOT)  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        RobotSpec(
            **{
                **robot.model_dump(),
                "joints": (SameValueRobotJoint.ROOT, *robot.joints[1:]),
            }
        )


def test_robot_spec_checkpoint_reconstructs_qualified_enum_classes(tmp_path):
    path = tmp_path / "robot.json"
    fixture_robot().save_json(path)

    restored = RobotSpec.load(path)

    assert restored.vocabulary.joints is FixtureRobotJoint
    assert restored.vocabulary.links is FixtureRobotLink
    assert restored.vocabulary.geometries is FixtureRobotGeometry
    assert restored.vocabulary.roles is FixtureRobotRole
    assert restored.joints == tuple(FixtureRobotJoint)
    assert restored.link_for_role(FixtureRobotRole.LEFT_FOOT) is FixtureRobotLink.LEFT_FOOT


def test_robot_spec_serialized_strings_resolve_only_through_declared_vocabulary():
    payload = fixture_robot().model_dump(mode="json", exclude={"vocabulary"})
    payload["vocabulary"] = {
        "joints": "tests.typed_fixtures:FixtureRobotJoint",
        "links": "tests.typed_fixtures:FixtureRobotLink",
        "geometries": "tests.typed_fixtures:FixtureRobotGeometry",
        "roles": "tests.typed_fixtures:FixtureRobotRole",
    }

    restored = RobotSpec.model_validate(payload)

    assert restored.joints[0] is FixtureRobotJoint.ROOT
    assert next(iter(restored.joint_roles)) is FixtureRobotRole.PELVIS


@pytest.mark.parametrize(
    ("factory", "key"),
    (
        (lambda provenance: ObjectSpec(name="object", provenance=provenance), "mesh_path"),
        (lambda provenance: TerrainSpec(provenance=provenance), "sample_points"),
        (
            lambda provenance: SceneSpec(
                task_kind=SceneSpec.robot_only().task_kind,
                terrain=TerrainSpec(),
                provenance=provenance,
            ),
            "ground_size",
        ),
    ),
)
def test_scene_provenance_rejects_behavior(factory, key):
    with pytest.raises(ValueError):
        factory({key: "behavior"})


def test_user_vocabulary_bases_are_empty_extension_points():
    for base in (RobotJoint, RobotLink, RobotGeometry, RobotRole):
        assert issubclass(base, StrEnum)
        assert tuple(base) == ()
