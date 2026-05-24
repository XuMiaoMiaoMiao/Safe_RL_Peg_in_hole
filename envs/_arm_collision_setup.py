"""Runtime apply of PhysX CollisionAPI to iiwa arm links.

Why this exists
---------------
`assets/usd/dual_arm_iiwa/configuration/dual_arm_iiwa_robot.usd` (binary crate)
does NOT contain `UsdPhysics.CollisionAPI` on the arm links. As a result, the
mushroom-rl CollisionHelper sets up RigidContactView but PhysX has nothing
collidable on arm_link_*, so `epoch_absorb_physx` stays at 0 forever.

This module adds invisible capsule colliders to each arm link at runtime, sized
from the iiwa7 URDF (`~/Downloads/dual_arm_iiwa (1).xml`). The capsules are
conservative approximations of the link geometry: slightly larger than the
visual mesh so collision triggers reliably without being overly tight.

When to use
-----------
Only when `enable_physx_arm_collision=True` in env. Default is False to preserve
backward compatibility with all existing trained checkpoints / configs.

Architecture
------------
Called via monkey-patch of `CollisionHelper.prepare_env` so that:
  1. USD loads on `/World/envs/env_0/Robot`
  2. WE add capsule + CollisionAPI to env_0 arm links  (this module)
  3. mushroom CollisionHelper applies PhysxRigidBodyAPI + PhysxContactReportAPI
  4. Cloner.replicate_physics() copies everything to env_1..env_N
  5. RigidContactView registers and PhysX detects arm_L vs arm_R contacts
"""

from typing import Iterable, Tuple


# iiwa7 link capsule approximations: (radius_m, height_m, z_offset_m)
# - radius: slightly larger than actual link cross-section for reliable contact
# - height: distance from this link's origin (= its joint axis) to the next joint
# - z_offset: capsule center along the link's local Z, half the link length
# Calibrated from URDF joint origins:
#   A2 z=0.1925, A3 z=0.2075, A4 z=0.1925, A5 z=0.2075, A6 z=0.1925, A7 z=0.0925
# All capsules use Z axis (iiwa link Z is roughly along the limb).
_IIWA7_BASE = {
    # link_name (without /):  (radius, height, z_offset)
    "left_arm_link_1": (0.075, 0.18, 0.09),
    "left_arm_link_2": (0.075, 0.18, 0.10),
    "left_arm_link_3": (0.060, 0.18, 0.10),
    "left_arm_link_4": (0.060, 0.17, 0.10),
    "left_arm_link_5": (0.050, 0.16, 0.09),
    "left_arm_link_6": (0.050, 0.08, 0.05),
    "left_arm_link_7": (0.040, 0.04, 0.02),
}
IIWA7_CAPSULES = dict(_IIWA7_BASE)
IIWA7_CAPSULES.update({k.replace("left_", "right_"): v for k, v in _IIWA7_BASE.items()})


def _capsule_params_for(link_name_no_slash: str) -> Tuple[float, float, float] | None:
    """Returns (radius, height, z_offset) or None if no params known for this link."""
    return IIWA7_CAPSULES.get(link_name_no_slash)


def apply_arm_link_colliders(
    stage,
    robot_prim_path: str,
    link_paths: Iterable[str],
    *,
    strict: bool = True,
    visible: bool = False,
) -> int:
    """Add capsule collider child prims to each named arm link.

    Args:
        stage: pxr.Usd.Stage from `omni.usd.get_context().get_stage()`
        robot_prim_path: e.g. "/World/envs/env_0/Robot"
        link_paths: paths relative to robot, e.g. ["/left_arm_link_1", ...]
        strict: when True (default), raise RuntimeError if:
            - 0 capsules were applied (means USD prims weren't found, patch
              ran too early, or robot path is wrong), OR
            - n_skipped_invalid_prim > 0 (means some named arm links don't
              exist in USD — likely a USD/link naming mismatch).
            This converts silent-failure into loud failure so smoke test can
            actually verify the patch worked.
        visible: render the capsule proxies for debugging. Default False keeps
            training behaviour unchanged. When True, left-arm capsules are blue
            and right-arm capsules are red so the PhysX guard boundary is easy
            to inspect in visualize_policy.py.

    Returns:
        Number of links successfully equipped with collider.

    Raises:
        RuntimeError: in strict mode when the apply clearly failed.
    """
    # Lazy imports — pxr only available inside IsaacSim Python.
    from pxr import UsdGeom, UsdPhysics, PhysxSchema, Gf

    n_applied = 0
    n_skipped_no_params = 0
    n_skipped_invalid = 0
    n_skipped_already = 0
    invalid_paths = []

    for link in link_paths:
        link_name_no_slash = link.strip("/")
        params = _capsule_params_for(link_name_no_slash)
        if params is None:
            n_skipped_no_params += 1
            continue
        radius, height, z_offset = params

        link_prim_path = robot_prim_path + link
        link_prim = stage.GetPrimAtPath(link_prim_path)
        if not link_prim.IsValid():
            n_skipped_invalid += 1
            invalid_paths.append(link_prim_path)
            continue

        capsule_path = link_prim_path + "/CollisionProxy"
        existing = stage.GetPrimAtPath(capsule_path)
        if existing.IsValid():
            if visible:
                UsdGeom.Imageable(existing).MakeVisible()
            n_skipped_already += 1
            continue

        # Create capsule child prim. It is hidden by default and only shown for
        # explicit debug visualization.
        capsule = UsdGeom.Capsule.Define(stage, capsule_path)
        capsule.CreateRadiusAttr().Set(float(radius))
        capsule.CreateHeightAttr().Set(float(height))
        capsule.CreateAxisAttr().Set("Z")
        capsule.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, float(z_offset)))
        if visible:
            color = (
                Gf.Vec3f(0.05, 0.45, 1.0)
                if link_name_no_slash.startswith("left_")
                else Gf.Vec3f(1.0, 0.20, 0.05)
            )
            capsule.CreateDisplayColorAttr().Set([color])
            try:
                capsule.CreateDisplayOpacityAttr().Set([0.35])
            except Exception:
                pass
            UsdGeom.Imageable(capsule).MakeVisible()
        else:
            UsdGeom.Imageable(capsule).MakeInvisible()

        # Apply collision APIs to the capsule itself
        capsule_prim = capsule.GetPrim()
        UsdPhysics.CollisionAPI.Apply(capsule_prim)
        PhysxSchema.PhysxCollisionAPI.Apply(capsule_prim)

        n_applied += 1

    # Loud log — multiple writers (print + sys.stderr) since IsaacSim sometimes
    # swallows stdout from inside callbacks. Also written to a known location
    # for greppability after the run.
    msg = (
        f"[arm_collision_setup] applied={n_applied} "
        f"skipped_no_params={n_skipped_no_params} "
        f"skipped_invalid_prim={n_skipped_invalid} "
        f"skipped_already_exists={n_skipped_already} "
        f"under {robot_prim_path}"
    )
    print(msg, flush=True)
    import sys as _sys
    print(msg, file=_sys.stderr, flush=True)
    if invalid_paths:
        warn = (
            f"[arm_collision_setup] WARNING invalid prim paths "
            f"(USD probably doesn't have these links yet): {invalid_paths[:5]}"
            + ("..." if len(invalid_paths) > 5 else "")
        )
        print(warn, flush=True)
        print(warn, file=_sys.stderr, flush=True)

    if strict:
        # A 2nd prepare_env on the same stage (e.g. world.reset called again)
        # is a NORMAL path: collider prims already exist, so n_applied=0 but
        # n_skipped_already_exists covers them. Only fail when BOTH applied
        # AND already-exists are 0 — that means nothing was actually equipped.
        if n_applied == 0 and n_skipped_already == 0:
            raise RuntimeError(
                "[arm_collision_setup] applied=0 AND already_exists=0. Robot "
                f"USD prims not found at {robot_prim_path}/<arm_link>. Either "
                "patched_prepare_env ran before USD load (mushroom init order "
                f"changed?), or robot_prim_path '{robot_prim_path}' is wrong, "
                "or link names don't match URDF. Cannot proceed — PhysX "
                "arm-arm collision guard would silently never fire."
            )
        if n_skipped_invalid > 0:
            raise RuntimeError(
                f"[arm_collision_setup] {n_skipped_invalid} arm link prim(s) "
                f"missing from USD: {invalid_paths}. Either the USD asset is "
                "different from expected (renamed/removed links) or "
                "robot_prim_path is wrong. Fix the link list or USD before "
                "running with strict=True."
            )

    return n_applied


def apply_table_collider(
    stage,
    robot_prim_path: str,
    *,
    table_path: str = "/TableCollision",
    table_z: float = 0.0,
    table_size: float = 3.0,
    table_thickness: float = 0.04,
    strict: bool = True,
    visible: bool = False,
) -> int:
    """Add one per-env invisible PhysX table collider under the robot root.

    The project has no stable table prim in the USD asset.  The env's table
    clearance proxy uses an env-local plane at table_z; this function creates a
    matching thin box collider with its top face at table_z so absorbing table
    contact can be detected by PhysX instead of by the geometry proxy.
    """
    from pxr import UsdGeom, UsdPhysics, PhysxSchema, Gf

    full_path = robot_prim_path + table_path
    existing = stage.GetPrimAtPath(full_path)
    if existing.IsValid():
        if visible:
            UsdGeom.Imageable(existing).MakeVisible()
        else:
            UsdGeom.Imageable(existing).MakeInvisible()
        return 0

    cube = UsdGeom.Cube.Define(stage, full_path)
    cube.CreateSizeAttr().Set(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, float(table_z) - float(table_thickness) / 2.0))
    cube.AddScaleOp().Set(Gf.Vec3f(float(table_size), float(table_size), float(table_thickness)))
    if visible:
        cube.CreateDisplayColorAttr().Set([Gf.Vec3f(0.55, 0.55, 0.55)])
        try:
            cube.CreateDisplayOpacityAttr().Set([0.25])
        except Exception:
            pass
        UsdGeom.Imageable(cube.GetPrim()).MakeVisible()
    else:
        UsdGeom.Imageable(cube.GetPrim()).MakeInvisible()

    prim = cube.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    rigid = UsdPhysics.RigidBodyAPI.Apply(prim)
    try:
        rigid.CreateKinematicEnabledAttr().Set(True)
    except Exception:
        pass
    PhysxSchema.PhysxCollisionAPI.Apply(prim)
    PhysxSchema.PhysxRigidBodyAPI.Apply(prim)

    msg = (
        f"[arm_collision_setup] table_collider applied=1 path={full_path} "
        f"top_z={float(table_z):+.4f} size={float(table_size):.2f} "
        f"thickness={float(table_thickness):.3f}"
    )
    print(msg, flush=True)
    import sys as _sys
    print(msg, file=_sys.stderr, flush=True)

    if strict and not stage.GetPrimAtPath(full_path).IsValid():
        raise RuntimeError(f"[arm_collision_setup] failed to create table collider at {full_path}")
    return 1


def install_collision_helper_patch(
    arm_link_paths: Iterable[str],
    *,
    visible: bool = False,
    table_enabled: bool = False,
    table_path: str = "/TableCollision",
    table_z: float = 0.0,
    table_size: float = 3.0,
    table_thickness: float = 0.04,
    table_visible: bool = False,
) -> None:
    """Monkey-patch `mushroom_rl.utils.isaac_sim.CollisionHelper.prepare_env`
    so it runs our `apply_arm_link_colliders` BEFORE the original prepare_env
    (which only applies PhysxRigidBodyAPI + PhysxContactReportAPI).

    Per-env opt-in: the patch checks a class-level *enable counter* before
    applying capsules. `install_collision_helper_patch(...)` increments the
    counter; `uninstall_collision_helper_patch()` decrements. Capsule logic
    only runs when the counter is positive AND link_paths is set. This means
    in the same Python process you can construct one env with
    enable_physx_arm_collision=True and a later env with =False, and the
    second env will not get capsules.

    The patch itself stays installed (monkey-patches are cheap and removing
    them is fragile if other CollisionHelper instances were created).

    Args:
        arm_link_paths: list of arm link paths to equip, e.g.
            ["/left_arm_link_1", ..., "/right_arm_link_7"]
        visible: render debug capsule proxies instead of hiding them.
    """
    from mushroom_rl.utils.isaac_sim.collision_helper import CollisionHelper

    arm_link_paths = list(arm_link_paths)
    # Update / append the link list before incrementing the active counter so
    # the patched prepare_env (when fired) sees the latest link set.
    CollisionHelper._dual_arm_collider_link_paths = arm_link_paths
    CollisionHelper._dual_arm_collider_visible = bool(visible)
    CollisionHelper._dual_arm_table_enabled = bool(table_enabled)
    CollisionHelper._dual_arm_table_path = str(table_path)
    CollisionHelper._dual_arm_table_z = float(table_z)
    CollisionHelper._dual_arm_table_size = float(table_size)
    CollisionHelper._dual_arm_table_thickness = float(table_thickness)
    CollisionHelper._dual_arm_table_visible = bool(table_visible)
    active = getattr(CollisionHelper, "_dual_arm_collider_active", 0)
    CollisionHelper._dual_arm_collider_active = active + 1

    if getattr(CollisionHelper, "_dual_arm_collider_patched", False):
        return

    original_prepare_env = CollisionHelper.prepare_env

    def patched_prepare_env(self, stage):
        # Original mushroom logic always runs (applies PhysxRigidBodyAPI +
        # PhysxContactReportAPI to the listed prims regardless of capsules).
        # Capsule application is opt-in via the active counter — if no
        # currently-constructing env asked for it, skip cleanly.
        if getattr(CollisionHelper, "_dual_arm_collider_active", 0) > 0:
            robot_prim_path = self.ZERO_ENV_PATH + "/Robot"
            links_for_this_patch = getattr(
                CollisionHelper, "_dual_arm_collider_link_paths", []
            )
            did_prepare = False
            if links_for_this_patch:
                # strict=True raises if neither new nor existing colliders are
                # found — smoke test fails LOUDLY instead of silently going to
                # PhysX=0. apply_arm_link_colliders accepts already-existing
                # capsules as a success path (re-fire on same stage is fine).
                apply_arm_link_colliders(
                    stage,
                    robot_prim_path,
                    links_for_this_patch,
                    strict=True,
                    visible=bool(getattr(
                        CollisionHelper, "_dual_arm_collider_visible", False
                    )),
                )
                did_prepare = True
            if getattr(CollisionHelper, "_dual_arm_table_enabled", False):
                apply_table_collider(
                    stage,
                    robot_prim_path,
                    table_path=getattr(CollisionHelper, "_dual_arm_table_path", "/TableCollision"),
                    table_z=getattr(CollisionHelper, "_dual_arm_table_z", 0.0),
                    table_size=getattr(CollisionHelper, "_dual_arm_table_size", 3.0),
                    table_thickness=getattr(CollisionHelper, "_dual_arm_table_thickness", 0.04),
                    strict=True,
                    visible=bool(getattr(CollisionHelper, "_dual_arm_table_visible", False)),
                )
                did_prepare = True
            if did_prepare:
                # Count successful patch fires rather than newly-created prims, so
                # a re-fire on a stage with existing colliders still bumps the
                # counter (env post-assert checks delta of this counter).
                CollisionHelper._dual_arm_collider_fire_count = (
                    getattr(CollisionHelper, "_dual_arm_collider_fire_count", 0) + 1
                )
        return original_prepare_env(self, stage)

    CollisionHelper.prepare_env = patched_prepare_env
    CollisionHelper._dual_arm_collider_patched = True


def uninstall_collision_helper_patch() -> None:
    """Decrement the active counter so future env constructions without
    enable_physx_arm_collision skip the capsule applier.

    Idempotent — never goes below 0. Safe to call from env teardown / __del__
    if you want to be explicit. Not strictly required; the counter is process-
    local and harmless to leave positive (worst case: capsules also applied
    to an env that didn't request them, which is what we used to do).
    """
    from mushroom_rl.utils.isaac_sim.collision_helper import CollisionHelper

    active = getattr(CollisionHelper, "_dual_arm_collider_active", 0)
    CollisionHelper._dual_arm_collider_active = max(0, active - 1)
