"""Simulated gripper tactile sensor: PhysX contact -> a normal+shear force field.

Implements the **TacSL/FlexiTac force-field representation** (normal force R×C + shear
force R×C×2) natively on Isaac Sim 6.0's `RigidPrim` contact API — NO IsaacLab, no
GelSight gel asset, no Isaac 5.1 (see PROJECT.md §9 2026-06-30 for why the FlexiTac
fork itself isn't usable on this stack). We read per-contact world points + normal
forces (`get_contact_force_data`) and tangential forces (`get_friction_data`) on the
two Panda finger pads, project each onto the pad face, and bin into a GRID_ROWS×GRID_COLS
taxel grid summed over both pads. The grid renders to RGB and publishes as a
sensor_msgs/Image, so it flows into the recorder/dataset as just another camera (the
pipeline is modality-agnostic) — no schema/policy changes.

Import-safe (numpy only at top); the contact view + RigidPrim are imported post-boot.
"""
import numpy as np

from franka_scene import quat_to_rotmat  # world->local point = R.T @ (p - origin); vector = R.T @ v

GRID_ROWS, GRID_COLS = 32, 12           # 32 across the pad width, 12 along its length
FINGER_NAMES = ("panda_leftfinger", "panda_rightfinger")

# --- pad-face geometry (CALIBRATION) -------------------------------------------------
# Project a contact, in the finger-link LOCAL frame, onto the grasping face. The pad is
# OFFSET from the link origin (PAD_ORIGIN) — measured from the grasp contact cloud
# (x≈±0.011 spread, y≈0 normal, z≈0.0536 down the finger). Map: local X -> GRID_ROWS
# (across pad), local Z -> GRID_COLS (along finger), local Y = face normal (unused).
# Tuned 2026-06-30 against the auto-grasp contact data; re-measure if the URDF changes.
LONG_AXIS, WIDE_AXIS = 2, 0
PAD_ORIGIN = (0.0, 0.0, 0.0536)         # pad-contact center in the finger-link local frame (m)
PAD_LEN_M, PAD_WID_M = 0.020, 0.026     # full extents along LONG_AXIS (Z) / WIDE_AXIS (X)
FORCE_MAX_N = 6.0                       # normal force at full-scale colour (~5 N firm grasp)
SHEAR_MAX_N = 3.0                       # shear force at full-scale colour
PHYSICS_DT = 1.0 / 60.0                 # impulse -> force scale; match the sim physics step
PATCH = 2                               # taxels to grow the contact bbox by (fills the cube footprint)


def _cell(local):
    """Pad-local contact point -> (row, col) taxel index (clamped to the grid)."""
    u = (local[LONG_AXIS] - PAD_ORIGIN[LONG_AXIS]) / PAD_LEN_M + 0.5   # -> [0,1] along length
    v = (local[WIDE_AXIS] - PAD_ORIGIN[WIDE_AXIS]) / PAD_WID_M + 0.5   # -> [0,1] across width
    return (min(GRID_ROWS - 1, max(0, int(v * GRID_ROWS))),
            min(GRID_COLS - 1, max(0, int(u * GRID_COLS))))


def _bin(grid, local, force):
    """Accumulate a scalar `force` into the taxel under a pad-local point (in place)."""
    r, c = _cell(local)
    grid[r, c] += force


def _bbox(cells):
    """Bounding-box slices over taxel cells, grown by PATCH and clamped to the grid."""
    rows, cols = [r for r, _ in cells], [c for _, c in cells]
    r0, r1 = max(0, min(rows) - PATCH), min(GRID_ROWS - 1, max(rows) + PATCH)
    c0, c1 = max(0, min(cols) - PATCH), min(GRID_COLS - 1, max(cols) + PATCH)
    return (slice(r0, r1 + 1), slice(c0, c1 + 1))


def jet_rgb(t):
    """Jet colormap (dark blue -> cyan -> green -> yellow -> red), the FlexiTac/GelSight look.
    t in [0,1] -> (...,3) uint8 RGB; 0 force -> dark-blue background. Pure numpy (no cv2)."""
    t = np.clip(t, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * t - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * t - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * t - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def _gauss1d(sigma, radius):
    x = np.arange(-radius, radius + 1)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    return k / k.sum()


GAUSS = _gauss1d(1.5, 3)                 # separable kernel for the smooth blob falloff


def _blur(g):
    """Separable Gaussian blur of a 2D grid (pure numpy) for a smooth contact blob."""
    pad = len(GAUSS) // 2
    for axis in (0, 1):
        gp = np.pad(g, pad, mode="edge")
        acc = np.zeros_like(g)
        for i, w in enumerate(GAUSS):
            acc += w * (gp[i:i + g.shape[0], pad:pad + g.shape[1]] if axis == 0
                        else gp[pad:pad + g.shape[0], i:i + g.shape[1]])
        g = acc
    return g


def shear_rgb(normal, shear):
    """TacSL-style tactile shear image (cf. FlexiTac's compute_tactile_shear_image):
    R,G = in-plane shear vector (signed, centred at mid-grey); B = normal intensity."""
    sx = np.clip(0.5 + shear[..., 0] / (2.0 * SHEAR_MAX_N), 0.0, 1.0)
    sy = np.clip(0.5 + shear[..., 1] / (2.0 * SHEAR_MAX_N), 0.0, 1.0)
    b = np.clip(normal / FORCE_MAX_N, 0.0, 1.0)
    return (np.stack([sx, sy, b], axis=-1) * 255.0).astype(np.uint8)


def enable_contact_reporting(robot_path, cube_path):
    """Apply PhysxContactReportAPI to the finger pads + cube so the tensor contact view can
    read them (without it: 'Failed to find contact report API'). Must run BEFORE timeline.play()
    — PhysX reads the schema at physics init."""
    import isaacsim.core.experimental.utils.stage as stage_utils
    from pxr import PhysxSchema, Usd

    stage = stage_utils.get_current_stage()
    prims = [p for p in Usd.PrimRange(stage.GetPrimAtPath(robot_path)) if p.GetName() in FINGER_NAMES]
    prims.append(stage.GetPrimAtPath(cube_path))
    done = []
    for prim in prims:
        if prim and prim.IsValid():
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.0)
            done.append(prim.GetName())
    print(f"[tactile] contact reporting enabled on {done}", flush=True)


class TactileGrid:
    """Normal + shear contact-force field over the two finger pads (summed into one grid)."""

    def __init__(self, robot_path, filter_path, max_contacts=64, mode="normal", debug=False):
        import isaacsim.core.experimental.utils.stage as stage_utils
        from isaacsim.core.experimental.prims import RigidPrim
        from pxr import Usd

        self.mode = mode
        self.debug = debug
        self.frame = 0
        self.every = 3                    # ponytail: read contacts every Nth step (~20 Hz) — 3 GPU
        self.failed = False               #   syncs/step at 60 Hz stalls the sim; tactile needs no 60 Hz
        self._read_ok = False
        self._had_contact = False
        self._dbg_locals = []             # pad-local contact coords this frame (calibration diagnostic)
        self.last = (0, 0.0, 0.0)         # (contact count, max normal N, max shear N) — first-run diagnostic
        stage = stage_utils.get_current_stage()
        self.paths = sorted(str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath(robot_path))
                            if p.GetName() in FINGER_NAMES)           # sorted -> left pad, then right
        if len(self.paths) != 2:
            raise RuntimeError(f"expected 2 finger pads, found {self.paths}")
        print(f"[tactile] building contact view over {self.paths} (filter {filter_path}) ...", flush=True)
        self.prims = RigidPrim(self.paths, contact_filter_paths=[filter_path],
                               max_contact_count=max_contacts)
        print("[tactile] contact view constructed.", flush=True)

    def read(self, dt=PHYSICS_DT):
        """-> (normal (P,R,C) float32, shear (P,R,C,2) float32), one grid PER pad (P pads).
        Mirrors TacSL's tactile_normal_force / tactile_shear_force fields."""
        if not self._read_ok:
            print("[tactile] first contact read (if it hangs here, the contact API is the cause) ...",
                  flush=True)
        nf, npts, _no, _d, ncnt, nst = self.prims.get_contact_force_data(dt=dt)
        tf, tpts, tcnt, tst = self.prims.get_friction_data(dt=dt)
        nf, npts = nf.numpy().reshape(-1), npts.numpy().reshape(-1, 3)
        ncnt, nst = ncnt.numpy().reshape(len(self.paths), -1), nst.numpy().reshape(len(self.paths), -1)
        tf, tpts = tf.numpy().reshape(-1, 3), tpts.numpy().reshape(-1, 3)
        tcnt, tst = tcnt.numpy().reshape(len(self.paths), -1), tst.numpy().reshape(len(self.paths), -1)
        pos, quat = self.prims.get_world_poses()
        pos, quat = pos.numpy(), quat.numpy()

        npads = len(self.paths)
        normals = np.zeros((npads, GRID_ROWS, GRID_COLS), dtype=np.float32)
        shears = np.zeros((npads, GRID_ROWS, GRID_COLS, 2), dtype=np.float32)
        ncontacts, max_n, max_s = 0, 0.0, 0.0
        dbg = []
        for i in range(npads):
            Rt = quat_to_rotmat(quat[i]).T                          # world -> pad-local
            cells, pad_fmax = [], 0.0
            s, n = int(nst[i, 0]), int(ncnt[i, 0])                  # normal contacts (pair [i, 0])
            for k in range(s, s + n):
                f = float(nf[k])
                if f > 0.0:
                    local = Rt @ (npts[k] - pos[i])
                    cells.append(_cell(local))
                    ncontacts += 1
                    pad_fmax, max_n = max(pad_fmax, f), max(max_n, f)
                    if self.debug:
                        dbg.append((i, local))
            scells, svecs = [], []
            s, n = int(tst[i, 0]), int(tcnt[i, 0])                  # tangential (shear) contacts
            for k in range(s, s + n):
                t_local = Rt @ tf[k]                                # rotate the force vector (no translate)
                scells.append(_cell(Rt @ (tpts[k] - pos[i])))
                svecs.append((t_local[LONG_AXIS], t_local[WIDE_AXIS]))
                max_s = max(max_s, float(np.hypot(t_local[LONG_AXIS], t_local[WIDE_AXIS])))
            # Rigid contact reports are sparse (flickering points at the patch edges), but the cube
            # covers the whole pad -> fill the contacts' bounding box into a solid rectangle per pad.
            # ponytail: bbox-fill assumes a convex contact (true for the cube); revisit for odd shapes.
            if cells:
                normals[i][_bbox(cells)] = pad_fmax
            if scells:
                shears[i][_bbox(scells)] = np.mean(svecs, axis=0)
        self.last = (ncontacts, max_n, max_s)
        self._dbg_locals = dbg
        if not self._read_ok:
            print(f"[tactile] first read OK — {ncontacts} contacts this frame.", flush=True)
            self._read_ok = True
        return normals, shears

    def image(self, dt=PHYSICS_DT):
        """RGB uint8: the per-pad grids side by side (left | right), 1-px divider.
        'normal' = jet colormap + smooth blob (FlexiTac look); 'shear' = TacSL shear+normal."""
        normals, shears = self.read(dt)
        if self.mode == "shear":
            imgs = [shear_rgb(normals[i], shears[i]) for i in range(len(normals))]
        else:
            imgs = [jet_rgb(_blur(normals[i] / FORCE_MAX_N)) for i in range(len(normals))]
        sep = np.zeros((GRID_ROWS, 1, 3), dtype=np.uint8)           # divider between pads
        out = imgs[0]
        for im in imgs[1:]:
            out = np.concatenate([out, sep, im], axis=1)
        return out


def demo():
    """Self-check the pure binning + colormaps (no Isaac needed)."""
    c = np.array(PAD_ORIGIN)
    g = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)
    _bin(g, c, 5.0)                                                  # pad center -> center cell
    assert g[GRID_ROWS // 2, GRID_COLS // 2] == 5.0
    assert _cell(c + np.eye(3)[LONG_AXIS] * PAD_LEN_M)[1] == GRID_COLS - 1   # +full long -> last col
    assert _cell(c - np.eye(3)[WIDE_AXIS] * PAD_WID_M)[0] == 0               # -full wide -> row 0
    assert _bbox([(2, 6), (29, 6)]) == (slice(0, 32), slice(4, 9))           # PATCH=2 -> solid rectangle
    assert tuple(jet_rgb(0.0)) == (0, 0, 127) and tuple(jet_rgb(1.0)) == (127, 0, 0)  # blue -> red
    assert jet_rgb(0.5)[1] == 255                                                      # green at mid
    bg = np.zeros((5, 5), np.float32); bg[2, 2] = 1.0
    sb = _blur(bg)
    assert sb[2, 2] == sb.max() and 0 < sb[2, 2] < 1.0 and sb[1, 2] > 0               # smooth spread
    # shear image: no force -> mid-grey + dark blue; +long shear pushes R up, normal lights B
    n, s = np.zeros((1, 1), np.float32), np.zeros((1, 1, 2), np.float32)
    assert tuple(shear_rgb(n, s)[0, 0]) == (127, 127, 0)
    s[0, 0, 0], n[0, 0] = SHEAR_MAX_N, FORCE_MAX_N
    px = shear_rgb(n, s)[0, 0]
    assert px[0] > 127 and px[2] == 255
    print("tactile demo: cell-binning + normal/shear colormaps pass")


if __name__ == "__main__":
    demo()
