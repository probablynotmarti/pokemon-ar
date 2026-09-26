"""AR tests (need OpenCV: run with ~/.venvs/memecv/bin/python)."""
import unittest

try:
    import cv2
    import numpy as np
except ImportError:
    raise unittest.SkipTest("OpenCV not installed (use ~/.venvs/memecv)")

from game.ar.pose import PoseTracker, camera_matrix, focal_from_homography
from game.vision.board import BOARD_MM, board_corners_mm


def synthetic_camera(f=1450.0, size=(1920, 1080), tilt_deg=55, dist=900):
    """A camera on your side of the table, looking down at the board
    centre at `tilt_deg` from horizontal (board: x right, y toward you,
    z into the table)."""
    K = camera_matrix(f, size)
    t = np.radians(tilt_deg)
    centre = np.array([BOARD_MM / 2, BOARD_MM / 2, 0.0])
    cam = centre + np.array([0, dist * np.cos(t), -dist * np.sin(t)])
    zc = (centre - cam) / np.linalg.norm(centre - cam)   # looking direction
    xc = np.array([1.0, 0, 0])                           # board x = image right
    yc = np.cross(zc, xc)                                # image down
    R = np.stack([xc, yc, zc])
    tvec = -R @ cam
    return K, cv2.Rodrigues(R)[0], tvec.reshape(3, 1)


def homography_of(K, rvec, tvec):
    obj = np.hstack([board_corners_mm(), np.zeros((4, 1), np.float32)])
    img, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
    return cv2.getPerspectiveTransform(board_corners_mm(),
                                       np.float32(img.reshape(-1, 2)))


class Pose(unittest.TestCase):
    def test_focal_recovered_from_board(self):
        for f in (1100.0, 1450.0, 1800.0):
            for tilt in (35, 55, 70):
                K, r, t = synthetic_camera(f=f, tilt_deg=tilt)
                est = focal_from_homography(homography_of(K, r, t), (1920, 1080))
                self.assertAlmostEqual(est, f, delta=f * 0.02, msg=(f, tilt))

    def test_pose_reprojects_a_point_above_the_board(self):
        K, r, t = synthetic_camera()
        H = homography_of(K, r, t)
        Ke, re, te = PoseTracker().update(H, (1920, 1080))
        top = np.float32([[254, 381, -60]])          # 60 mm above your half
        want, _ = cv2.projectPoints(top, r, t, K, None)
        got, _ = cv2.projectPoints(top, re, te, Ke, None)
        self.assertLess(np.abs(want - got).max(), 3.0)

    def test_up_is_toward_the_camera(self):
        K, r, t = synthetic_camera()
        R = cv2.Rodrigues(r)[0]
        cam_pos = (-R.T @ t).ravel()
        self.assertLess(cam_pos[2], 0)               # camera sits at −z (above)


if __name__ == "__main__":
    unittest.main()


from game.ar.figurines import BUILDERS, FLOATS  # noqa: E402
from game.ar.loader import get_mesh, load_model  # noqa: E402
from game.ar.render import Placement, draw_mesh, yaw_towards  # noqa: E402
from game.ar.stage import BATTLE_SCALE, Stage  # noqa: E402
from game.battle import Event  # noqa: E402


class Figurines(unittest.TestCase):
    def test_every_card_has_a_sane_figurine(self):
        from game.data import POKEDEX
        self.assertEqual(set(BUILDERS), set(POKEDEX))
        for key, build in BUILDERS.items():
            m = build()
            with self.subTest(key=key):
                self.assertTrue((m.faces >= 0).all() and
                                (m.faces < len(m.verts)).all())
                self.assertEqual(len(m.colors), len(m.faces))
                self.assertTrue(45 <= m.verts[:, 2].max() <= 90)   # tall enough
                # feet may sink ≤ 1 mm into the card; floaters hover
                self.assertGreaterEqual(m.verts[:, 2].min(),
                                        -1 if key not in FLOATS else 5)
                w = m.verts[:, 0].max() - m.verts[:, 0].min()
                self.assertLess(w, 80)                  # about a card wide (63 mm)

    def test_faces_point_outward(self):
        # summed signed volume of a closed outward mesh is positive
        m = BUILDERS["seedot"]()
        v = m.verts[m.faces]
        vol = (np.cross(v[:, 0], v[:, 1]) * v[:, 2]).sum() / 6
        self.assertGreater(vol, 0)


class Render(unittest.TestCase):
    def test_draws_on_the_right_card(self):
        K, r, t = synthetic_camera()
        img = np.zeros((1080, 1920, 3), np.uint8)
        draw_mesh(img, BUILDERS["registeel"](), Placement(254, 380), (K, r, t))
        ys, xs = np.nonzero(img.sum(2))
        foot, _ = cv2.projectPoints(np.float32([[254, 380, 0]]), r, t, K, None)
        fx, fy = foot.ravel()
        self.assertLess(abs(xs.mean() - fx), 40)          # centred on the card
        self.assertLess(ys.max(), fy + 25)                # stands on it…
        self.assertLess(ys.min(), fy - 40)                # …and rises up

    def test_facing_away_hides_the_face(self):
        K, r, t = synthetic_camera()
        def red_pixels(yaw):
            img = np.zeros((1080, 1920, 3), np.uint8)
            draw_mesh(img, BUILDERS["registeel"](), Placement(254, 300, yaw),
                      (K, r, t))
            b, g, rr = img[..., 0].astype(int), img[..., 1], img[..., 2]
            return int(((rr > 120) & (b < 90)).sum())      # the red eye-dots
        front, back = red_pixels(0.0), red_pixels(np.pi)
        self.assertGreater(front, 150)                    # facing you
        # from behind and above only the top dots peek over the dome
        self.assertLess(back, 0.5 * front)

    def test_yaw_towards(self):
        self.assertAlmostEqual(yaw_towards(0, 0, 0, 10), 0.0)      # +y
        self.assertAlmostEqual(abs(yaw_towards(0, 0, 0, -10)), np.pi)


class Loader(unittest.TestCase):
    def test_real_model_replaces_figurine(self):
        import tempfile
        import trimesh
        with tempfile.TemporaryDirectory() as d:
            # a 2 x 4 x 1 box, glTF style: +Y up (4 tall), facing +Z
            box = trimesh.creation.box(extents=(2, 4, 1))
            box.visual.face_colors = [255, 0, 0, 255]         # red
            box.export(f"{d}/riolu.glb")
            m = get_mesh("riolu", d)
            self.assertAlmostEqual(m.height, 70, delta=0.5)   # scaled, z-up
            self.assertAlmostEqual(m.verts[:, 2].min(), 0, delta=1e-3)
            w = m.verts[:, 0].max() - m.verts[:, 0].min()
            self.assertAlmostEqual(w, 35, delta=0.5)          # 2:4 kept
            self.assertTrue((m.colors[:, 2] > 200).all())     # red → BGR
            self.assertEqual(len(get_mesh("seedot", d).faces),
                             len(BUILDERS["seedot"]().faces))  # no file: figurine

    def test_broken_file_falls_back(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            open(f"{d}/ducklett.glb", "wb").write(b"not a model")
            m = get_mesh("ducklett", d)
            self.assertEqual(len(m.faces), len(BUILDERS["ducklett"]().faces))


def ev(kind, side, **data):
    return Event(kind, "", side, data)


WHERE = {("player", 1): (85, 318), ("cpu", 1): (85, 64),
         ("player", 2): (254, 318), ("cpu", 4): (85, 190)}


class StageAnimation(unittest.TestCase):
    def setUp(self):
        self.s = Stage()
        self.s.start_battle({"player": {1: "riolu", 2: "seedot"},
                             "cpu": {1: "politoed", 4: "solrock"}})
        self.s.send_out("cpu", 1, 0.0)
        self.s.send_out("player", 1, 0.0)

    def at(self, now):
        return {p.y: (k, p) for k, p, *_ in
                self.s.placements(now, lambda s, n: WHERE[(s, n)])}

    def test_send_out_grows_then_settles(self):
        start = self.at(0.0)[318][1].scale
        self.assertLess(start, 0.1)
        self.assertAlmostEqual(self.at(2.0)[318][1].scale, BATTLE_SCALE)

    def test_figurines_face_each_other(self):
        pl = self.at(2.0)
        self.assertAlmostEqual(abs(pl[318][1].yaw), np.pi, delta=0.05)  # away
        self.assertAlmostEqual(pl[64][1].yaw, 0, delta=0.05)            # at you

    def test_far_drawn_first(self):
        ys = [p.y for _, p, *_ in self.s.placements(2.0, lambda s, n: WHERE[(s, n)])]
        self.assertEqual(ys, sorted(ys))

    def test_attack_lunges_toward_the_foe_and_comes_back(self):
        self.s.on_event(ev("move", "player", move="Force Palm"), 2.0)
        mid = self.at(2.0 + 0.275)
        self.assertLess(min(mid), 318 - 20)                # moved toward CPU
        self.assertIn(318, self.at(3.0))                   # back home

    def test_hit_flashes_and_faint_disappears(self):
        self.s.on_event(ev("damage", "cpu", slot=1), 2.0)
        flashes = [self.at(2.0 + i / 30)[64][1].flash for i in range(12)]
        self.assertGreater(max(flashes), 0.2)        # visibly brighter…
        self.assertLess(max(flashes), 0.5)           # …never a white-out
        self.s.on_event(ev("faint", "cpu", slot=1), 3.0)
        self.at(3.95)                                      # animation ends
        self.assertNotIn(64, self.at(4.0))

    def test_switch_swaps_the_figurine(self):
        self.s.on_event(ev("switch_out", "player", slot=1), 2.0)
        self.s.on_event(ev("switch_in", "player", slot=2), 2.5)
        keys = {k for k, _ in self.at(4.0).values()}
        self.assertIn("seedot", keys)
        self.assertNotIn("riolu", keys)

    def test_status_tints(self):
        pl = self.s.placements(2.0, lambda s, n: WHERE[(s, n)],
                               lambda s, n: "par" if s == "cpu" else None)
        cpu = [p for k, p, *_ in pl if k == "politoed"][0]
        self.assertGreater(cpu.tint_amount, 0.1)

    def test_setup_mode_shows_every_recognized_card(self):
        s = Stage()
        s.sync_setup({("player", 1): "riolu", ("cpu", 4): "solrock"}, 0.0)
        got = s.placements(2.0, lambda a, b: WHERE[(a, b)])
        self.assertEqual({k for k, *_ in got}, {"riolu", "solrock"})
        s.sync_setup({("player", 1): "riolu"}, 2.1)          # card misread…
        self.assertEqual(len(s.placements(2.5, lambda a, b: WHERE[(a, b)])), 2)
        s.sync_setup({("player", 1): "riolu", ("cpu", 4): "solrock"}, 2.6)
        solrock = [p for k, p, *_ in s.placements(2.7, lambda a, b: WHERE[(a, b)])
                   if k == "solrock"][0]
        self.assertAlmostEqual(solrock.scale, 1.0)           # …no blink / re-pop
        s.sync_setup({("player", 1): "riolu"}, 3.0)          # really removed
        s.sync_setup({("player", 1): "riolu"}, 4.1)
        self.assertEqual(len(s.placements(4.2, lambda a, b: WHERE[(a, b)])), 1)


from game.ar import move_fx  # noqa: E402
from game.ar.fx import FxSystem, fx_image  # noqa: E402
from game.ar.sprites import billboard_quad, camera_position, get_sprite  # noqa: E402
from game.data import MOVES  # noqa: E402

A, B = (254.0, 318.0), (254.0, 190.0)


class MoveEffects(unittest.TestCase):
    def test_every_move_has_an_effect(self):
        self.assertEqual(set(MOVES) | {"Struggle"}, set(move_fx.MOVE_FX))

    def test_every_effect_image_exists(self):
        if fx_image("fireball") is None:
            self.skipTest("effect images not downloaded (git-ignored)")
        used = set()
        for move in move_fx.MOVE_FX:
            fx = FxSystem()
            move_fx.launch(fx, move, A, B, 0.0)
            move_fx.impact(fx, move, B, 1.2)
            used |= {p.img for p in fx.particles}
        for f in (move_fx.status_fx, ):
            for st in ("par", "slp", "frz", "brn", "seeded"):
                fx = FxSystem()
                f(fx, st, B, 0.0)
                used |= {p.img for p in fx.particles}
        missing = sorted(n for n in used if fx_image(n) is None)
        self.assertEqual(missing, [])

    def test_projectile_reaches_the_target_on_the_beat(self):
        fx = FxSystem()
        move_fx.launch(fx, "Shadow Ball", A, B, 0.0)
        ball = fx.particles[0]
        (x, y, h), *_ = ball.at(move_fx.ARRIVE - 1e-3)
        self.assertLess(abs(y - B[1]), 3)
        self.assertGreater(ball.at(0.5)[0][1], B[1])        # still on its way

    def test_rocks_fall_onto_the_target(self):
        fx = FxSystem()
        move_fx.launch(fx, "Rock Slide", A, B, 0.0)
        rock = fx.particles[0]
        start, end = rock.path(0.0), rock.path(1.0)
        self.assertGreater(start[2], 100)                   # high above
        self.assertLess(end[2], 15)                         # lands on the card
        self.assertLess(abs(end[1] - B[1]), 35)


class StageEffects(unittest.TestCase):
    def setUp(self):
        self.s = Stage()
        self.s.start_battle({"player": {1: "politoed"}, "cpu": {1: "registeel"}})
        where = lambda s, n: WHERE[(s, n)]
        self.s.placements(0.0, where)
        self.s.send_out("player", 1, 0.0)
        self.s.send_out("cpu", 1, 0.0)
        self.s.placements(1.0, where)
        self.s.fx.clear()

    def flashes(self):
        return sum(p.img == "kenney/star_09" for p in self.s.fx.particles)

    def test_hit_only_when_damage_lands(self):
        self.s.on_event(ev("move", "player", move="Ice Beam"), 2.0)
        self.s.on_event(ev("miss", "player"), 3.2)
        self.assertEqual(self.flashes(), 0)                  # missed: no hit
        self.s.on_event(ev("move", "player", move="Ice Beam"), 4.0)
        self.s.on_event(ev("damage", "cpu", slot=1), 5.2)
        self.assertGreater(len(self.s.fx.particles), 10)

    def test_super_effective_shakes(self):
        self.assertEqual(self.s.fx.shake_offset(2.0), (0, 0))
        self.s.on_event(ev("effective", "player", effectiveness=2), 2.0)
        self.assertNotEqual(self.s.fx.shake_offset(2.05), (0, 0))
        self.assertEqual(self.s.fx.shake_offset(3.0), (0, 0))

    def test_send_out_and_faint_effects(self):
        self.s.on_event(ev("faint", "cpu", slot=1), 2.0)
        self.assertTrue(self.s.fx.particles)


class Sprites(unittest.TestCase):
    def test_all_front_and_back_sprites_load(self):
        from game.data import POKEDEX
        if get_sprite("riolu", "front") is None:
            self.skipTest("sprites not downloaded (git-ignored)")
        for key in POKEDEX:
            for view in ("front", "back"):
                s = get_sprite(key, view)
                self.assertIsNotNone(s, (key, view))
                self.assertGreater(len(s.frames), 10)
                self.assertEqual(s.frames[0].shape[2], 4)    # has alpha

    def test_billboard_stands_on_the_card_facing_the_camera(self):
        pose = synthetic_camera()
        q = billboard_quad(Placement(254, 300), 60, 80, pose)
        self.assertAlmostEqual(float(q[2][2]), 0.0)          # feet on the board
        self.assertAlmostEqual(float(q[0][2]), -80.0)        # 80 mm up (−z)
        cam = camera_position(pose)
        right = q[1] - q[0]
        to_cam = cam - np.array([254, 300, 0])
        self.assertAlmostEqual(float(right[:2] @ to_cam[:2]), 0.0, places=3)
