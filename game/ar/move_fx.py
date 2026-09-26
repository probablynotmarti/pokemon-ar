"""What each move looks like. Every move has a launch (plays when the move
is announced: projectiles, beams, auras…) and an impact (plays only when
the damage event arrives — so a miss shows no hit).

Positions are board mm: A = attacker's card, B = target's card. The
damage event lands ~1.2 s after the move event (the text beat), so
launches that travel arrive at ~1.1 s.
"""

import math

from game.data import MOVES

CHEST = 50.0          # mm above the card where attacks come from / land
ARRIVE = 1.45         # s after the move event: projectiles reach the target
                      # (the damage event follows at 1.6 s — see hand_ui)


def _path(A, B, h0=CHEST, h1=CHEST, arc=0.0):
    (ax, ay), (bx, by) = A, B
    return lambda p: (ax + (bx - ax) * p, ay + (by - ay) * p,
                      h0 + (h1 - h0) * p + arc * math.sin(math.pi * p))


def _at(P, h=CHEST, dx=0.0, dy=0.0, dh=0.0):
    return lambda p: (P[0] + dx, P[1] + dy, h + dh)


# ── building blocks ──────────────────────────────────────────────────
def projectile(fx, img, A, B, now, size=30, arc=30, delay=0.3, spin=0,
               glow=False, tint=None, n=1, gap=0.12):
    dur = ARRIVE - delay
    for i in range(n):
        jitter = (i - (n - 1) / 2) * 10
        fx.add(img=img, t0=now + delay + i * gap, dur=dur,
               path=_path((A[0] + jitter, A[1]), (B[0] + jitter, B[1]),
                          arc=arc + 8 * i),
               size=size, spin=spin, glow=glow, tint=tint)


def stream(fx, img, A, B, now, n=12, size=22, delay=0.3, glow=False,
           tint=None, spin=0, over=0.55):
    dur = ARRIVE - delay - over
    for i in range(n):
        j = fx.rng.uniform(-5, 5)
        fx.add(img=img, t0=now + delay + over * i / n, dur=dur,
               path=_path((A[0] + j, A[1]), (B[0] - j, B[1]), arc=6),
               size=size * fx.rng.uniform(0.8, 1.2), glow=glow, tint=tint,
               spin=spin, angle=fx.rng.uniform(0, 360))


def drop(fx, imgs, B, now, n=6, size=28, delay=0.35, spread=28, spin=180):
    for i in range(n):
        dx, dy = fx.rng.uniform(-spread, spread), fx.rng.uniform(-spread, spread)
        t0 = now + delay + i * 0.09
        dur = ARRIVE + 0.05 - delay - i * 0.09 + 0.25
        fx.add(img=imgs[i % len(imgs)], t0=t0, dur=max(0.25, dur),
               path=(lambda dx, dy: lambda p: (B[0] + dx, B[1] + dy,
                                               160 * (1 - p) ** 1.6 + 8))(dx, dy),
               size=size * fx.rng.uniform(0.8, 1.2), spin=spin,
               alpha=lambda p: min(1.0, p * 6))


def rings(fx, img, A, B, now, n=3, size0=20, size1=70, delay=0.25,
          glow=False, tint=None):
    for i in range(n):
        fx.add(img=img, t0=now + delay + i * 0.18, dur=ARRIVE - delay - 0.2,
               path=_path(A, B), size=(lambda p: size0 + (size1 - size0) * p),
               glow=glow, tint=tint, alpha=lambda p: min(1, p * 5) * (1 - p * 0.7))


def aura(fx, imgs, P, now, n=10, size=16, rise=70, radius=28, delay=0.1,
         glow=True, tint=None, dur=0.9):
    for i in range(n):
        ang = 2 * math.pi * i / n
        r = radius * fx.rng.uniform(0.7, 1.1)
        dx, dy = r * math.cos(ang), r * 0.7 * math.sin(ang)
        fx.add(img=imgs[i % len(imgs)], t0=now + delay + i * 0.05, dur=dur,
               path=(lambda dx, dy: lambda p: (P[0] + dx, P[1] + dy,
                                               10 + rise * p))(dx, dy),
               size=size, glow=glow, tint=tint,
               alpha=lambda p: math.sin(math.pi * p))


def burst(fx, img, P, now, size0=20, size1=90, dur=0.55, h=CHEST, glow=False,
          tint=None, delay=0.0, angle=0.0):
    fx.add(img=img, t0=now + delay, dur=dur, path=_at(P, h),
           size=lambda p: size0 + (size1 - size0) * (1 - (1 - p) ** 2),
           alpha=lambda p: 1 - p ** 2, glow=glow, tint=tint, angle=angle)


def contact(fx, img, P, now, size=45, dur=0.45, angle=0.0, dx=0.0, delay=0.0):
    fx.add(img=img, t0=now + delay, dur=dur, path=_at(P, CHEST, dx=dx),
           size=lambda p: size * (0.6 + 0.6 * math.sin(math.pi * min(1, p * 1.4))),
           alpha=lambda p: 1 - p ** 3, angle=angle)


def bite(fx, P, now, size=60):
    for img, h0, h1 in (("topbite", CHEST + 35, CHEST + 8),
                        ("bottombite", CHEST - 35, CHEST - 8)):
        fx.add(img=img, t0=now, dur=0.32,
               path=(lambda h0, h1: lambda p: (P[0], P[1], h0 + (h1 - h0) * min(1, p * 1.6)))(h0, h1),
               size=size, alpha=lambda p: 1 - max(0.0, p - 0.7) / 0.3)


def debris(fx, imgs, P, now, n=6, size=16, h=12):
    for i in range(n):
        ang = 2 * math.pi * i / n + fx.rng.uniform(-0.3, 0.3)
        v = fx.rng.uniform(35, 60)
        fx.add(img=imgs[i % len(imgs)], t0=now, dur=0.5,
               path=(lambda a, v: lambda p: (P[0] + v * p * math.cos(a),
                                             P[1] + v * p * math.sin(a) * 0.7,
                                             h + 40 * math.sin(math.pi * p)))(ang, v),
               size=size, spin=360, alpha=lambda p: 1 - p ** 2)


def hit(fx, P, now, big=False):
    """Generic contact flash: white starburst over a dark impact splat."""
    burst(fx, "impact", P, now, 20, 70 if big else 50, dur=0.45)
    burst(fx, "kenney/star_09", P, now, 30, 150 if big else 100, dur=0.4,
          glow=True)
    if big:
        burst(fx, "kenney/scorch_02", P, now, 30, 120, dur=0.35, glow=True,
              tint=(170, 220, 255))


ROCKS = ["rock1", "rock2", "rock3"]
SMOKE = ["kenney/smoke_01", "kenney/smoke_04", "kenney/smoke_07"]
STARS = ["kenney/star_04", "kenney/star_06", "shine"]


# ── the 40 moves (+ Struggle) ────────────────────────────────────────
def _fx(launch=None, impact=None):
    return (launch or (lambda fx, A, B, now: None),
            impact or (lambda fx, B, now: hit(fx, B, now)))


MOVE_FX = {
    # Registeel
    "Heavy Slam": _fx(lambda fx, A, B, now: aura(fx, ["kenney/dirt_01"], A, now, n=5, rise=20, size=30, glow=False, tint=(120, 140, 160)),
                      lambda fx, B, now: (hit(fx, B, now, big=True), debris(fx, ROCKS, B, now, n=7), fx.shake(now, 14))),
    "Flash Cannon": _fx(lambda fx, A, B, now: (burst(fx, "shine", A, now, 10, 60, dur=0.4, glow=True),
                                               stream(fx, "wisp", A, B, now, n=14, size=26, glow=True, tint=(255, 250, 230), delay=0.45, over=0.4)),
                        lambda fx, B, now: (burst(fx, "shine", B, now, 20, 90, glow=True), hit(fx, B, now))),
    "Hammer Arm": _fx(None, lambda fx, B, now: (fx.add(img="fist", t0=now - 0.25, dur=0.3, path=lambda p: (B[0], B[1], CHEST + 90 * (1 - p)), size=55),
                                                hit(fx, B, now, big=True), fx.shake(now, 8))),
    "Iron Defense": _fx(lambda fx, A, B, now: (aura(fx, ["gear", "greenmetal1", "greenmetal2"], A, now, n=9, size=20, glow=False),
                                               burst(fx, "shine", A, now, 10, 70, dur=0.5, glow=True, delay=0.6))),
    # Sudowoodo
    "Rock Slide": _fx(lambda fx, A, B, now: drop(fx, ROCKS, B, now, n=7, size=30),
                      lambda fx, B, now: (debris(fx, SMOKE, B, now, n=5, size=40, h=8), fx.shake(now, 8))),
    "Double-Edge": _fx(lambda fx, A, B, now: aura(fx, ["wisp"], A, now, n=6, size=26, rise=30, dur=0.5),
                       lambda fx, B, now: (hit(fx, B, now, big=True), fx.shake(now, 12))),
    "Brick Break": _fx(None, lambda fx, B, now: (contact(fx, "rightchop", B, now, 55, angle=-20), hit(fx, B, now))),
    "Sucker Punch": _fx(lambda fx, A, B, now: stream(fx, "blackwisp", A, B, now, n=6, size=30, delay=0.2, over=0.25),
                        lambda fx, B, now: (contact(fx, "fist", B, now, 45), hit(fx, B, now))),
    # Ducklett
    "Bubble Beam": _fx(lambda fx, A, B, now: stream(fx, "waterwisp", A, B, now, n=14, size=16, glow=True),
                       lambda fx, B, now: (burst(fx, "waterwisp", B, now, 20, 80, glow=True), hit(fx, B, now))),
    "Air Slash": _fx(lambda fx, A, B, now: (projectile(fx, "leftslash", A, B, now, size=40, arc=15, n=2),
                                            projectile(fx, "rightslash", A, B, now, size=40, arc=25, n=1, delay=0.36)),
                     lambda fx, B, now: (contact(fx, "leftslash", B, now, 60), contact(fx, "rightslash", B, now, 60, delay=0.08), hit(fx, B, now))),
    "Feather Dance": _fx(lambda fx, A, B, now: drop(fx, ["feather"], B, now, n=8, size=24, spin=120, spread=35)),
    "Roost": _fx(lambda fx, A, B, now: (drop(fx, ["feather"], A, now, n=5, size=20, spread=25),
                                        aura(fx, STARS, A, now, n=8, tint=(170, 255, 170), delay=0.5))),
    # Skwovet
    "Bite": _fx(None, lambda fx, B, now: (bite(fx, B, now), hit(fx, B, now))),
    "Body Slam": _fx(lambda fx, A, B, now: aura(fx, SMOKE, A, now, n=4, size=34, rise=10, glow=False, dur=0.6),
                     lambda fx, B, now: (hit(fx, B, now, big=True), debris(fx, SMOKE, B, now, n=5, size=34), fx.shake(now, 12))),
    "Super Fang": _fx(None, lambda fx, B, now: (bite(fx, B, now, size=85), hit(fx, B, now, big=True))),
    "Stuff Cheeks": _fx(lambda fx, A, B, now: (aura(fx, ["heart"], A, now, n=6, size=14, glow=False),
                                               aura(fx, STARS, A, now, n=8, tint=(160, 255, 160), delay=0.5))),
    # Politoed
    "Hydro Pump": _fx(lambda fx, A, B, now: stream(fx, "waterwisp", A, B, now, n=22, size=34, glow=True, over=0.5),
                      lambda fx, B, now: (burst(fx, "waterwisp", B, now, 30, 120, glow=True), hit(fx, B, now, big=True))),
    "Ice Beam": _fx(lambda fx, A, B, now: (stream(fx, "iceball", A, B, now, n=14, size=24, glow=True),
                                           stream(fx, "icicle", A, B, now, n=6, size=24, delay=0.35)),
                    lambda fx, B, now: (burst(fx, "iceball", B, now, 30, 110, glow=True), debris(fx, ["icicle"], B, now, n=6, size=20, h=CHEST))),
    "Hyper Voice": _fx(lambda fx, A, B, now: rings(fx, "kenney/circle_02", A, B, now, n=5, size0=25, size1=90,
                                                   glow=True, tint=(150, 240, 255)),
                       lambda fx, B, now: (burst(fx, "kenney/circle_03", B, now, 30, 140, glow=True,
                                                 tint=(150, 240, 255)), hit(fx, B, now))),
    "Hypnosis": _fx(lambda fx, A, B, now: (rings(fx, "purplewisp", A, B, now, n=4, size0=20, size1=70, glow=True),
                                           burst(fx, "stare", B, now, 30, 60, dur=0.6, delay=0.9))),
    # Marshadow
    "Spectral Thief": _fx(lambda fx, A, B, now: stream(fx, "blackwisp", B, A, now, n=10, size=30, delay=0.15, over=0.35),
                          lambda fx, B, now: (burst(fx, "shadowball", B, now, 20, 90, glow=False), contact(fx, "foot", B, now, 50), hit(fx, B, now))),
    "Close Combat": _fx(None, lambda fx, B, now: ([contact(fx, img, B, now, 45, dx=dx, delay=d) for img, dx, d in
                                                   (("fist", -12, 0.0), ("foot", 10, 0.08), ("fist1", -4, 0.16), ("foot", 14, 0.24), ("fist", 0, 0.32))],
                                                  hit(fx, B, now, big=True), fx.shake(now, 10))),
    "Shadow Sneak": _fx(lambda fx, A, B, now: stream(fx, "blackwisp", A, B, now, n=12, size=34, delay=0.15, over=0.3),
                        lambda fx, B, now: (burst(fx, "blackwisp", B, now, 20, 90), hit(fx, B, now))),
    "Bulk Up": _fx(lambda fx, A, B, now: (aura(fx, ["angry", "kenney/muzzle_01"], A, now, n=8, size=18, glow=False),
                                          burst(fx, "shine", A, now, 10, 70, dur=0.5, glow=True, delay=0.6))),
    # Gimmighoul
    "Shadow Ball": _fx(lambda fx, A, B, now: projectile(fx, "shadowball", A, B, now, size=42, arc=20, glow=False),
                       lambda fx, B, now: (burst(fx, "shadowball", B, now, 30, 110), hit(fx, B, now))),
    "Hex": _fx(lambda fx, A, B, now: (rings(fx, "purplewisp", A, B, now, n=3, glow=True),
                                      aura(fx, ["purplewisp"], B, now, n=8, delay=0.7, dur=0.6)),
               lambda fx, B, now: (burst(fx, "stare", B, now, 30, 70), hit(fx, B, now))),
    "Power Gem": _fx(lambda fx, A, B, now: (projectile(fx, "shine", A, B, now, size=26, arc=35, n=3, glow=True),
                                            projectile(fx, "rainbow", A, B, now, size=40, arc=20, delay=0.4)),
                     lambda fx, B, now: (burst(fx, "shine", B, now, 20, 100, glow=True), hit(fx, B, now))),
    "Nasty Plot": _fx(lambda fx, A, B, now: (aura(fx, ["purplewisp"], A, now, n=8, size=22),
                                             burst(fx, "stare", A, now, 20, 55, dur=0.6, delay=0.5, h=CHEST + 20))),
    # Riolu
    "Force Palm": _fx(None, lambda fx, B, now: (contact(fx, "leftchop", B, now, 55), burst(fx, "lightning", B, now, 20, 60, dur=0.3, delay=0.05), hit(fx, B, now))),
    "Quick Attack": _fx(lambda fx, A, B, now: stream(fx, "wisp", A, B, now, n=10, size=22, delay=0.15, over=0.25, glow=True),
                        lambda fx, B, now: hit(fx, B, now)),
    "Reversal": _fx(lambda fx, A, B, now: aura(fx, ["flareball"], A, now, n=6, size=24),
                    lambda fx, B, now: (contact(fx, "fist", B, now, 60), burst(fx, "flareball", B, now, 30, 110, glow=True), hit(fx, B, now, big=True))),
    "Crunch": _fx(None, lambda fx, B, now: (bite(fx, B, now, size=75), burst(fx, "blackwisp", B, now, 20, 70, delay=0.2), hit(fx, B, now))),
    # Seedot
    "Seed Bomb": _fx(lambda fx, A, B, now: projectile(fx, "energyball", A, B, now, size=22, arc=55, n=3, glow=True),
                     lambda fx, B, now: (debris(fx, ["leaf1", "leaf2"], B, now, n=8, size=18, h=CHEST), hit(fx, B, now))),
    "Synthesis": _fx(lambda fx, A, B, now: (aura(fx, ["leaf1", "leaf2"], A, now, n=6, size=16, glow=False),
                                            aura(fx, STARS, A, now, n=10, tint=(150, 255, 150), delay=0.4))),
    "Leech Seed": _fx(lambda fx, A, B, now: projectile(fx, "energyball", A, B, now, size=14, arc=60, n=3, glow=True),
                      lambda fx, B, now: debris(fx, ["leaf1", "leaf2"], B, now, n=6, size=14, h=10)),
    "Explosion": _fx(lambda fx, A, B, now: (burst(fx, "flareball", A, now, 20, 260, dur=0.8, glow=True, delay=0.2),
                                            burst(fx, "kenney/fire_01", A, now, 20, 200, dur=0.8, glow=True, delay=0.3),
                                            debris(fx, SMOKE, A, now + 0.4, n=8, size=60), fx.shake(now + 0.3, 22, 0.7)),
                     lambda fx, B, now: hit(fx, B, now, big=True)),
    # Solrock
    "Zen Headbutt": _fx(lambda fx, A, B, now: aura(fx, ["mistball"], A, now, n=6, size=28),
                        lambda fx, B, now: (burst(fx, "mistball", B, now, 30, 100, glow=True), hit(fx, B, now, big=True))),
    "Flare Blitz": _fx(lambda fx, A, B, now: (aura(fx, ["fireball", "kenney/flame_02"], A, now, n=10, size=24),
                                              stream(fx, "fireball", A, B, now, n=8, size=26, delay=0.5, over=0.3)),
                       lambda fx, B, now: (burst(fx, "flareball", B, now, 30, 130, glow=True),
                                           burst(fx, "kenney/fire_02", B, now, 30, 110, glow=True, delay=0.05),
                                           hit(fx, B, now, big=True), fx.shake(now, 12))),
    "Cosmic Power": _fx(lambda fx, A, B, now: (burst(fx, "moon", A, now, 10, 45, dur=0.8, h=CHEST + 45),
                                               aura(fx, STARS, A, now, n=12, delay=0.3))),
    "Struggle": _fx(None, lambda fx, B, now: hit(fx, B, now)),
}


def launch(fx, move, A, B, now):
    MOVE_FX.get(move, _fx())[0](fx, A, B, now)


def impact(fx, move, B, now, big=False):
    fn = MOVE_FX.get(move, _fx())[1]
    fn(fx, B, now)
    if big:
        fx.shake(now, 12)


# ── non-move moments ─────────────────────────────────────────────────
def status_fx(fx, status, P, now):
    if status == "par":
        for i in range(3):
            burst(fx, "lightning", P, now, 20, 50, dur=0.25, delay=i * 0.15,
                  angle=(-30, 20, -5)[i])
    elif status == "slp":
        rings(fx, "purplewisp", P, P, now, n=3, size0=20, size1=60, delay=0.0)
    elif status == "frz":
        burst(fx, "iceball", P, now, 30, 110, dur=0.6, glow=True)
        debris(fx, ["icicle"], P, now, n=6, size=18, h=CHEST)
    elif status == "brn":
        aura(fx, ["fireball", "kenney/flame_02"], P, now, n=8, size=22, rise=50)
    elif status == "seeded":
        debris(fx, ["leaf1", "leaf2"], P, now, n=6, size=14, h=10)


def drain_fx(fx, frm, to, now):
    projectile(fx, "energyball", frm, to, now, size=14, arc=40, n=3, delay=0.0,
               glow=True)


def heal_fx(fx, P, now):
    aura(fx, STARS, P, now, n=10, tint=(150, 255, 150))


def stat_fx(fx, P, now, up):
    if up:
        aura(fx, ["kenney/spark_05", "shine"], P, now, n=8, size=14,
             tint=(120, 200, 255))
    else:
        for i in range(6):
            dx = (i - 2.5) * 10
            fx.add(img="wisp", t0=now + i * 0.05, dur=0.6,
                   path=(lambda dx: lambda p: (P[0] + dx, P[1], 90 - 70 * p))(dx),
                   size=16, glow=True, tint=(255, 150, 90))


def faint_fx(fx, P, now):
    debris(fx, SMOKE, P, now, n=6, size=40, h=10)


def send_out_fx(fx, P, now):
    burst(fx, "pokeball", P, now, 10, 30, dur=0.35, h=10)
    burst(fx, "shine", P, now, 10, 90, dur=0.5, glow=True, delay=0.15, h=CHEST)
    aura(fx, STARS, P, now, n=8, delay=0.2, dur=0.6)
