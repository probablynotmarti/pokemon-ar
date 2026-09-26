"""Low-poly figurines for the 10 cards, built from primitives.

Each fits on a card (63 x 88 mm), faces +y, feet at z = 0, and is about
50–75 mm tall. Stylized, not game-accurate: a real model dropped in
assets/models/<key>.glb replaces its figurine (see loader.py).
Colours are BGR.
"""

import numpy as np

from .mesh import box, cone, cylinder, ellipsoid, merge, mirror_x, sphere

BLACK, WHITE = (35, 30, 30), (245, 245, 245)
RED_EYE = (50, 40, 215)


def registeel():
    steel, dark, dot = (185, 175, 170), (120, 112, 110), (50, 40, 200)
    parts = [ellipsoid((0, 0, 42), (22, 18, 27), steel, 2),
             ellipsoid((0, 11, 46), (13, 8, 15), dark)]
    for x, z in ((-6, 55), (-6, 47), (-6, 39), (6, 55), (6, 47), (6, 39),
                 (0, 47)):
        parts.append(sphere((x, 18, z), 2.4, dot))
    parts += mirror_x([cylinder((19, 0, 52), (29, 4, 26), 6, 7, steel),
                       sphere((30, 5, 21), 8, dark),
                       cylinder((10, 0, 18), (12, 0, 3), 7, 7.5, steel),
                       ellipsoid((12, 3, 3), (8, 10, 3.5), dark)])
    return merge(parts)


def sudowoodo():
    bark, leaf, spot = (40, 80, 125), (70, 165, 80), (70, 205, 225)
    parts = [cylinder((0, 0, 10), (0, 0, 56), 11, 9, bark, 12),
             sphere((0, 0, 62), 10, bark),
             cylinder((0, 0, 70), (0, 0, 76), 2, 1.5, bark, 6),
             sphere((0, 0, 79), 4.5, spot),
             cylinder((-8, 0, 44), (-22, 2, 60), 3, 2.5, bark, 6),
             sphere((-23, 3, 63), 7, leaf),
             cylinder((8, 0, 40), (24, 3, 46), 3, 2.5, bark, 6),
             sphere((26, 4, 47), 7, leaf)]
    for x, z in ((-3, 30), (4, 22), (-4, 16)):
        parts.append(ellipsoid((x, 10, z), (3, 1.5, 3.5), spot))
    parts += mirror_x([sphere((4, 9, 64), 1.8, BLACK),
                       cylinder((5, 0, 0), (5, 0, 11), 4.5, 5, bark, 8)])
    return merge(parts)


def ducklett():
    blue, pale, beak = (230, 175, 95), (250, 215, 160), (40, 200, 245)
    parts = [ellipsoid((0, 0, 21), (17, 15, 18), blue, 2),
             ellipsoid((0, 11, 21), (9, 5, 9), pale),
             sphere((0, 3, 44), 11, blue, 2),
             ellipsoid((0, 14, 42), (7, 7, 3), beak),
             cone((0, 0, 53), (0, -2, 63), 3.5, blue, 8)]
    parts += mirror_x([sphere((5, 11, 47), 1.8, BLACK),
                       ellipsoid((15, -2, 24), (4, 9, 10), blue),
                       ellipsoid((6, 4, 1.5), (5, 7, 1.8), beak)])
    return merge(parts)


def skwovet():
    fur, belly, tail = (120, 130, 150), (200, 225, 240), (95, 105, 120)
    parts = [ellipsoid((0, 0, 19), (15, 13, 19), fur, 2),
             ellipsoid((0, 9, 17), (10, 5, 13), belly),
             ellipsoid((0, 2, 39), (12, 11, 10), fur, 2),
             ellipsoid((0, -17, 33), (12, 9, 22), tail, 2),
             sphere((0, 12, 25), 4.5, (170, 140, 250)),        # berry
             sphere((0, 12.5, 39), 1.6, (60, 140, 235))]       # nose
    parts += mirror_x([ellipsoid((8, 7, 36), (5, 5, 5), belly),
                       cone((7, 0, 46), (8, 0, 52), 3, fur, 8),
                       sphere((4.5, 10.5, 42), 1.6, BLACK),
                       ellipsoid((6, 4, 2), (4, 6, 2), fur)])
    return merge(parts)


def politoed():
    green, yellow, swirl = (95, 190, 110), (70, 215, 240), (50, 120, 65)
    parts = [ellipsoid((0, 0, 26), (20, 17, 24), green, 2),
             ellipsoid((0, 11, 23), (13, 7, 15), yellow),
             cylinder((0, -1, 48), (0, -3, 54), 2, 2, green, 6)]
    for i in range(9):                                  # belly swirl
        a = i * 0.9
        r = 1.5 + i * 0.9
        parts.append(sphere((r * np.cos(a), 17.5, 23 + r * np.sin(a)), 1.6,
                            swirl))
    for i in range(10):                                 # head curl
        a = np.pi / 2 + i * 0.62
        r = 7 - i * 0.45
        parts.append(sphere((0, -3 + r * np.cos(a), 60 + r * np.sin(a)), 2.2,
                            green))
    parts += mirror_x([sphere((7, 11, 41), 5, WHITE),
                       sphere((7, 15, 41.5), 2.4, BLACK),
                       ellipsoid((12, 12, 34), (3, 1.5, 2.5), (170, 150, 250)),
                       sphere((20, 7, 26), 5, yellow),
                       ellipsoid((9, 5, 2), (7, 8, 2.5), yellow)])
    return merge(parts)


def marshadow():
    body, hair, tips = (55, 50, 55), (120, 190, 110), (110, 225, 215)
    parts = [ellipsoid((0, 0, 24), (11, 10, 15), body, 2),
             ellipsoid((0, 2, 44), (13, 12, 11), hair, 2),
             ellipsoid((0, 10, 42), (7, 3.5, 6), body),
             cone((0, -2, 52), (0, -5, 68), 4.5, tips, 8)]
    parts += mirror_x([cone((7, 0, 50), (15, -2, 62), 4, tips, 8),
                       cone((11, 2, 40), (20, 0, 46), 3, hair, 8),
                       sphere((3.5, 13, 43.5), 1.9, RED_EYE),
                       cylinder((9, 0, 32), (17, 5, 20), 3, 3, body, 8),
                       sphere((18, 6, 18), 4.5, body),
                       cylinder((5, 0, 11), (6, 1, 0), 3.5, 4, body, 8)])
    return merge(parts)


def gimmighoul():
    wood, gold, ghost = (40, 45, 150), (40, 185, 225), (180, 165, 165)
    parts = [box((0, 0, 11), (36, 24, 22), wood),
             box((0, 12.2, 11), (37, 0.8, 4), gold),
             box((0, 0, 21.5), (37, 25, 2.5), gold),
             box((0, -10, 32), (36, 4, 20), wood),            # open lid
             sphere((0, 3, 29), 8, ghost, 2),
             cylinder((0, 3, 36), (0, 3, 44), 1, 1, ghost, 6),
             sphere((0, 3, 45), 2.2, ghost),
             cylinder((-8, 6, 23), (-8, 6, 24.5), 3, 3, gold, 8),
             cylinder((9, 5, 23), (9, 5, 24.5), 3, 3, gold, 8)]
    parts += mirror_x([sphere((3, 10.5, 30), 1.8, (60, 225, 250))])
    return merge(parts)


def riolu():
    blue, black = (215, 140, 60), (40, 35, 40)
    parts = [ellipsoid((0, 0, 26), (11, 10, 14), blue, 2),
             ellipsoid((0, 7, 28), (7, 4, 8), (230, 200, 150)),
             sphere((0, 1, 45), 11, blue, 2),
             ellipsoid((0, 8, 46), (11, 4.5, 3.2), black),
             sphere((0, -10, 17), 4, blue)]
    parts += mirror_x([cone((7, 0, 52), (14, -2, 66), 3.5, black, 8),
                       sphere((4, 11.5, 46.5), 1.8, RED_EYE),
                       cylinder((10, 0, 34), (16, 6, 24), 2.5, 2.5, blue, 8),
                       sphere((17, 7, 22), 3.5, black),
                       cylinder((5, 0, 14), (6, 1, 0), 3.5, 3.5, black, 8)])
    return merge(parts)


def seedot():
    body, cap, foot = (45, 95, 155), (30, 50, 85), (150, 205, 230)
    parts = [ellipsoid((0, 0, 21), (15, 14, 20), body, 2),
             ellipsoid((0, 0, 38), (16, 15, 9), cap, 2),
             cone((0, 0, 45), (0, 0, 54), 4, cap, 8)]
    parts += mirror_x([ellipsoid((5, 13, 31), (3.2, 1.5, 1.6), BLACK),
                       ellipsoid((6, 4, 1.5), (4, 6, 1.8), foot)])
    return merge(parts)


def solrock():
    core, ray, eye = (40, 75, 215), (40, 200, 240), (70, 215, 250)
    z0 = 36                                   # it floats
    parts = [ellipsoid((0, 0, z0), (14, 10, 14), core, 2)]
    for i in range(10):
        a = i * 2 * np.pi / 10
        d = np.array([np.cos(a), 0, np.sin(a)])
        length = 30 if i % 2 == 0 else 22
        parts.append(cone(np.array([0, 0, z0]) + 11 * d,
                          np.array([0, 0, z0]) + length * d, 4.5, ray, 8))
    parts += mirror_x([ellipsoid((5, 9.5, z0 + 2), (3, 1.2, 4.5), eye),
                       ellipsoid((5, 10.6, z0 + 2), (1, 0.6, 3.5), BLACK)])
    return merge(parts)


BUILDERS = {f.__name__: f for f in (registeel, sudowoodo, ducklett, skwovet,
                                    politoed, marshadow, gimmighoul, riolu,
                                    seedot, solrock)}
FLOATS = {"solrock"}                          # bob high off the card
