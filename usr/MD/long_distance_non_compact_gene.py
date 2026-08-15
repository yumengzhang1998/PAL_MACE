#!/usr/bin/env python3

import numpy as np
from scipy.spatial.transform import Rotation as R
import os
from tqdm import tqdm


# ----------------------------
# Geometry helpers
# ----------------------------

def center(coords):
    return coords - np.mean(coords, axis=0)


def random_rotation(coords):
    return R.random().apply(coords)


def translate(coords, vec):
    return coords + vec


def min_pair_distance(A, B):
    A = np.atleast_2d(A)
    B = np.atleast_2d(B)

    diff = A[:, None, :] - B[None, :, :]
    dist = np.linalg.norm(diff, axis=2)

    return dist.min()


def no_clash(*mols, min_atomic_distance=3.0):
    for i in range(len(mols)):
        for j in range(i + 1, len(mols)):
            if min_pair_distance(mols[i], mols[j]) < min_atomic_distance:
                return False
    return True


# ----------------------------
# Placement (robust version)
# ----------------------------

def place_fragment(core, frag,
                   target_dist=5.0,
                   tol=0.2,
                   min_atomic_distance=3.0):

    frag_rot = random_rotation(frag)

    direction = np.random.randn(3)
    direction /= np.linalg.norm(direction)

    t = 0.0
    step = 0.3

    for _ in range(200):
        trial = translate(frag_rot, direction * t)
        d = min_pair_distance(core, trial)

        if d >= target_dist:
            break
        t += step

    # refine
    for _ in range(50):
        trial = translate(frag_rot, direction * t)
        d = min_pair_distance(core, trial)

        if abs(d - target_dist) < tol and no_clash(core, trial, min_atomic_distance):
            return trial

        t += (target_dist - d) * 0.5

    return None


# ----------------------------
# Generators
# ----------------------------

def generate_bi4_bi7(bi4, bi7, n):
    bi4 = center(bi4)
    bi7 = center(bi7)

    samples = []

    for _ in tqdm(range(n), desc="Bi4+Bi7"):
        while True:
            placed = place_fragment(bi7, bi4, target_dist=4.5)
            if placed is None:
                continue

            merged = np.vstack([placed, bi7])
            samples.append(merged)
            break

    return samples


def generate_bi2_bi7_bi2(bi2, bi7, n):
    bi2 = center(bi2)
    bi7 = center(bi7)

    samples = []

    for _ in tqdm(range(n), desc="Bi2+Bi7+Bi2"):
        while True:
            a = place_fragment(bi7, bi2, target_dist=4.5)
            if a is None:
                continue

            b = place_fragment(bi7, bi2, target_dist=4.5)
            if b is None:
                continue

            if not no_clash(a, b, min_atomic_distance=4.5):
                continue

            merged = np.vstack([a, bi7, b])
            samples.append(merged)
            break

    return samples


# ----------------------------
# IO
# ----------------------------

def read_xyz(filename):
    with open(filename) as f:
        lines = f.readlines()

    mols = []
    i = 0
    while i < len(lines):
        n = int(lines[i])
        i += 2
        coords = []
        for _ in range(n):
            parts = lines[i].split()
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
            i += 1
        mols.append(np.array(coords))

    return mols


def save_npz(filename, coords, labels):
    """
    coords: (N, n_atoms, 3)
    labels: 0 = Bi4-based, 1 = Bi2-based
    """
    np.savez_compressed(
        filename,
        coords=coords,
        labels=labels
    )
    print(f"Saved: {filename}")
    print("coords shape:", coords.shape)


# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":

    # -------- control --------
    n_bi4 = 100
    n_bi2 = 100

    prefix_bi2 = "Bi2-2"
    prefix_bi4 = "Bi4-2"
    prefix_bi7 = "Bi7-3"

    # -------- load --------
    bi2 = read_xyz(f"building_blocks/{prefix_bi2}_optimized.xyz")[0]
    bi4 = read_xyz(f"building_blocks/{prefix_bi4}_optimized.xyz")[0]
    bi7 = read_xyz(f"building_blocks/{prefix_bi7}_optimized.xyz")[0]
    bi2 = np.atleast_2d(bi2)
    bi4 = np.atleast_2d(bi4)
    bi7 = np.atleast_2d(bi7)
    print("bi4 shape:", bi4.shape)
    print("bi7 shape:", bi7.shape)
    # -------- generate --------
    samples_bi4 = generate_bi4_bi7(bi4, bi7, n_bi4)
    samples_bi2 = generate_bi2_bi7_bi2(bi2, bi7, n_bi2)

    # -------- merge --------
    all_samples = samples_bi4 + samples_bi2

    labels = np.array(
        [0] * len(samples_bi4) + [1] * len(samples_bi2)
    )

    coords = np.array(all_samples)

    print("Total dataset:", coords.shape)

    # -------- save --------
    save_npz("building_blocks/bi_mixed_dataset.npz", coords, labels)