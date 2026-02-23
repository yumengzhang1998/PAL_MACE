import os
import json
import random
import shutil



# ============================================================
# 1. Generator: initialize pool (ONCE)
# ============================================================

def init_pool(init_path, results_dir, POOL_NAME = "starting_point_pool.csv"):
    """
    Copy init_path -> results_dir/starting_point_pool.csv
    """
    pool_path = os.path.join(results_dir, POOL_NAME)
    if not os.path.exists(pool_path):
        shutil.copy(init_path, pool_path)
    else:
        print(f"Pool already exists at {pool_path}, skipping initialization.")
    return pool_path


# ============================================================
# Helpers
# ============================================================

def _count_rows(csv_path):
    # number of data rows (excluding header)
    with open(csv_path, "r") as f:
        return sum(1 for _ in f) - 1


def _load_json(path, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def _save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ============================================================
# 2. Generator: random pop indices (NO pool modification)
# ============================================================

def random_pop_indices(results_dir, rank, k, seed=None, POOL_NAME = "starting_point_pool.csv"):
    """
    Randomly pick k UNUSED indices from the pool CSV.
    Each rank keeps its own used.json.
    """
    if seed is not None:
        random.seed(seed)

    pool_path = os.path.join(results_dir, POOL_NAME)
    gen_dir = os.path.join(results_dir, f"generator_{rank}")
    os.makedirs(gen_dir, exist_ok=True)

    used_path = os.path.join(gen_dir, "used.json")

    # load used indices
    used = set(_load_json(used_path, []))

    # current pool size (model may have appended)
    n_rows = _count_rows(pool_path)
    all_indices = set(range(n_rows))

    available = list(all_indices - used)
    if not available:
        return []

    chosen = random.sample(available, min(k, len(available)))

    used.update(chosen)
    _save_json(used_path, sorted(used))

    return chosen


# ============================================================
# 3. Generator: retrieve ALL visited rows (indices)
# ============================================================

def get_all_used_indices(results_dir, rank):
    gen_dir = os.path.join(results_dir, f"generator_{rank}")
    used_path = os.path.join(gen_dir, "used.json")
    return _load_json(used_path, [])

# ============================================================
# 4. DUmmy checkpoint handler so that MACE do not save checkpoints
# ============================================================

class DummyCheckpointHandler:
    def __init__(self, *args, **kwargs):
        pass

    def save(self, *args, **kwargs):
        pass

    def load_latest(self, *args, **kwargs):
        return None

    def load_best(self, *args, **kwargs):
        return None
