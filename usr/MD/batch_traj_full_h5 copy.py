import time
import numpy as np
import pandas as pd
# from xtb.interface import Calculator
# from xtb.utils import get_method
import numpy as np
# from xtb.utils import get_solvent
import sys
import pickle
import openmm
import h5py
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../pretrain")
import os
current_path = os.getcwd()
from openmm import unit
from utils_multi_traj import Molecule, convert_to_data_object
import torch
import ast
import periodictable
import openmm as mm
from scipy.spatial.distance import pdist
from tqdm import tqdm
import argparse
from pretrain.evaluation import evaluate
import openmm.app as app
from pathlib import Path
import tempfile
import traceback
import gc
print("Available platforms:", [mm.Platform.getPlatform(i).getName()
                               for i in range(mm.Platform.getNumPlatforms())])
from sklearn.model_selection import StratifiedShuffleSplit
def stratified_sample(df, label_col, n_total, random_state=42):
    rng = np.random.default_rng(random_state)

    groups = df.groupby(label_col)
    n_classes = groups.ngroups

    # minimum 1 per class
    base = n_total // n_classes
    remainder = n_total % n_classes

    sampled = []

    for i, (label, g) in enumerate(groups):
        n = min(len(g), base + (i < remainder))
        sampled.append(g.sample(n=n, random_state=random_state))

    return pd.concat(sampled).sample(frac=1, random_state=random_state).reset_index(drop=True)


def vec3_to_numpy(vec3_list):
    # Convert list of Vec3 to a NumPy array
    return np.array([[v.x, v.y, v.z] for v in vec3_list])
import h5py
import numpy as np
import math


def build_mixed_topology(atom_numbers, n_traj):
    top = app.Topology()
    elem_cache = {}
    for t in range(n_traj):
        chain = top.addChain()
        res = top.addResidue(f"TRAJ{t}", chain)
        for z in atom_numbers:
            sym = periodictable.elements[z].symbol
            if sym not in elem_cache:
                elem_cache[sym] = app.Element.getBySymbol(sym)
            top.addAtom(sym, elem_cache[sym], res)
    return top

def build_mixed_system(n_atoms, n_traj, mass_amu=200.0):
    system = mm.System()
    for _ in range(n_traj * n_atoms):
        system.addParticle(mass_amu * unit.amu)
    return system

def make_custom_force(n_particles):
    force = mm.CustomExternalForce("fx*x + fy*y + fz*z")
    force.addPerParticleParameter("fx")
    force.addPerParticleParameter("fy")
    force.addPerParticleParameter("fz")
    for i in range(n_particles):
        force.addParticle(i, [0.0, 0.0, 0.0])
    return force

def write_frame_minimal(grp, step, frame):
    grp["coords"][step] = frame[0]
    grp["pred_forces"][step] = frame[5]
    grp["pred_energy"][step] = frame[6]

class Generate_TrajsBatch:
    def __init__(
        self,
        data_batch,
        result_path,
        model_number,
        prefix,
        temperature=298.0,
        save_stride=50,
        inner_steps=1,
        torchforce = False,
    ):
        self.data_batch = data_batch
        self.temperature = temperature * unit.kelvin
        self.collision_rate = 1.0 / unit.picosecond
        self.timestep = 2.0 * unit.femtoseconds
        self.save_stride = save_stride
        self.inner_steps = inner_steps
        self.r_max = 30.0
        if not torchforce:
            path = f"../../results/{prefix}_org/model_{model_number}.pt"
            self.model = torch.load(path, map_location="cuda")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device).eval()

        print(f"🔮 Loaded model {model_number} on {self.device}")

    @staticmethod
    def _max_pair_distance(coords_angstrom: np.ndarray) -> float:
        """Return max interatomic distance (Å) for coords shape (N,3)."""
        if coords_angstrom.shape[0] < 2:
            return 0.0
        return float(pdist(coords_angstrom).max())

    
    def set_up(self, data):
        atom_numbers = data[1]
        atom_types = [periodictable.elements[i].symbol for i in atom_numbers]
        coordinates = torch.tensor(data[0])
        coordinates -= coordinates.mean(axis=0)  # center at origin
        # reduced_density = 0.05

        # sigma = 3.4 * unit.angstroms
        # nparticles = len(atom_numbers)
        # number_density = reduced_density / sigma**3
        # volume = nparticles * (number_density ** -1)
        # box_edge = volume ** (1. / 3.)
        # box_vectors = np.diag([box_edge/unit.angstrom for i in range(3)]) * unit.angstroms
        molecule = Molecule(atom_types=atom_types, coordinates=coordinates)
        return molecule
    def custom_force_initilize(self, molecule):
        force = openmm.CustomExternalForce('fx * x + fy * y + fz * z')
        force.addPerParticleParameter('fx')
        force.addPerParticleParameter('fy')
        force.addPerParticleParameter('fz')
        for i in range(molecule.system.getNumParticles()):
            force.addParticle(i, [0.0, 0.0, 0.0])
        return force

        
    def update_forces(self,force, predicted_forces):
        """
        Update the forces acting on each atom using the predicted forces.

        :param context: The OpenMM Context object managing the simulation state.
        :param force: The CustomExternalForce object used to apply custom forces.
        :param predicted_forces: A (4, 3) array of predicted forces, one row per atom.
                                Each row contains [Fx, Fy, Fz] force components.
        """
        # Loop over each atom and set the predicted force components
        conversion_factor = 96.485 * 10  # kcal/mol/Å -> kJ/mol/nm
        current_forces = -1 * predicted_forces * conversion_factor # * unit.kilojoule_per_mole / unit.nanometer
        # current_forces = predicted_forces * unit.kilojoule_per_mole / unit.nanometer
        for i in range(predicted_forces.shape[0]):
            fx, fy, fz = current_forces[i]
            # fx = fx.value_in_unit(unit.kilojoule_per_mole / unit.angstrom)
            # fy = fy.value_in_unit(unit.kilojoule_per_mole / unit.angstrom)
            # fz = fz.value_in_unit(unit.kilojoule_per_mole / unit.angstrom)
            force.setParticleParameters(i, i, [fx, fy, fz])
        return force
    
    def update(self, geometry, simulation):
        simulation.step(1)
        # print(geometry[0])
        current_positions = simulation.context.getState(getPositions=True).getPositions()
        # print(current_positions)
        # coords_distorded = np.array(current_positions)
        coords_distorded = np.array([pos.value_in_unit(unit.angstrom) for pos in current_positions])
        # velosity = simulation.context.getState(getVelocities=True).getVelocities()
        # forces = simulation.context.getState(getForces=True).getForces()
        true_force_empty = np.zeros(coords_distorded.shape)
        traj = [
            coords_distorded, # pos
            geometry[1], #atom_numbers
            None, # true_energy
            true_force_empty, # true_forces
            geometry[4], # charge
            geometry[5], # pred_forces 
            geometry[-3],   # pred_energy
            geometry[-2], # patience
            None ]# velocity
        return traj

    def set_up_simulations(self, data_batch):
        simulators = []
        for idx, data in enumerate(tqdm(data_batch, desc="Setting up simulations")):
            try:
                molecule = self.set_up(data)
                init_force = self.custom_force_initilize(molecule)
                molecule.system.addForce(init_force)
                # PBC not needed, skip box vectors
                # platform = openmm.Platform.getPlatformByName('CPU')
                platform = openmm.Platform.getPlatformByName('CUDA')
                platform.setPropertyDefaultValue("CudaDeviceIndex", "0")
                platform.setPropertyDefaultValue("CudaPrecision", "mixed")
                properties = {
                    "CudaDeviceIndex": "0",   # same masked GPU as PyTorch
                    "Precision": "single",    # faster unless you truly need double
                }
                # integrator = openmm.VerletIntegrator(self.timestep)
                integrator = openmm.LangevinIntegrator(self.temperature, self.collision_rate, self.timestep)
                simulation = openmm.app.Simulation(molecule.get_Topology(), molecule.get_System(), integrator, platform, properties)
                simulation.context.setPositions(data[0] * 0.1)
                simulation.context.setVelocitiesToTemperature(self.temperature)

                simulators.append((simulation, init_force))
            except Exception as e:
                print(f"❌ Failed to set up simulation #{idx}: {e}")
                continue
        return simulators
    def init_h5(self, h5_path, steps, n_atoms, n_trajs, labels=None):

        n_frames = math.ceil(steps / self.save_stride)
        h5f = h5py.File(h5_path, "w")

        groups = []

        for i in range(n_trajs):

            g = h5f.create_group(f"traj_{i}")

            # ---- SAFELY STORE LABEL PER TRAJ ----
            if labels is not None:
                g.attrs["label"] = str(labels[i])
            else:
                g.attrs["label"] = "None"

            g.create_dataset(
                "coords",
                shape=(n_frames, n_atoms, 3),
                dtype="f4",
                chunks=(1, n_atoms, 3),
                compression="gzip",
            )

            g.create_dataset(
                "pred_forces",
                shape=(n_frames, n_atoms, 3),
                dtype="f4",
                chunks=(1, n_atoms, 3),
                compression="gzip",
            )

            g.create_dataset(
                "pred_energy",
                shape=(n_frames,),
                dtype="f8",
            )

            g.create_dataset(
                "md_step",
                shape=(n_frames,),
                dtype="i8",
            )

            groups.append(g)

        return h5f, groups


    def run(self, steps, h5_path, label_batch=None, final_path=None):

        simulators = []
        frames = []

        # ----------------- SETUP (unchanged physics) -----------------
        for data in self.data_batch:
            mol = self.set_up(data)
            force = self.custom_force_initilize(mol)
            mol.system.addForce(force)

            platform = openmm.Platform.getPlatformByName("CUDA")
            integrator = openmm.LangevinIntegrator(
                self.temperature, self.collision_rate, self.timestep
            )

            sim = openmm.app.Simulation(
                mol.get_Topology(), mol.get_System(), integrator, platform
            )
            sim.context.setPositions(data[0] * 0.1)
            sim.context.setVelocitiesToTemperature(self.temperature)

            simulators.append((sim, force))
            frames.append(data)

        n_trajs = len(simulators)
        n_atoms = frames[0][0].shape[0]

        # ----------------- SAFE HDF5 INITIALIZATION -----------------
        tmp_path = h5_path + ".tmp"
        print(f"💾 Writing to temporary file: {tmp_path}")

        h5f, groups = self.init_h5(
            tmp_path, steps, n_atoms, n_trajs, labels=label_batch
        )

        active = [True] * n_trajs
        write_step = 0

        try:
            for step in tqdm(range(steps), desc="MD"):

                if not any(active):
                    print("🛑 All trajectories terminated early.")
                    break

                energies, forces = self.get_predicted_energy_and_forces(frames)

                for i, (sim, force) in enumerate(simulators):

                    if not active[i]:
                        continue

                    # apply ML forces
                    self.update_forces(force, forces[i])
                    force.updateParametersInContext(sim.context)
                    sim.step(self.inner_steps)

                    # get positions
                    state = sim.context.getState(getPositions=True)
                    coords = np.array(
                        [p.value_in_unit(unit.angstrom) for p in state.getPositions()]
                    )

                    # update in-memory frames
                    frames[i][0] = coords
                    frames[i][5] = forces[i]
                    frames[i][6] = energies[i]

                    # ---- SAVE ONLY EVERY save_stride ----
                    if step % self.save_stride == 0:
                        g = groups[i]
                        g["coords"][write_step] = coords
                        g["pred_forces"][write_step] = forces[i]
                        g["pred_energy"][write_step] = energies[i]
                        g["md_step"][write_step] = step

                # flush once per saved frame (not per traj)
                if step % self.save_stride == 0:
                    h5f.flush()
                    write_step += 1

                # explosion check
                for i in range(n_trajs):
                    if active[i] and self._max_pair_distance(frames[i][0]) > self.r_max:
                        print(f"💥 Traj {i} exploded at step {step}")
                        active[i] = False

        except KeyboardInterrupt:
            print("🛑 Caught Ctrl+C — saving partial file safely.")

        finally:
            try:
                h5f.flush()
                h5f.close()
            except Exception:
                pass

            # ---- ATOMIC RENAME: never leave corrupted files ----
            if final_path is not None:
                os.replace(tmp_path, final_path)
                print(f"✅ Final file written safely: {final_path}")
            else:
                os.replace(tmp_path, h5_path)
                print(f"✅ Final file written safely: {h5_path}")

    def run_mixed(self, steps, h5_path, label_batch=None, final_path=None):

        # ---------- setup ----------
        frames = self.data_batch
        n_trajs = len(frames)
        n_atoms = frames[0][0].shape[0]

        # Build ONE mixed molecule/system
        atom_numbers = frames[0][1]
        atom_types = [periodictable.elements[i].symbol for i in atom_numbers]

        # Build topology manually (one chain per trajectory)
        topology = openmm.app.Topology()
        element_cache = {}

        for t in range(n_trajs):
            chain = topology.addChain()
            res = topology.addResidue(f"TRAJ{t}", chain)

            for z in atom_numbers:
                sym = periodictable.elements[z].symbol
                if sym not in element_cache:
                    element_cache[sym] = openmm.app.Element.getBySymbol(sym)
                topology.addAtom(sym, element_cache[sym], res)

        # Build system
        system = openmm.System()
        for _ in range(n_trajs * n_atoms):
            system.addParticle(200.0 * unit.amu)

        # One CustomExternalForce for all particles
        force = openmm.CustomExternalForce("fx*x + fy*y + fz*z")
        force.addPerParticleParameter("fx")
        force.addPerParticleParameter("fy")
        force.addPerParticleParameter("fz")

        for i in range(n_trajs * n_atoms):
            force.addParticle(i, [0.0, 0.0, 0.0])

        system.addForce(force)

        # Integrator + Simulation
        integrator = openmm.LangevinIntegrator(
            self.temperature,
            self.collision_rate,
            self.timestep
        )

        platform = openmm.Platform.getPlatformByName("CUDA")
        properties = {"CudaDeviceIndex": "0", "Precision": "single"}

        sim = openmm.app.Simulation(
            topology, system, integrator, platform, properties
        )

        # Initial positions (stack all trajs)
        pos0 = np.concatenate([f[0] for f in frames], axis=0) * 0.1  # Å → nm
        sim.context.setPositions(pos0)
        sim.context.setVelocitiesToTemperature(self.temperature)

        # HDF5
        h5f, groups = self.init_h5(
            h5_path, steps, n_atoms, n_trajs, labels=label_batch
        )

        active = [True] * n_trajs
        write_step = 0
        conversion_factor = 96.485 * 10  # kcal/mol/Å → kJ/mol/nm

        # ---------- MD loop ----------
        for step in tqdm(range(steps), desc="MD (mixed)"):

            if not any(active):
                print("🛑 All trajectories terminated early.")
                break

            # 1) Predict for all trajs
            energies, forces_pred = self.get_predicted_energy_and_forces(frames)

            # forces_pred shape: (n_trajs, n_atoms, 3)

            # 2) Update ONE force object
            flat_forces = forces_pred.reshape(-1, 3)
            current_forces = -1.0 * flat_forces * conversion_factor

            for i in range(n_trajs * n_atoms):
                fx, fy, fz = current_forces[i]
                force.setParticleParameters(i, i, [fx, fy, fz])

            force.updateParametersInContext(sim.context)

            # 3) Integrate
            sim.step(self.inner_steps)

            # 4) Sync when needed
            if step % self.save_stride == 0 or any(active):

                state = sim.context.getState(getPositions=True)
                coords_nm = np.array([
                    p.value_in_unit(unit.nanometer)
                    for p in state.getPositions()
                ])

                coords = coords_nm.reshape(n_trajs, n_atoms, 3) / 0.1  # → Å

                for i in range(n_trajs):

                    if not active[i]:
                        continue

                    frames[i][0] = coords[i]
                    frames[i][5] = forces_pred[i]
                    frames[i][6] = energies[i]

                    # Save
                    if step % self.save_stride == 0:
                        g = groups[i]
                        g["coords"][write_step] = coords[i]
                        g["pred_forces"][write_step] = forces_pred[i]
                        g["pred_energy"][write_step] = energies[i]
                        g["md_step"][write_step] = step

                    # Explosion check
                    if self._max_pair_distance(coords[i]) > self.r_max:
                        print(f"💥 Traj {i} exploded at step {step}")
                        active[i] = False

            if step % self.save_stride == 0:
                write_step += 1

            if step % (1000 * self.save_stride) == 0:
                h5f.flush()

        h5f.close()
        print("✅ HDF5 trajectory written successfully (mixed)")
    def run_mixed_skip_bad(self, steps, h5_path, label_batch=None, final_path=None):

        frames = self.data_batch
        n_trajs = len(frames)
        n_atoms = frames[0][0].shape[0]
        atom_numbers = frames[0][1]

        # ---------------- Topology ----------------
        topology = openmm.app.Topology()
        element_cache = {}
        for t in range(n_trajs):
            chain = topology.addChain()
            res = topology.addResidue(f"TRAJ{t}", chain)
            for z in atom_numbers:
                sym = periodictable.elements[z].symbol
                if sym not in element_cache:
                    element_cache[sym] = openmm.app.Element.getBySymbol(sym)
                topology.addAtom(sym, element_cache[sym], res)

        # ---------------- System ----------------
        system = openmm.System()
        for _ in range(n_trajs * n_atoms):
            system.addParticle(200.0 * unit.amu)

        ml_force = openmm.CustomExternalForce("fx*x + fy*y + fz*z")
        ml_force.addPerParticleParameter("fx")
        ml_force.addPerParticleParameter("fy")
        ml_force.addPerParticleParameter("fz")
        for p in range(n_trajs * n_atoms):
            ml_force.addParticle(p, [0.0, 0.0, 0.0])
        system.addForce(ml_force)

        integrator = openmm.LangevinIntegrator(self.temperature, self.collision_rate, self.timestep)
        platform = openmm.Platform.getPlatformByName("CUDA")
        properties = {"CudaDeviceIndex": "0", "Precision": "single"}
        sim = openmm.app.Simulation(topology, system, integrator, platform, properties)

        pos0_nm = np.concatenate([f[0] for f in frames], axis=0) * 0.1
        sim.context.setPositions(pos0_nm)
        sim.context.setVelocitiesToTemperature(self.temperature)

        # ---------------- HDF5 ----------------
        base_path = final_path if final_path else h5_path
        tmp_path = h5_path + ".tmp" if final_path else h5_path
        h5f, groups = self.init_h5(tmp_path, steps, n_atoms, n_trajs, labels=label_batch)

        # ---------------- CSV log ----------------
        log_path = base_path.replace(".h5", "_explosions.csv")
        log_file = open(log_path, "w")
        log_file.write("traj_id,label,md_step,max_dist\n")
        log_file.flush()

        active = [True] * n_trajs
        last_good_coords = [frames[i][0].copy() for i in range(n_trajs)]  # Å

        forces_full = np.zeros((n_trajs, n_atoms, 3), dtype=np.float64)
        energies_full = np.zeros((n_trajs,), dtype=np.float64)

        conversion_factor = 96.485 * 10
        write_step = 0
        timing_file = open("timing_log.csv", "w")
        timing_file.write("step,t_frame,t_force_calc,t_explode_force_alloc,t_force_apply,t_run_step,t_update_frames,t_explosion_check,t_save, alloc, reserved\n")

        try:
            for step in tqdm(range(steps), desc="MD (mixed_skip_bad)"):
                t0 = time.time()
                t_save = None

                if not any(active):
                    print("🛑 All trajectories marked bad.")
                    break

                # -------- 1) ML inference ONLY for active trajs --------
                active_ids = [i for i, a in enumerate(active) if a]
                active_frames = [frames[i] for i in active_ids]
                t_frame = time.time()

                if active_frames:
                    e_act, f_act = self.get_predicted_energy_and_forces(active_frames)
                    for k, i in enumerate(active_ids):
                        energies_full[i] = e_act[k]
                        forces_full[i] = f_act[k]
                t_force_calc = time.time()
                # inactive -> zero forces so they don't slow ML / don't affect dynamics much
                for i in range(n_trajs):
                    if not active[i]:
                        energies_full[i] = 0.0
                        forces_full[i].fill(0.0)
                t_explode_force_alloc = time.time()
                # -------- 2) Apply ML forces --------
                flat_forces = forces_full.reshape(-1, 3)
                current_forces = -1.0 * flat_forces * conversion_factor

                for p in range(n_trajs * n_atoms):
                    fx, fy, fz = current_forces[p]
                    ml_force.setParticleParameters(p, p, [fx, fy, fz])
                ml_force.updateParametersInContext(sim.context)
                t_force_apply = time.time()

                # -------- 3) Integrate --------
                sim.step(self.inner_steps)
                t_run_step = time.time()

                # -------- 4) IMPORTANT: sync coords EVERY STEP (like your old method effectively did) --------
                state = sim.context.getState(getPositions=True)
                coords_nm = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
                coords_ang = coords_nm.reshape(n_trajs, n_atoms, 3) / 0.1

                # update frames for next ML call (only active)
                for i in range(n_trajs):
                    if active[i]:
                        frames[i][0] = coords_ang[i]
                        last_good_coords[i] = coords_ang[i].copy()
                t_update_frames = time.time()
                # -------- 5) Explosion check + logging ONLY at save_stride --------
                if step % self.save_stride == 0:
                    for i in range(n_trajs):
                        if not active[i]:
                            continue

                        max_dist = self._max_pair_distance(coords_ang[i])
                        if max_dist > self.r_max:
                            label = label_batch[i] if label_batch else "None"
                            print(f"💥 Traj {i} ({label}) exploded at step {step} (max={max_dist:.2f})")

                            log_file.write(f"{i},{label},{step},{max_dist:.6f}\n")
                            log_file.flush()

                            active[i] = False
                            forces_full[i].fill(0.0)
                            energies_full[i] = 0.0
                    t_explosion_check = time.time()
                    # -------- 6) Save --------
                    for i in range(n_trajs):
                        out_coords = last_good_coords[i]  # keep last good for dead ones too
                        g = groups[i]
                        g["coords"][write_step] = out_coords
                        g["pred_forces"][write_step] = forces_full[i]
                        g["pred_energy"][write_step] = energies_full[i]
                        g["md_step"][write_step] = step

                    write_step += 1
                    if write_step % 1000 == 0:
                        h5f.flush()
                        t_save = time.time()
                if step % 1000 == 0:
                    # torch.cuda.empty_cache()
                    # gc.collect()
                    # print("alloc/reserved MB:", torch.cuda.memory_allocated() / 1e6, torch.cuda.memory_reserved() / 1e6, flush=True)
                    timing_file.write(f"{step},{t_frame - t0:.4f},{t_force_calc - t_frame:.4f},{t_explode_force_alloc - t_force_calc:.4f},{t_force_apply - t_explode_force_alloc:.4f},{t_run_step - t_force_apply:.4f},{t_update_frames - t_run_step:.4f},{t_explosion_check - t_update_frames:.4f},{t_save - t_explosion_check if t_save is not None else None}, {torch.cuda.memory_allocated() / 1e6}, {torch.cuda.memory_reserved() / 1e6}\n")
                    timing_file.flush()
        except KeyboardInterrupt:
            print("🛑 Interrupted — saving cleanly.")

        finally:
            log_file.close()
            h5f.close()
            timing_file.close()

            if final_path is not None:
                os.replace(tmp_path, final_path)
                print(f"✅ Final file written: {final_path}")

        print("✅ Finished run_mixed_skip_bad")

    def get_predicted_energy_and_forces(self, data_list):
        data_list = convert_to_data_object(data_list)
        for data in data_list:
            data.z = torch.tensor(data.z)
        # dataset = retrain_dataset(data_list, transforms=self.transform)
        # dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        y_pred, force_pred, _, _ = evaluate(model=self.model, eval_dataset=data_list, batch_size=128, default_dtype= 'float64', device=self.device, compute_stress=False, return_contributions=False)
        num_data = len(data_list)

        force_pred = np.stack(force_pred).reshape(num_data, -1, 3)
        return y_pred, force_pred
    @staticmethod
    def pair_distance_status(coords_angstrom: np.ndarray,
                            r_max: float = 15.0,
                            r_min: float = 2.0):
        """
        Check for both:
        - explosion (atoms too far apart)
        - clash (atoms unrealistically close)

        Returns a dict with:
        max_dist : maximum pair distance (Å)
        min_dist : minimum pair distance (Å)
        exploded : True if any distance > r_max
        clashed  : True if any distance < r_min
        """
        n = coords_angstrom.shape[0]
        if n < 2:
            return {
                "max_dist": 0.0,
                "min_dist": float("inf"),
                "exploded": False,
                "clashed": False,
            }

        dists = pdist(coords_angstrom)   # all pairwise distances

        max_d = float(dists.max())
        min_d = float(dists.min())

        return {
            "max_dist": max_d,
            "min_dist": min_d,
            "exploded": (max_d > r_max),
            "clashed": (min_d < r_min),
        }


def convert_to_numpy_array(value):
    try:
        # Replace the 'tensor' part and the dtype specification to make it a valid list
        value = value.replace('tensor(', '').replace('dtype=torch.float32', '').replace(')', '').strip()
        # Use ast.literal_eval to safely evaluate the string to a list
        value = ast.literal_eval(value)
        # Convert the list to a NumPy array
        return np.array(value)
    except (ValueError, SyntaxError):
        return value



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate batch trajectories.")
    parser.add_argument("--element", type=str, required=True, help="Element symbol (e.g., 'bi')")
    parser.add_argument("--charge", type=int, required=True, help="Charge of the system (e.g., -2)")
    parser.add_argument("--num_atom", type=int, required=True, help="Number of atoms (e.g., 4)")
    parser.add_argument("--model_number", type=int, required=True, help="Model number (e.g., 25)")
    parser.add_argument("--steps", type=int, required=True, help="Number of steps to simulate")
    parser.add_argument("--synthesis", type=str, default='False', help="Synthesis or not (e.g., True or False)")
    parser.add_argument("--compact_type", type=str, required=False,default='bi4', help="Compact type bi4 or bi2")
    parser.add_argument("--T", type=float, default=298.0, help="Temperature in Kelvin")

    args = parser.parse_args()


    # Extract arguments
    element = args.element
    charge = args.charge
    num_atom = args.num_atom
    model_number = args.model_number
    steps = args.steps
    synthesis = args.synthesis
    compact_type = args.compact_type
    temp = args.T
    # Convert synthesis to boolean
    if synthesis.lower() == 'true':
        synthesis = True
        if compact_type == 'bi2':
            prefix = f"{element}{num_atom}{charge}_samples_bi2"
        else:
            prefix = f"{element}{num_atom}{charge}_samples"
    elif synthesis.lower() == 'false':
        synthesis = False
        prefix = f"{element}{num_atom}{charge}"
    else:
        raise ValueError("Invalid value for synthesis. Use 'True' or 'False'.")
    


    # Generate file prefix and result path
    
    # prefix = f"{element}{num_atom}{charge}_samples"
    print(prefix)
    
    result_path = f'../../results/{prefix}'

    job_tmp = Path(os.environ["PAL_MACE_JOB_TMP"]) if "PAL_MACE_JOB_TMP" in os.environ else Path("./tmp")
    print("Job temp dir:", job_tmp)

    result_path = job_tmp / "results" / prefix
    result_path.mkdir(parents=True, exist_ok=True)

    file_path = f"../../usr/pretrain/results/charge_embedding/{prefix}_logs"
    print(result_path)

    df = pd.read_csv(f'{file_path}/{prefix}.csv', delimiter=',', on_bad_lines='skip')



    print(f"Newly generated validation samples: {len(df)}")
    # df['node_feature'] = df['node_feature'].apply(lambda x: convert_to_numpy_array(x)) 

    # df['node_feature'] = [x.reshape(num_atom, 3) for x in df['node_feature']]
    df['coordinates'] = df['coordinates'].apply(lambda x: convert_to_numpy_array(x)) 
    df['coordinates'] = [x.reshape(num_atom, 3) for x in df['coordinates']]
    
    df['atoms'] = df["atoms"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    # convert atom type to elementt number, list of elements
    df['atoms'] = df['atoms'].apply(lambda x: [periodictable.elements.symbol(i).number for i in x])
    df['atoms'] = df['atoms'].apply(lambda x: np.array(x))
    atom_number = df['atoms'][0]
    batch_size = 50
    # --- Stratified Sampling if "source" exists ---
    if (
        "source" in df.columns
        and df["source"].notna().any()
        and (df["source"].astype(str).str.strip() != "").any()
    ):
        # Optionally remove "real" rows if you don’t want them
        df['source'] = df['source'].astype(str).str.strip().str.lower()
        print("Performing stratified sampling based on 'source' column.")
        # df = df[df["source"].astype(str).str.strip().str.lower() != "real"].reset_index(drop=True)

        df_syn = df[df["source"] != "real"].reset_index(drop=True)
        len_syn = len(df_syn)
        df_real = df[df["source"] == "real"].reset_index(drop=True)
        #sample same number of real data as synthetic data if possible
        
        n_total = min(batch_size, len(df_syn))
        df_sampled = stratified_sample(df_syn, "source", n_total)
        df_real_sampled = df_real.sample(len(df_sampled) // 2, random_state=42).reset_index(drop=True)
        df_sampled = pd.concat([df_sampled, df_real_sampled], ignore_index=True).reset_index(drop=True)
        label_batch = df_sampled['source'].to_list()
        print(f"Stratified sampling successful: {len(df_sampled)} samples selected.")
        print("each source distribution in sampled data:")
        print(df_sampled['source'].value_counts())


    else:
        # No valid "source" column → random sampling
        df_sampled = df.sample(n=min(batch_size, len(df)), random_state=42).reset_index(drop=True)
        label_batch = [None] * len(df_sampled)

    coordinates_batch = df_sampled['coordinates'].to_list()

    data_batch = [
        [coordinates_batch[i], atom_number, None, None, charge, None, None, label_batch[i] , None ]
        for i in range(len(coordinates_batch))
    ] 

    # Define the directory path
    dir_path = f'../../trajs/{prefix}'
    print("finished generating batch trajectories")

    # Ensure the directory exists
    os.makedirs(dir_path, exist_ok=True)   

    traj_gen = Generate_TrajsBatch(data_batch, result_path, model_number, prefix, temperature=temp)
    out_file = f"{result_path}/{model_number}_{steps}steps_traj.h5"

    try:
        traj_gen.run_mixed_skip_bad(
        steps=steps,
        h5_path=out_file,      # <-- write to temp file!
        label_batch=label_batch,
        final_path=out_file,   # <-- rename to final file at the end
    )
    except Exception:
        traceback.print_exc()
        raise
