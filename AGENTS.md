# PAL_MACE Agent Guide

This file is the working map for agents modifying `PAL_MACE`. Read it before changing the pretraining pipeline, uncertainty statistics, MPI runtime, record metadata, configuration generators, model layout, standalone MD workflow, or job scripts.

## Project purpose

PAL_MACE is a parallel active-learning workflow for bismuth clusters. It couples:

- OpenMM molecular-dynamics generators;
- an ensemble of pretrained/retrained MACE models for energy and force prediction;
- ensemble-uncertainty and structural filters;
- TURBOMOLE calculations as the ground-truth oracle;
- online MACE retraining; and
- MPI controllers that move flattened NumPy records between those workers.

The full intended project lifecycle is:

```text
parsed quantum-chemistry data
  -> bootstrap-pretrain a MACE ensemble
  -> measure fixed-validation ensemble uncertainty
  -> use those models and statistics to start online active learning
  -> repeatedly acquire oracle labels and retrain the ensemble
  -> select one resulting AL model
  -> run independent long MD trajectories with that model
  -> analyze the HDF5 trajectories and AL results
```

These are three distinct execution phases. `usr/pretrain/` prepares the starting ensemble and calibration statistics. `main.py` and the other modules directly under `usr/` implement online AL. `usr/MD/` runs post-AL trajectory experiments with one selected model and is not part of the MPI loop.

## Repository structure

```text
PAL_MACE/
├── AGENTS.md                         this guide
├── README.md                         short user-facing notes
├── LICENSE
├── environment.yaml, mace_al.yaml     environment specifications
├── install.sh                         project environment/install helper
├── main.py                           current MPI AL entry point and rank routing
├── main_v1.py, main_v2.py             older MPI implementations for reference
├── interface.py                      constructors connecting main.py to usr modules
├── generate_config_yaml.py           generate config.yaml for one AL system
├── generate_al_setting.py            generate al_setting.py role/process map
├── generate_config_yaml_no_v.py       experimental alternate record/config format
├── optimized.csv                     per-prefix geometry and threshold metadata
├── script/                           Slurm launchers for online AL
├── cleanup.py, clean.sh, cache_del.py maintenance helpers; inspect before running
├── check_buffer.py, get_max_dist.py   one-off inspection/data utilities
├── usr/
│   ├── generator_multi_traj.py       online OpenMM trajectory generators
│   ├── model_multi_traj.py           AL predictors and online MACE trainers
│   ├── oracle.py                     TURBOMOLE ground-truth worker
│   ├── utils_multi_traj.py           uncertainty, filtering, and queue logic
│   ├── starting_point_pool.py        optional mutable starting-structure pool
│   ├── *_no_v.py                     experimental alternate model/record path
│   ├── pretrain/
│   │   ├── raw/                      parsed input CSV files
│   │   ├── configs/                  MACE pretraining configurations
│   │   ├── scripts/                  current pretraining Slurm launchers
│   │   ├── charge_embed_scripts/      older launchers using a missing entry point
│   │   ├── samples/                  fixed/bootstrap train and validation splits
│   │   ├── results/                  pretrained models, logs, and statistics tools
│   │   ├── functions/                pretraining configuration helpers
│   │   ├── full_data_charge_embed/    alternative full-data/OOB experiments
│   │   ├── full_data_latent_normalized/ alternative latent-charge experiments
│   │   ├── mace_scripts/, test/       model experiments and checks
│   │   ├── boot_strap_with_fixed_samples.py
│   │   ├── data.py                   CSV-to-PyG conversion and data structures
│   │   ├── evaluation.py             common energy/force/MD evaluation helpers
│   │   └── plot.py and utility *.py   diagnostics and one-off data maintenance
│   └── MD/
│       ├── batch_traj_full_h5.py      standard batched CUDA MD/HDF5 runner
│       ├── bi2_md.py, bi4_md.py       fragment-separated starting geometries
│       ├── building_blocks/           optimized Bi fragment XYZ/NPZ inputs
│       ├── long_distance_*.py          older separated-fragment experiments
│       └── submit_multi_T*.py, *.sh   temperature sweeps and Slurm launchers
├── results/                           online AL checkpoints and resumable state
├── trajs/                             copied trajectories and HDF/pickle analyses
├── result_analysis/                   AL acquisition/retraining analyses
├── error_eval/                        ensemble uncertainty/error calibration
└── no_al/                             non-AL baselines and comparisons
```

Some generated data and model directories are large. Treat `usr/pretrain/raw/`, `usr/pretrain/samples/`, both `results/` trees, and `trajs/` as scientific artifacts rather than disposable build products. Top-level older `main*` variants and experimental model/config trees remain for research history; `main.py` plus the standard non-`_no_v` modules are the current path.

## Phase 1: pretrain the starting ensemble and derive statistics

### Input contract and naming

Run the canonical pretraining commands from `usr/pretrain/`. The input is:

```text
raw/<prefix>_parsed.csv
```

Each row must contain `atoms`, `coordinates`, `total_energy`, and `forces`; `source` is optional. `data.py::big_list()` converts element symbols to atomic numbers, reshapes coordinates and forces to `[N,3]`, and attaches atom count and charge. The canonical file must already contain physical forces. `gradient_to_force.py` negates a gradient column for old raw data; using it on forces a second time silently reverses the sign.

`boot_strap_with_fixed_samples.py` constructs the prefix as:

```python
prefix = f"{atom.lower()}{num_atom}{charge}"
```

For example, Bi14 at charge -6 with the samples suffix uses:

```bash
cd usr/pretrain
python boot_strap_with_fixed_samples.py \
  --atom bi --num_atom 14 --charge="-6_samples" \
  --num_samples 5 --config charge_embedding.yaml \
  --results_dir results/charge_embedding
```

This yields `bi14-6_samples`. The current matching Slurm launcher is `usr/pretrain/scripts/charge_embed_scripts/bi14-6_samples.sh`; launch it from its own directory because its `cd ../..` is relative. The similarly named older scripts directly under `usr/pretrain/charge_embed_scripts/` call the absent `synthetic_boot_train.py` and are stale.

### Split, bootstrap, and training behavior

The standard trainer performs the following operations:

1. Reserve one fixed 10% test/holdout set and save it as `results/charge_embedding/<prefix>_logs/<prefix>.csv`. If `source` exists, the split is stratified by source.
2. Split the remaining 90% into bootstrap-training and fixed-validation pools, approximately 72% and 18% of the original data.
3. For each `sample_i`, bootstrap the training pool with replacement, stratifying within source when available, while reusing the same fixed validation pool.
4. Save that member's inputs under `samples/<prefix>/sample_i/` as `train_val.pkl`, `train.csv`, and `val.csv`.
5. Train one model and write it under `results/charge_embedding/<prefix>_logs/sample_i/`.

The important per-member artifacts are:

```text
usr/pretrain/samples/<prefix>/sample_i/train.csv
usr/pretrain/samples/<prefix>/sample_i/val.csv
usr/pretrain/results/charge_embedding/<prefix>_logs/sample_i/<prefix>.model
usr/pretrain/results/charge_embedding/<prefix>_logs/sample_i/logs/<prefix>_run-123.log
usr/pretrain/results/charge_embedding/<prefix>_logs/sample_i/results/metrics.pkl
```

`generate_config_yaml.py --use_pretrain_reference True --num_models <N>` can also use the selected ensemble's training data as the structural/energy reference for AL. `N` configures the AL predictor/trainer ensemble and combines `sample_0/train.csv` through `sample_(N-1)/train.csv`. Across those CSVs it verifies a consistent `[N_atoms,3]` coordinate shape, derives `num_atom`, sets `energy_threshold` to the combined median training energy, and sets `max_dist` to the largest pairwise distance observed in any selected row. It derives the otherwise-unused soft/hard energy bounds from the combined energy quartiles.

The generated config records `reference_sources` and `reference_model_indices`, and writes `coord: null`. Because bootstrap members can contain different rows and duplicates, changing `num_models` can change these global AL references. Keep the choice intentional and frozen with the run configuration.

If `train_val.pkl` already exists, the script reloads it instead of recreating the split. Thus an old pickle can preserve an old dataset or split even if the raw CSV or arguments changed. The message that a samples directory “already exists. Exiting.” is misleading: current code continues.

The canonical `configs/charge_embedding.yaml` trains `MACE_with_charge` from scratch with the project MACE fork. Its important defaults include force weight 100, energy weight 1, `r_max: 9`, batch size 32, 200 maximum epochs, RMS-force scaling, EMA, CUDA, and seed 123. `boot_strap_with_fixed_samples.py::run()` derives average atomic energies and scale/shift information from the data, trains, evaluates train/validation energy and force errors, and saves the model. Alternative charge-head, latent-charge, normalized, full-data, and low-`r_max` trees are experiments; they are not interchangeable with the `charge_embedding` paths expected by the standard AL configuration.

### Validation ensemble uncertainty

After all intended ensemble members exist, evaluate their common validation set from `usr/pretrain/results/`:

```bash
cd usr/pretrain/results
python make_val_std_distribution.py \
  --prefix bi14-6_samples \
  --logs_root charge_embedding --samples_root ../samples \
  --k 3 --device cuda --default_dtype float64
```

Choose `--k` or explicit `--indices` deliberately. Ideally the models used to calibrate the thresholds are the same members later loaded as the AL ensemble. The script loads each selected model, checks the validation configuration IDs, evaluates the first member's fixed `val.csv`, and writes per-configuration ensemble energy and force statistics to:

```text
usr/pretrain/results/charge_embedding/<prefix>_logs/<prefix>_VAL_uncertainty/
```

The directory contains `val_ensemble_uncertainty.csv`, plots, and several summary JSON files. `generate_config_yaml.py` specifically consumes the `q75` values from:

```text
energy_std_summary.json
force_std_max_atomnorm_summary.json
force_std_rms_summary.json
```

Do not substitute a differently named statistic without changing the configuration generator and selection semantics together. The stats script may print suggestions involving higher quantiles, but the current generated AL config reads `q75`.

`full_data_charge_embed/` plus `make_oob_std_distribution.py` is a separate full-data/OOB experiment: each configuration is evaluated only by models that did not train on it. It is not the canonical per-prefix fixed-validation route above.

### Pretraining utilities

`full_dataset_gene.py`, `merg.py`, `recall_source.py`, `rename_source.py`, and `bi11-3.py` are hard-coded data-maintenance scripts. Inspect their paths and mutation behavior before use. RMSE readers, plotting scripts, `charge_eval.py`, and test scripts are analysis helpers. They do not produce the exact three AL threshold files unless they explicitly call the canonical statistics path.

## Working-directory assumption

Run configuration generation and `main.py` with `PAL_MACE/` as the current directory. Most paths are deliberately relative to it.

The Slurm scripts in `script/` use `cd ..`, so their current convention assumes the job starts from `PAL_MACE/script/`. Submitting one from another directory can move to the wrong working directory.

Do not assume `config.yaml` or `al_setting.py` is authoritative source code. They are generated run artifacts. The durable inputs are `generate_config_yaml.py`, `generate_al_setting.py`, either `optimized.csv` or the selected pretraining reference CSV, the remaining pretrained assets, and the selected job script.

At MPI startup, rank 0 copies `config.yaml` to `config_used_${SLURM_JOB_ID}.yaml` (or `config_used_default.yaml`) and broadcasts its path through `CONFIG_USED_PATH`. `ConfigLoader` then caches that frozen file in each process. Changing `config.yaml` after startup has no effect on a running job.

## Phase 2: online active learning — MPI architecture

`main.py` assigns contiguous rank groups in this order:

| Role | Ranks | Responsibility |
|---|---:|---|
| Exchange | `0` | Fan generator records out to all predictors, collect ensemble predictions, call filtering utilities, and forward oracle candidates to the manager. |
| Manager | `1` | Queue oracle work, collect labels, form retraining batches, persist buffers, and coordinate trainers. |
| Predictors | starting at `2` | One MACE ensemble member per rank; batch-predict all generator trajectories. |
| Generators | after predictors | Maintain multiple OpenMM trajectories per rank and advance each by one step per exchange cycle. |
| Oracles | after generators | Run one TURBOMOLE energy/gradient calculation at a time. |
| Trainers | after oracles | One online-retrained MACE ensemble member per rank. |

The required process count is:

```text
2 + pred_process + gene_process + orcl_process + ml_process
```

The generated setting makes `ml_process == pred_process`. Keep that equality: model indexing, weight exchange, and predictor-to-trainer checkpoint mapping assume a one-to-one ensemble. `main.py` only asserts that MPI size is at least the required count, but using the exact count is clearer; extra ranks do no work and wait at the final barrier.

All MPI payload arrays are one-dimensional. Sizes use `np.int64`/`MPI.LONG`, and numerical payloads use `np.float64`/`MPI.DOUBLE`. Changing a dtype, record length, role count, collective-call order, or stop header on only one side can deadlock the whole job.

## End-to-end data flow

1. Each generator chooses `num_traj_per_gene` initial structures from `usr/pretrain/samples/<prefix>/sample_0/train.csv` and constructs one OpenMM simulation per structure.
2. A generator advances each trajectory by one 2 fs Langevin step and sends all flattened trajectory records to Exchange.
3. Exchange broadcasts every generator packet to every predictor.
4. Each predictor batches the reconstructed structures through one MACE ensemble member and returns energy and force predictions.
5. `usr/utils_multi_traj.py::prediction_check()` computes ensemble means and sample standard deviations, writes the mean prediction back into each trajectory record, increments structural/energy patience counters, and selects uncertain structures for oracle labeling.
6. Mean forces go back to the generators and drive the next OpenMM step. Oracle candidates go Exchange -> Manager -> an available oracle rank.
7. The oracle runs TURBOMOLE and returns energy plus gradient components. `UserModel.add_trainingset()` negates the returned gradient to store physical forces.
8. After Manager accumulates `retrain_size` oracle responses, it broadcasts the same batch to all trainer ranks. Failed all-zero responses still occupy a batch slot and are discarded later by the trainers.
9. Every trainer extends its own bootstrapped dataset, retrains its ensemble member, saves it, and sends flattened weights to the predictors.
10. If `dynamic_orcale_list` is enabled (the spelling is part of the current API), the newly trained ensemble re-scores the pending oracle queue. The queue is filtered and ranked by force and energy uncertainty before more calculations are scheduled.

The workflow stops if a generator returns `True` or a trainer reaches a stopping condition. Stop propagation touches every MPI role, so preserve the existing collective sequence when editing shutdown code.

## Canonical trajectory record

The standard metadata order is defined by `generate_config_yaml.py` and must remain identical everywhere:

```text
0  coords          float64 [N, 3], Å
1  atomic_numbers  int64   [N]
2  energy          nullable float
3  forces          float64 [N, 3]
4  charge          integer scalar
5  pred_forces     float64 [N, 3]
6  pred_energy     nullable float
7  patience        integer [2] = [energy_patience, structure_patience]
8  velocities      float64 [N, 3]
```

`None` is serialized as the float sentinel `99999999.0`. For the standard format, the flattened record length is `13*N + 5`.

The last field is named `velocities`, but the current generator does not evolve or use it as an OpenMM velocity record. It is carried as a zero-filled placeholder. Do not silently remove it: doing so changes every MPI packet length.

Wire formats add control values around that record:

- Generator -> predictor, per trajectory: `[sent, flat_record...]`.
- Generator -> predictor, per generator: `T` concatenated trajectory blocks.
- Predictor -> Exchange, per generator: `[model_iteration, T * (energy, flat_forces...)]`.
- Exchange -> generator: `[model_iteration, T * (sent, flat_record...)]`.
- Oracle result: `[energy, 3*N gradient components]`.

`_record_invariants()` in `main.py` assumes a bismuth-only system: it finds the contiguous block of atomic number `83` to infer `N`. A multi-element extension requires redesigning this validation.

## Generator behavior

`usr/generator_multi_traj.py::UserGene` is stateful. Each generator rank owns `num_traj_per_gene` independent OpenMM `Simulation` objects.

- Initial positions come from the sample training CSV, not from `config.yaml::coord`.
- OpenMM runs on the CPU in the online AL generator.
- The integrator is Langevin, with a 2 fs timestep and a `1/ps` collision rate.
- MACE forces are assumed to be in eV/Å. `update_forces()` applies the eV/Å -> kJ/mol/nm conversion needed by OpenMM's `CustomExternalForce`.
- A trajectory restarts after 100,000 steps, after either patience counter exceeds `patience_threshold`, or after an OpenMM exception.
- Restarts choose another sample and use a random temperature between 298 K and 700 K.
- `num_traj_per_gene` must not exceed the number of available starting rows when `starting_pool_update=False`, because initialization uses `random.sample()`.

With `starting_pool_update=True`, `usr/starting_point_pool.py` copies the initial training CSV to `results/<prefix>/starting_point_pool.csv`. Each generator tracks used row indices in `generator_<rank>/used.json`. The first trainer can append ten newly labeled structures with the largest force error after retraining. Be careful when the pool is exhausted: `random_pop_indices()` can return fewer indices than requested, while the generator expects a full trajectory batch.

## Prediction, selection, and patience

`usr/model_multi_traj.py::UserModel.predict()` reconstructs all trajectories from all received generator packets, performs one batched evaluation, and repacks outputs by generator.

`prediction_check()` then uses the ensemble to calculate:

- energy sample standard deviation;
- RMS force standard deviation across atoms and coordinates; and
- maximum per-atom norm of the force standard deviation.

A structure enters the oracle queue if any configured uncertainty threshold is met. These thresholds are the `q75` values read from validation-summary JSON files during configuration generation.

Separately, the record's patience counters are incremented when:

- ensemble-mean energy is greater than `energy_threshold + bound`; or
- maximum pair distance is more than 30% above `max_dist`, or minimum pair distance is below 2.5 Å.

Patience controls trajectory restart; uncertainty controls oracle acquisition. They are related but not interchangeable.

The dynamic oracle filter uses only `energy_std_threshold` and `force_rms_std`, then ranks retained structures primarily by force RMS standard deviation. It does not use `force_atom_max_std` during queue re-filtering.

## Model and retraining behavior

Prediction and training share `usr/model_multi_traj.py::UserModel`, selected by `mode`.

For ensemble member `i`, a fresh run requires:

```text
usr/pretrain/results/charge_embedding/<prefix>_logs/sample_<i>/<prefix>.model
usr/pretrain/results/charge_embedding/<prefix>_logs/sample_<i>/logs/<prefix>_run-123.log
usr/pretrain/samples/<prefix>/sample_<i>/train.csv
usr/pretrain/samples/<prefix>/sample_<i>/val.csv
```

The log is parsed for the atomic reference-energy dictionary (`E0s`). Missing logs can therefore fail model construction even when the `.model` exists.

The load flags have narrow meanings:

- `load_model=False` still loads the pretrained model above. It means "do not resume an earlier active-learning checkpoint."
- `load_model=True` loads `results/<prefix>/model_<trainer-rank>.pt` for trainers and the corresponding trainer checkpoint for predictors.
- `load_dataset=False` loads the original bootstrapped `train.csv` and `val.csv`.
- `load_dataset=True` loads `results/<prefix>/<trainer-rank>_added_data.csv`, split by its `type` column.

Resume flags should normally be changed together and only when all rank-specific files exist.

New oracle data is split 90/10 into train/validation. With `boot_strap=True`, each trainer additionally resamples its new training portion with replacement. Oracle failures return an all-zero label vector and are skipped.

Retraining uses the full in-memory train and validation sets, MACE's AL training routine, EMA if configured, and a dummy checkpoint handler. After retraining it saves:

- `model_<rank>.pt`;
- `<rank>_added_data.csv`;
- `retrain_history_<rank>.json` and its log;
- `al_state_<rank>.json`;
- metrics JSON and prediction plots.

Current early stopping protects both the initial validation distribution and newly acquired validation data, but its patience limits are 10,000. Dataset-size gates are therefore more likely to stop a normal run than dual patience.

`UserModel.retrain()` accepts an incoming-data MPI request but does not currently poll it inside the MACE training call. A retraining cycle therefore finishes before that trainer handles the next batch.

On resume, the constructor loads `al_state_<rank>.json` and `retrain_history_<rank>.json` from `results/<prefix>/` so counters, patience state, baseline dataset sizes, and plotting history continue from the prior run. Keep the rank layout unchanged: checkpoint and state filenames are keyed by trainer MPI rank.

Although `gpu_pred` and `gpu_ml` are passed through the interface, `UserModel` currently chooses `cuda:<ensemble-index>` directly. Resource assignments must expose enough visible GPUs for the ensemble indices, or the device-selection logic must be changed consistently.

## Oracle behavior

Despite several legacy variable names mentioning XTB, the active oracle is TURBOMOLE through `quantum_chem_python`:

- basis: `dhf-TZVP`;
- functional: `tpss`;
- method: `ridft`;
- COSMO enabled with infinite epsilon;
- coordinates interpreted as Å; and
- molecular charge taken from the record.

Each oracle rank uses `/tmp/oracle_scratch_<rank>`. On failure it returns an all-zero vector of length `1 + 3*N` and appends the failed reconstructed record to `results/<prefix>/xtbfail` (the filename is also legacy).

Manager waits at least `orcl_time` seconds after dispatch before polling a busy oracle. Reducing or increasing that value changes label-delivery latency, not the quantum calculation itself.

## Configuration sources and adding a prefix

`generate_config_yaml.py` combines three sources:

1. either `optimized.csv`, or the first `num_models` files under `usr/pretrain/samples/<prefix>/sample_<i>/train.csv` when `--use_pretrain_reference True`, for the structural/energy references;
2. `usr/pretrain/results/charge_embedding/<prefix>_logs/<prefix>_VAL_uncertainty/*_summary.json` for uncertainty thresholds; and
3. hard-coded defaults and the CLI flags for the remaining settings.

For a new prefix, all of the following must agree:

- in optimized-reference mode, a row in `optimized.csv` whose lowercased `Name` exactly matches the prefix, plus membership in the `prefixes` list;
- in pretraining-reference mode, valid `sample_0` through `sample_(num_models-1)` training CSVs; this mode does not require an `optimized.csv` row or whitelist membership;
- enough `sample_<i>/train.csv` and `val.csv` files for the predictor/trainer ensemble;
- corresponding pretrained `.model` files and log files;
- validation uncertainty summary JSON files; and
- atom count, charge, metadata shape, model cutoff, and force/energy conventions.

`coord` is required only in optimized-reference mode. No active runtime module reads `config["coord"]`; pretraining-reference mode therefore writes it as YAML `null`. Likewise, `hard_bound`, `soft_bound`, and the standard config's `source` mapping are generated but unused by the current online selection code. `max_dist`, `num_atom`, `energy_threshold`, `bound`, and the three uncertainty thresholds are operational.

`full_dataset=True` is explicitly unsupported by the current generator, model, and config generator. Do not enable it based on old commented branches.

`generate_al_setting.py` currently reads retrain size from `config["args_dict"].get("retrain_size", 50)`, not the top-level `config["retrain_size"]`. The generated value therefore remains 50 unless that behavior is intentionally fixed.

## Starting a fresh AL run from the pretrained ensemble

From `PAL_MACE/`, the normal fresh-run configuration sequence is:

```bash
python generate_config_yaml.py \
  --prefix <prefix> --full_dataset False \
  --num_traj_per_gene <T> \
  --load_model False --load_dataset False \
  --use_pretrain_reference True --num_models 2
python generate_al_setting.py
```

Omit the final two options to retain the legacy `optimized.csv` reference mode. Then use the matching script under `script/` or launch the generated rank layout with `mpirun ... python main.py`. In pretraining-reference mode, configuration generation requires the selected bootstrap CSV, uncertainty JSON files, and model/log artifacts, but no optimized geometry.

In this context, `load_model=False` does **not** mean “run without a model.” It means “start a new AL history from the pretrained `.model` files.” Likewise, `load_dataset=False` means “start from each pretrained member's bootstrap `train.csv` and fixed `val.csv`.” Set the flags to `True` only to resume compatible rank-specific artifacts already stored in `results/<prefix>/`.

The pretraining-to-AL handoff is therefore:

```text
sample_i/<prefix>.model  ──> predictor i and trainer i initial weights
sample_i/train.csv      ──> trainer i initial training set
sample_i/val.csv        ──> trainer i protected validation set
sample_i log            ──> parsed atomic reference energies (E0s)
*_summary.json q75      ──> online uncertainty acquisition thresholds
```

Do not move only the model file and assume the run is reproducible; the log, member-specific datasets, statistics, generated config, and ensemble-member ordering are all part of the starting state.

## Phase 3: select an AL model and run standalone MD

### Model identity and handoff

Online retraining saves the latest checkpoint for each trainer as:

```text
results/<prefix>/model_<trainer-rank>.pt
```

`<trainer-rank>` is the MPI rank assigned in `al_setting.py`, not the zero-based pretraining sample number. With the common generated layout the two trainer ranks are 56 and 57, but always verify the actual role map for the run. Each retraining cycle overwrites that member's `model_<rank>.pt`, so it represents the last saved version rather than a full checkpoint history.

The project does not automatically choose the best member. Select one using the run's protected/new-validation metrics, plots, and stability history, while checking that its rank-specific dataset and retraining history belong to the same run. Standalone MD then uses that single model; it does not calculate ensemble uncertainty and does not invoke an oracle.

There is a path mismatch that must be handled explicitly. The current MD constructor loads:

```text
../../results/<prefix>_org/model_<model_number>.pt
```

when executed from `usr/MD/`, whereas online AL writes `results/<prefix>/model_<rank>.pt`. Before a production MD run, either deliberately stage the selected checkpoint in the expected `<prefix>_org` directory or update and verify the loader path for the intended run. Do not assume that the normal AL output directory is read automatically.

### Standard HDF5 MD workflow

Run `batch_traj_full_h5.py` from `usr/MD/` because both model and data locations use `../../` relative paths. A representative invocation is:

```bash
cd usr/MD
python batch_traj_full_h5.py \
  --element bi --charge -3 --num_atom 11 \
  --model_number 56 --steps 1000000 \
  --synthesis True --T 700.0
```

For the standard synthesis case this constructs prefix `bi<num_atom><charge>_samples`. `--compact_type bi2` changes it to the `_samples_bi2` form; `--synthesis False` omits `_samples`. Check the prefix printed by the program before trusting a long allocation.

Starting structures come from the pretraining fixed holdout:

```text
usr/pretrain/results/charge_embedding/<prefix>_logs/<prefix>.csv
```

They do not come from the final AL dataset. The default batch size is 50. If `source` is present, synthetic structures are sampled stratified by source and additional real structures are sampled at half that count; otherwise rows are sampled randomly.

The runner places every trajectory in one OpenMM `System` as a separate chain and applies independent MACE-predicted external forces. It runs both MACE and OpenMM on CUDA device 0, uses a 2 fs Langevin timestep, evaluates active trajectories in batches, and saves every 50 steps. `1,000,000` steps therefore represents 2 ns, even though several job names say `1ns`.

Trajectories whose maximum pair distance exceeds 30 Å are marked exploded at a save boundary, removed from further inference, and retain their last good frame. The active `run_mixed_skip_bad()` path does not apply all checks present in older helper functions, so do not infer additional clash rejection from unused code.

The HDF5 output is written atomically under:

```text
${PAL_MACE_JOB_TMP:-./tmp}/results/<prefix>/<model_number>_<steps>steps_traj.h5
```

Each `traj_i` group stores `coords`, `pred_forces`, `pred_energy`, and `md_step`, plus a `label` attribute. Coordinates and forces are float32; energies are float64. The same result directory receives an explosion CSV and timing/MACE logs. Current Slurm scripts create a job-local temporary directory, trap normal exit/signals, and `rsync` its `results/` tree to LSDF. Verify the destination in the selected job script; local temporary output can disappear after cleanup if staging fails.

### Fragment-separated Bi11 workflows

`bi4_md.py` constructs Bi4 + Bi7, while `bi2_md.py` constructs Bi2 + Bi7 + Bi2. Both use optimized XYZ building blocks under `usr/MD/building_blocks/`, randomly rotate and place fragments subject to distance/clash rules, save a compressed starting-geometry NPZ, and then call the same HDF5 trajectory engine. Their active model prefix is hard-coded to `bi11-3_samples`; changing charge or composition arguments alone does not select a different checkpoint.

The matching `submit_multi_T_long_dist_bi2.py` and `submit_multi_T_long_dist_bi4.py` scripts generate temperature-specific Slurm jobs. Inspect the active temperature list and generated commands rather than relying on filenames or earlier assignments in the file. `submit_multi_T.py` currently overwrites its first temperature list and effectively uses 500, 525, and 575 K.

`long_distance_non_compact_gene.py` is an older starting-geometry generator. `long_distance_md.py` currently passes `prefix=None` into the model loader and consequently targets `results/None_org/...`; it is incomplete until that path contract is repaired. `traj.sh`, `traj_bi2_compact.sh`, and `run_traj.sh` call the removed `batch_traj_full.py`; use `batch_traj_full_h5.py`/`traj_h5.sh` as the maintained standard path.

After MD, use the scripts under `trajs/` for HDF/pickle conversion, energy trends, RDF/RMSD, and DFT comparisons. `result_analysis/` focuses on acquired AL data and retraining behavior, while `error_eval/` focuses on ensemble-error calibration. These analysis trees are downstream consumers and should not be imported into the online runtime.

## Standard versus `_no_v` variants

The generated runtime uses:

```text
usr/model_multi_traj.py
usr/utils_multi_traj.py
```

The `_no_v` variants are experimental:

- `model_multi_traj_no_v.py` loads pretrained models from `charge_embedding_low_rmax`;
- `generate_config_yaml_no_v.py` replaces the final `[N,3]` velocity field with a scalar `data_type`; and
- `utils_multi_traj_no_v.py` preserves a `source` scalar in that field.

Do not mix standard and `_no_v` components. Their metadata and packet lengths differ, and some imports inside the experimental modules still point to standard helpers. Verify the entire generator/model/utils/config chain before activating this variant in `AL_SETTING["usr_pkg"]`.

## Outputs and persisted state

Online outputs live under `results/<prefix>/` and include:

- `ml_buffer` and `orcl_buffer` pickle files;
- model and dataset files per trainer rank;
- generator restart CSVs and optional starting-pool state;
- acquisition logs, metrics, plots, and retraining histories;
- `oracle_filter.log`, `execution_time.txt`, and `xtbfail`; and
- `log_error.txt`, where most worker stderr is redirected.

Exchange also uses `oracl_buffer_at_EX` (legacy spelling) to persist unsent candidates at shutdown.

Do not delete or overwrite these files as part of an unrelated code change. They may be the only resumable state from a long allocation.

## Safe editing and verification

Before changing core code:

1. Trace both sender and receiver for every modified MPI payload.
2. Recalculate `compute_flat_length(metadata)` and all surrounding control fields.
3. Confirm predictor and trainer counts remain equal.
4. Confirm every process reaches collectives in the same order, including stop paths.
5. Preserve Å/eV/eV-per-Å conventions and the oracle gradient-to-force sign conversion.
6. Preserve relative-path behavior from the `PAL_MACE/` working directory.

There is no formal automated test suite. Proportionate local checks are:

```bash
python -m py_compile main.py interface.py usr/generator_multi_traj.py \
  usr/model_multi_traj.py usr/oracle.py usr/starting_point_pool.py \
  usr/utils_multi_traj.py
bash -n script/<job>.sh
python generate_config_yaml.py --prefix <prefix> --full_dataset False \
  --num_traj_per_gene <T> --load_model False --load_dataset False
python generate_al_setting.py
```

The config-generation check writes `config.yaml`; do not run it casually when a prepared run configuration must be preserved. A full MPI test additionally requires the matching pretrained ensemble, MPI, MACE fork, OpenMM, `quantum_chem_python`, TURBOMOLE environment, and cluster resources.

When diagnosing a run, inspect the generated frozen config, `al_setting.py`, Slurm stdout/stderr, and `results/<prefix>/log_error.txt` together. A worker exception often appears only in the redirected result log while the MPI job looks stalled.
