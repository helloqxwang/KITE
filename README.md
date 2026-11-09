# KITE: Decoupling Kinematics and Interaction for Zero-Shot Cross-Embodiment Manipulation

**[Project website](https://kite-manip.github.io/) · [Paper (arXiv:2606.22113)](https://arxiv.org/abs/2606.22113)**

Qianxu Wang, Shengmiao Samuel Jin, Kuan Fang · Cornell University

## Overview

KITE (Kinematic Interaction Transfer across Embodiments) splits manipulation into an embodiment-agnostic **interaction policy** and embodiment-specific **kinematic decoders**. The policy predicts **interaction intents** (sparse contact positions and outward normals), encoded as **latent intents** by a shared **intent encoder**. Each robot's kinematic decoder turns a latent intent into that robot's next configuration. The encoder and the decoders are trained only on synthetic data from each robot's kinematic model, so a policy trained on one source robot transfers zero-shot to other embodiments.

## Environment Setup

Requirements: Linux, an NVIDIA GPU, CUDA Toolkit 12.8, a C++ compiler, and Conda. Run all commands from the repository root.

```bash
conda env create -f environment.yml
conda activate kite
python -m pip install --no-deps -r requirements.txt
CUDA_HOME=/usr/local/cuda-12.8 CUB_HOME=/usr/local/cuda-12.8/include MAX_JOBS=2 \
  python -m pip install --no-deps --no-build-isolation ./third_party/pytorch3d

# Set in every new shell.
export MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

### Downloads

```bash
python -m pip install -r requirements-download.txt
python download_assets.py
```

This downloads three archives (~520 MiB) from the [release folder on Google Drive](https://drive.google.com/drive/folders/1TtcGy81tCMcYyHCjUKXQsZXcpSYNQvfd), verifies them, and installs their files in the repository.

| Archive | Contents | Needed for |
|---|---|---|
| `kite-decoder-v1-20261026.tar.gz` | Pretrained intent encoder and decoder bank (`checkpoints/decoder/release.pt`) | Visualization, Objects Placing |
| `kite-runtime-v1-20261026.tar.gz` | Robot models, surface contact templates, YCB objects, source grasps, start poses | All workflows |
| `kite-examples-v1-20261026.tar.gz` | The 100 held-out Objects Placing test episodes | Objects Placing |

To download only some archives, use `--only`, for example `python download_assets.py --only decoder runtime`. If the download fails, put the archives from the Drive folder in one directory and run `python download_assets.py --archive-dir <directory>`.

**MANO (optional).** Decoder training and the human-hand option in the viewers need the licensed [MANO](https://mano.is.tue.mpg.de/) models. Copy `MANO_RIGHT.pkl` and `MANO_LEFT.pkl` into `assets/mano_models/`, then check with `python scripts/check_mano_installation.py`.

## Kinematic Decoder Training

The intent encoder and the decoder bank for Wuji, Allegro, Barrett, Robotiq+Stick, and MANO are trained jointly on synthetic data. No demonstrations are used. Each example samples a configuration, takes two to six points on the robot's surface as the interaction intent, and perturbs the configuration to get the decoder's initial input. The data is generated on the first run and cached (about 84 GB).

```bash
python train_kinematic_decoder.py
```

The default configuration, [`conf/kite.yaml`](conf/kite.yaml), is the one used for the released `checkpoints/decoder/release.pt`. Checkpoints are written to `outputs/decoder_training/checkpoints/`. To view one, add `visualization.checkpoint_path=<checkpoint.pt>` to a viewer command.

## Visualization

Each viewer loads the pretrained decoder bank and serves an interactive 3D scene at the address it prints.

| Command | What it shows |
|---|---|
| `python visualize_kinematic_decoder.py` | Goal configurations decoded from synthetic interaction intents, for one embodiment |
| `python visualize_cross_embodiment.py` | One latent intent decoded by several embodiments |
| `python visualize_decoder_initialization.py` | How the decoded configuration changes with the initial configuration |
| `python visualize_contact_trajectory.py` | A target embodiment following a source embodiment's contact trajectory, frame by frame |

## Full Pipeline: Objects Placing (Pick-and-Place)

A Robotiq gripper picks up one of six YCB objects and places it in a bin. An iDP3 interaction policy trained only on Robotiq demonstrations then runs zero-shot on Barrett, Allegro, and Wuji hands through their kinematic decoders.

**1. Generate source demonstrations.** This collects 2,500 Robotiq demonstrations in MuJoCo: 2,400 for training and 100 for validation.

```bash
python experiment/collect_object_suite.py --config-name=collect_object_suite/paper_2500
```

**2. Extract interaction intents** from the demonstrations and encode them with the pretrained intent encoder.

```bash
python experiment/encode_contacts.py --config-name=train_policy/object_suite_paper2400 \
  preprocess.intent_encoder_device=cuda:0
```

**3. Train the interaction policy.** Checkpoints are saved to `experiment/checkpoints/high_level_policy/object_suite_paper2400_main_paper_contacts_v1/`.

```bash
python experiment/train_policy.py --config-name=train_policy/object_suite_paper2400 \
  seed=1 training.epochs=200
```

**4. Run the trained policy on the target hands** over the 100 held-out test episodes. Each run prints the success rate.

```bash
for hand in barrett allegro wuji; do
  python experiment/evaluate_object_suite.py --config-name=evaluate_policy/object_suite_${hand}
done
```

**Oracle Intent.** To test the kinematic decoders without a trained policy, replay the latent intents encoded from the held-out source demonstrations:

```bash
for hand in barrett allegro wuji; do
  python experiment/rollout_gt_contacts.py --config-name=rollout_gt_contacts/object_suite_${hand}
done
```

## Citation

```bibtex
@misc{wang2026kitedecouplingkinematicsinteraction,
  title={KITE: Decoupling Kinematics and Interaction for Zero-Shot Cross-Embodiment Manipulation},
  author={Qianxu Wang and Shengmiao Samuel Jin and Kuan Fang},
  year={2026},
  eprint={2606.22113},
  archivePrefix={arXiv},
  primaryClass={cs.RO}
}
```
