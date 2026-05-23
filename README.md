# HSG-BC: Hierarchical Subgoal-Guided Behavior Cloning

This repository contains the project page and implementation code for **HSG-BC**, a hierarchical robot manipulation system using LLM-based subgoal planning and behavior cloning-based low-level control for the SO-101 robot arm.

## Project Page

https://ssu-hongki.github.io/hsg-bc/

## Repository Structure

```text
.
├── index.html
├── assets/
│   ├── figure1.jpeg
│   ├── video1.mp4
│   └── evaluation_results.png
│   └── evaluation_LLM.png
└── code/
    ├── record_teleop.py
    ├── postprocess_lerobot_dataset.py
    ├── train_mlp_policy_final.py
    └── integrated_inference_final.py
```

## Installation

Create a Python environment and install the required packages.

```bash
pip install numpy pandas torch pyarrow huggingface_hub
```

For ArUco marker-based vision processing, install OpenCV contrib.

```bash
pip install opencv-contrib-python
```

For SO-101 robot execution, install LeRobot following the official installation guide.

## Execution Guide

### 1. Record Human Demonstrations

Human demonstrations are collected through leader-follower teleoperation.

```bash
python code/record_teleop.py
```

This script records SO-101 robot demonstrations using the leader-follower setup.

### 2. Post-process Demonstrations

After recording, demonstration files are post-processed to attach subgoal labels and build state-subgoal-action tuples.

```bash
python code/postprocess_lerobot_dataset.py
```

The post-processing step constructs training samples in the following format:

```text
(state, subgoal, action)
```

where `state` includes robot state and visual features, `subgoal` is a one-hot subgoal label, and `action` is the robot joint-level action.

### 3. Train Low-Level Policy

Train the MLP behavior cloning policy.

```bash
python code/train_mlp_policy_final.py
```

The policy is trained with MSE loss between the predicted action and the demonstrated action.

### 4. Run Integrated Inference

Set the Hugging Face API token as an environment variable.

For Windows:

```bash
set HF_TOKEN=your_huggingface_token
```

For macOS / Linux:

```bash
export HF_TOKEN=your_huggingface_token
```

Run dry-run inference without connecting the robot.

```bash
python code/integrated_inference_final.py --dry-run "Pour beaker A into beaker B and stir it with the stick."
```

Run inference with the SO-101 robot arm.

```bash
python code/integrated_inference_final.py --port COM4 "Pour beaker A into beaker B and stir it with the stick."
```

Change `COM4` to the correct robot port for your system.

## Method Overview

HSG-BC consists of two main components.

### High-Level Planner

An LLM decomposes a natural-language command into a sequence of subgoals.

### Low-Level Policy

An MLP behavior cloning policy predicts joint-level robot actions conditioned on the current state and subgoal.

The low-level policy is represented as:

```text
π(a_t | s_t, g_t)
```

where `s_t` is the current state, `g_t` is the current subgoal, and `a_t` is the predicted robot action.

## Reference Subgoal Schema

```text
1. Move to source object
2. Pick source object
3. Move source to target
4. Pour source into target
5. Move source back
6. Release source object
7. Move to tool object
8. Pick tool object
9. Move tool to target
10. Stir target
11. Move tool back
12. Release tool object
```

## Demo

The demo video is included in the project page.
