# Sirius-Fleet: Multi-Task Interactive Robot Fleet Learning with Visual World Models

<a href="https://ut-austin-rpl.github.io/sirius-fleet/" target="_blank"><img src="liu-corl24-siriusfleet.jpg" width="90%" /></a>

[Huihan Liu](https://huihanl.github.io/), [Yu Zhang](https://www.linkedin.com/in/yu-zhang-b004a9290/?trk=contact-info), [Vaarij Betala](https://www.linkedin.com/in/vaarij-betala/), [Evan Zhang](https://www.linkedin.com/in/evan-zhang-81a9a9269/), [James Liu](https://www.linkedin.com/in/jamesshuangliu/), [Crystal Ding](https://www.linkedin.com/in/ding-crystal/), [Yuke Zhu](https://yukezhu.me/)
<br> [UT Austin Robot Perception and Learning Lab](https://rpl.cs.utexas.edu/)
<br> Conference on Robot Learning, 2024
<br> **[[Paper]](https://arxiv.org/abs/2410.22689)** &nbsp;**[[Project Website]](https://ut-austin-rpl.github.io/sirius-fleet/)**

This codebase is build off open-source codebase [robocasa](https://github.com/robocasa/robocasa) and [robomimic](https://github.com/ARISE-Initiative/robomimic). 

## Installation

1. Clone repo and set up conda environment: 

```
git clone https://github.com/UT-Austin-RPL/sirius-fleet
conda env create -f sirius-fleet.yml
```

2. Set up robosuite dependency (important: use the master branch!):

```
git clone https://github.com/ARISE-Initiative/robosuite
cd robosuite
pip install -e .
```

3. Install Sirius Fleet

```
cd sirius-fleet
pip install -e .
```


## Usage 

### Downloading data

robocasa (simulation): [data]()

mutex (real robot): [data]()

### Downloading Pretrained World Model Checkpoints

robocasa (simulation): [checkpoint](https://utexas.box.com/s/8j3ktqg6ckig515n0479jcy4kfdydqqo)

mutex (real robot): [checkpoint](https://utexas.box.com/s/i4cw5q9ktq80sgpzcet4c4x4s0a6hmix)

### Loading pretrained checkpoints

```
python robomimic/scripts/load_policy_example.py robocasa.pth
```

### Training Sirius-Fleet

#### World Model

```
python robomimic/scripts/train.py --config train_world_model.json
```

#### Failure Classifier

We train the failure classifier basedo on the world model checkpoint `world_model.pth` from the previous timestep. 

Change the data path for your own task (e.g., `OpenDoorSingleHinge.hdf5`). Note that this dataset is assumed to contain the attribute `` as classifier labels. 

```
python robomimic/scripts/train.py --config train_failure_classifier.json --pretrained_world_model world_model.pth --data_path OpenDoorSingleHinge.hdf5
```

#### Policy

We use BC-Transformer policy architecture.

```
python robomimic/scripts/train.py --config train_policy.json
```

## Acknowledgements

This codebase is largely built off [robocasa](https://github.com/robocasa/robocasa), [robomimic](https://github.com/ARISE-Initiative/robomimic) and [robosuite](https://github.com/ARISE-Initiative/robosuite). 

For real-robot experiments, we used [Deoxys](https://ut-austin-rpl.github.io/deoxys-docs/html/getting_started/overview.html), a controller library for Franka Emika Panda developed by [Yifeng Zhu](https://zhuyifengzju.github.io/).

<br>