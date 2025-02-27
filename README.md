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
conda env create -f kitchen.yml
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

#### Downloading original data

robocasa (simulation): Follow robocasa's [documentation](https://robocasa.ai/docs/use_cases/downloading_datasets.html) to download the data. Note that the world model training is using the mimicgen dataset. 

mutex (real robot): Download [mutex](https://ut-austin-rpl.github.io/MUTEX/) dataset [here](https://utexas.app.box.com/s/wepivf85cgini0eqpho9jae9c6o99n4e). 

Create symlinks for the data such that robocasa dataset is at `~/robocasa`, and mutex is at `~/mutex`:

```
ln -s $ROBOCASA_DATASET_PATH ~/robocasa
ln -s $MUTEX_DATASET_PATH ~/mutex
```

#### Downloading human-in-the-loop data

```
python scripts/download_hitl_data.py
```

Create symlinks for the data so that the human-in-the-loop data will be saved at `~/robocasa_hitl` and `~/mutex_hitl`:

```
ln -s $HITL_ROBOCASA_DATASET_PATH ~/robocasa_hitl
ln -s $HITL_MUTEX_DATASET_PATH ~/mutex_hitl
```



### Downloading Pretrained World Model Checkpoints

robocasa (simulation): [checkpoint](https://utexas.box.com/s/8j3ktqg6ckig515n0479jcy4kfdydqqo)

mutex (real robot): [checkpoint](https://utexas.box.com/s/i4cw5q9ktq80sgpzcet4c4x4s0a6hmix)

### Loading pretrained checkpoints

```
python robomimic/scripts/load_policy_example.py robocasa.pth
```

### Training Sirius-Fleet

#### World Model

We suggest that you use the config generation script to generate a json file that is customized to the environments and corresponding datasets. It also allows you specify different hyperparameters using the flags, and also sweep different hyperparameter options. See `gen_config_world_model.py` and `../helper.py` for details.

##### Robocasa: 
```
python robomimic/scripts/config_gen/world_model.py --env robocasa --name world_model --batch_size 8
```

##### Mutex:
```
python robomimic/scripts/config_gen/world_model.py --env mutex --name world_model --batch_size 8
```

This will print out the actual command in the format of `python robomimic/scripts/train.py --config xxxx.json`. 

Note: This is equivalent to the case if you already have a `train_world_model.json` file where all configs are already defined. You can directly run: 

```
python robomimic/scripts/train.py --config train_world_model.json
```

This training generates a checkpoint, which we refer to here as `world_model.pth`. It will be used to train the failure classifier below.

#### Failure Classifier

We train the failure classifier based on the world model checkpoint `world_model.pth` from the previous timestep. 

Change the data path for your own task (e.g., `OpenDoorSingleHinge.hdf5`). Note that this dataset is assumed to contain the attribute `intv_labels` as classifier labels. 

```
python robomimic/scripts/train.py --config train_configs/train_failure_classifier.json --pretrained_model world_model.pth --data_path $PATH_TO_DATA/OpenDoorSingleHinge.hdf5
```

To finetune failure classifier on a previous failure classifier checkpoint: 

```
python robomimic/scripts/train.py --config train_configs.train_failure_classifier.json --pretrained_world_model world_model.pth --classifier_ckpt classifier.pth --data_path $PATH_TO_DATA/OpenDoorSingleHinge.hdf5
```

#### Policy

We use BC-Transformer policy architecture.

##### Robocasa: 
```
python robomimic/scripts/config_gen/bc.py --env robocasa --name bc --n_seeds 3
```

##### Mutex: 
```
python robomimic/scripts/config_gen/bc.py --env mutex --name bc --n_seeds 3
```

To finetune the policy on a previous policy checkpoint:

```
python robomimic/scripts/config_gen/bc.py --env $ENV --name bc --n_seeds 3 --ckpt_path $POLICY_CKPT
```

## Acknowledgements

This codebase is largely built off [robocasa](https://github.com/robocasa/robocasa), [robomimic](https://github.com/ARISE-Initiative/robomimic) and [robosuite](https://github.com/ARISE-Initiative/robosuite). 

For real-robot experiments, we used [Deoxys](https://ut-austin-rpl.github.io/deoxys-docs/html/getting_started/overview.html), a controller library for Franka Emika Panda developed by [Yifeng Zhu](https://zhuyifengzju.github.io/).

<br>