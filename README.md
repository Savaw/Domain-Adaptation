# Domain-Adaptation

## Prepare Environment

Install the requirements from `requirements.txt`.

Python version: 3.8

Having trouble with installing torch?
Use [this link](https://pytorch.org/get-started/previous-versions/) to find the currect version for your device.

## Structure

The main code for project is located in the `src/` directory.

- `main.py`

The entry to program.

**Available arguments:**

-- max_epochs: Maximum number of epochs to run the training

-- patience: Maximum number of epochs to continue if no improvement is seen

-- batch_size

-- num_workers

-- trials_count: Number of trials to run each of the tasks

-- initial_trial: The number to start indexing the trials from

-- download: Whether to download the dataset or not

-- root: Path to the root of project

-- data_root: Path to the data root

-- results_root: Path to the directory to store the results

-- model_names: Names of models to run separated by space - available options: DANN CDAN MMD MCD CORAL

-- lr: learning rate

-- gamma: Gamma value for ExponentialLR scheduler

-- hp_tune: Set true of you want to run for different hyperparameters, used for hyperparameter tuning

-- source: The source domain to run the training for, training will run for all the available domains if not specified - available options: amazon, dslr, webcam

-- target: The target domain to run the training for, training will run for all the available domains if not specified - available options: amazon, dslr, webcam

-- vishook_frequency: Number of epochs to wait before save a visualization

-- source_checkpoint_base_dir: Path to source-only trained model directory to use as base

-- source_checkpoint_trial_number: Trail number of source-only trained model to use

- `models.py`


- `train.py`

- `classifier_adapter.py`

- `load_source.py`

- `source.py`

- `train_source.py`

- `utils.py`

- `vis_hook.py`

