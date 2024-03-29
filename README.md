# Domain-Adaptation

This project contains code for different domain adaptation methods on Office31 dataset.

Available methods include: DANN, CDAN, MCD, CORAL, MMD

This project can be easily extended to use on other datasets or perform other adaptaion methods. (Check Code Structure to find out where you need to change.)

## Prepare Environment

Install the requirements using conda from `requirements-conda.txt` (or using pip from `requirements.txt`). 

*My setting:* conda 22.11.1 with Python 3.8

If you have trouble with installing torch check [this link](https://pytorch.org/get-started/previous-versions/)
to find the currect version for your device.

## Run

1. Go to `src` directory.
2. Run `main.py` with appropriate arguments.

Examples:

- Perform MMD and CORAL adaptation on all 6 domain adaptations tasks from office31 dataset:

```bash
python main.py --model_names MMD CORAL --batch_size 32 
```

- Tuning parameters of DANN model

```bash
python main.py --max_epochs 10 --patience 3 --trials_count 1 --model_names DANN --num_workers 2 --batch_size 32 --source amazon --target webcam --hp_tune True 
```

Check "Available arguments" in "Code Structure" section for all the available arguments.

## Code Structure

The main code for project is located in the `src/` directory.

- `main.py`: The entry to program
  - **Available arguments:**

    - `max_epochs`: Maximum number of epochs to run the training

    - `patience`: Maximum number of epochs to continue if no improvement is seen (Early stopping parameter)

    - `batch_size`

    - `num_workers`

    - `trials_count`: Number of trials to run each of the tasks

    - `initial_trial`: The number to start indexing the trials from

    - `download`: Whether to download the dataset or not

    - `root`: Path to the root of project

    - `data_root`: Path to the data root

    - `results_root`: Path to the directory to store the results

    - `model_names`: Names of models to run separated by space - available options: DANN, CDAN, MMD, MCD, CORAL, SOURCE

    - `lr`: learning rate**

    - `gamma`: Gamma value for ExponentialLR**

    - `hp_tune`: Set true of you want to run for different hyperparameters, used for hyperparameter tuning

    - `source`: The source domain to run the training for, training will run for all the available domains if not specified - available options: amazon, dslr, webcam

    - `target`: The target domain to run the training for, training will run for all the available domains if not specified - available options: amazon, dslr, webcam

    - `vishook_frequency`: Number of epochs to wait before save a visualization

    - `source_checkpoint_base_dir`: Path to source-only trained model directory to use as base, set `None` to not use source-trained model***

    - `source_checkpoint_trial_number`: Trail number of source-only trained model to use

- `models.py`: Contains models for adaptation

- `train.py`: Contains base training iteration, dataset is also loaded here

- `classifier_adapter.py`: Contains ClassifierAdapter class which is used for training a source-only model without adaptation

- `load_source.py`: Load source-only trained model to use as base model for adaptation

- `source.py`: Contains source model

- `train_source.py`: Contains base source-only training iteration

- `utils.py`: Contains utility classes

- `vis_hook.py`: Contains VizHook class which is used for visualization

** Use can also set different lr and gammas for different models and tasks by changing `hp_map` in `main.py` directly.

*** For perfoming domain adaptation on source-trained model, one must should train the model for source using option `--model_name SOURCE` first

## Acknowledgements

[Pytorch Adapt](https://github.com/KevinMusgrave/pytorch-adapt/tree/0b0fb63b04c9bd7e2cc6cf45314c7ee9d6e391c0)

