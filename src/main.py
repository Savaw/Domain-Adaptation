import argparse
import logging
import os
from datetime import datetime
from train import train
from train_source import train_source
from utils import HP, DAModels
import tracemalloc

logging.basicConfig()
logging.getLogger("pytorch-adapt").setLevel(logging.WARNING)


DATASET_PAIRS = [("amazon", "webcam"), ("amazon", "dslr"),
                ("webcam", "amazon"), ("webcam", "dslr"),
                ("dslr", "amazon"), ("dslr", "webcam")]


def run_experiment_on_model(args, model_name, trial_number):
    train_fn = train
    if model_name == DAModels.SOURCE:
        train_fn = train_source

    base_output_dir = f"{args.results_root}/{model_name}/{trial_number}"
    os.makedirs(base_output_dir, exist_ok=True)        

    hp_map = {
        'DANN': {
            'a2d': (5e-05, 0.99),
            'a2w': (5e-05, 0.99),
            'd2a': (0.0001, 0.9),
            'd2w': (5e-05, 0.99),
            'w2a': (0.0001, 0.99), 
            'w2d': (0.0001, 0.99),
        },
        'CDAN': {
            'a2d': (1e-05, 1),
            'a2w': (1e-05, 1),
            'd2a': (1e-05, 0.99),
            'd2w': (1e-05, 0.99),
            'w2a': (0.0001, 0.99),
            'w2d': (5e-05, 0.99),
        },
        'MMD': {
            'a2d': (5e-05, 1),
            'a2w': (5e-05, 0.99),
            'd2a': (0.0001, 0.99),
            'd2w': (5e-05, 0.9),
            'w2a': (0.0001, 0.99),
            'w2d': (1e-05, 0.99),
        },
        'MCD': {
            'a2d': (1e-05, 0.9),
            'a2w': (0.0001, 1),
            'd2a': (1e-05, 0.9),
            'd2w': (1e-05, 0.99),
            'w2a': (1e-05, 0.9),
            'w2d': (5*1e-6, 0.99),
        },
        'CORAL': {
            'a2d': (1e-05, 0.99),
            'a2w': (1e-05, 1),
            'd2a': (5*1e-6, 0.99),
            'd2w': (0.0001, 0.99),
            'w2a': (1e-5, 0.99),
            'w2d': (0.0001, 0.99),
        },
    }
    if not args.hp_tune:

        d = datetime.now()
        results_file = f"{base_output_dir}/e{args.max_epochs}_p{args.patience}_{d.strftime('%Y%m%d-%H:%M:%S')}.txt"
        with open(results_file, "w") as myfile:
            myfile.write("pair, source_acc, target_acc, best_epoch, best_score, time, cur, peak, lr, gamma\n")

        for source_domain, target_domain in DATASET_PAIRS:
            pair_name = f"{source_domain[0]}2{target_domain[0]}"

            hp_parmas = hp_map[DAModels.CDAN.name][pair_name]
            lr = args.lr if args.lr else hp_parmas[0]
            gamma = args.gamma if args.gamma else hp_parmas[1]
            hp = HP(lr=lr, gamma=gamma)

            tracemalloc.start()

            train_fn(args, model_name, hp, base_output_dir, results_file, source_domain, target_domain)

            cur, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            with open(results_file, "a") as myfile:
                myfile.write(f"{cur}, {peak}\n")

    
    else:
        # gamma_list = [1, 0.99, 0.9, 0.8]
        # lr_list = [1e-5, 3*1e-5, 1e-4, 3*1e-4]
        hp_values = {
            1e-4: [0.99, 0.9],
            1e-5: [0.99, 0.9],
            5*1e-5: [0.99, 0.9],
            5*1e-6: [0.99]
            # 5*1e-4: [1, 0.99],
            # 1e-3: [0.99, 0.9, 0.8],
        }

        d = datetime.now()
        hp_file = f"{base_output_dir}/hp_e{args.max_epochs}_p{args.patience}_{d.strftime('%Y%m%d-%H:%M:%S')}.txt"
        with open(hp_file, "w") as myfile:
            myfile.write("lr, gamma, pair, source_acc, target_acc, best_epoch, best_score\n")

        hp_best = None
        hp_best_score = None

        for lr, gamma_list in hp_values.items():
        # for lr in lr_list:
            for gamma in gamma_list:
                hp = HP(lr=lr, gamma=gamma)
                print("HP:", hp)

                for source_domain, target_domain in DATASET_PAIRS:
                    _, _, best_score = \
                        train_fn(args, model_name, hp, base_output_dir, hp_file, source_domain, target_domain)
                        
                    if best_score is not None and (hp_best_score is None or hp_best_score < best_score):
                        hp_best = hp
                        hp_best_score = best_score

        with open(hp_file, "a") as myfile:
            myfile.write(f"\nbest: {hp_best.lr}, {hp_best.gamma}\n")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epochs', default=60, type=int)
    parser.add_argument('--patience', default=10, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--trials_count', default=3, type=int)
    parser.add_argument('--initial_trial', default=0, type=int)
    parser.add_argument('--download', default=False, type=bool)
    parser.add_argument('--root', default="../")
    parser.add_argument('--data_root', default="../datasets/pytorch-adapt/")
    parser.add_argument('--results_root', default="../results/")
    parser.add_argument('--model_names', default=["DANN"], nargs='+')
    parser.add_argument('--lr', default=None, type=float)
    parser.add_argument('--gamma', default=None, type=float)
    parser.add_argument('--hp_tune', default=False, type=bool)
    parser.add_argument('--source', default=None)
    parser.add_argument('--target', default=None) 
    parser.add_argument('--vishook_frequency', default=5, type=int)
    parser.add_argument('--source_checkpoint_base_dir', default=None) # default='../results/DAModels.SOURCE/'
    parser.add_argument('--source_checkpoint_trial_number', default=-1, type=int)

    args = parser.parse_args()

    print(args)

    for trial_number in range(args.initial_trial, args.initial_trial + args.trials_count):
        if args.source_checkpoint_trial_number == -1:
            args.source_checkpoint_trial_number = trial_number
        
        for model_name in args.model_names:
            try:
                model_enum = DAModels(model_name)
            except ValueError:
                logging.warning(f"Model {model_name} not found. skipping...")
                continue

            run_experiment_on_model(args, model_enum, trial_number)

# Tune
# python main.py --max_epochs 10 --patience 3 --trials_count 1 --model_names MCD DANN CDAN CORAL MMD --source amazon --target webcam --hp_tune True --num_workers 4 --batch_size 64 --vishook_frequency 10


# Coral
# CUDA_VISIBLE_DEVICES=0 python main.py --max_epochs 60 --patience 10 --trials_count 3 --model_names CORAL --num_workers 1 --batch_size 40

# All
# CUDA_VISIBLE_DEVICES=0 python main.py --max_epochs 100 --patience 5 --trials_count 3 --model_names DANN MMD CDAN  --num_workers 2 --batch_size 32

# ALL 2
# CUDA_VISIBLE_DEVICES=0 python main.py --max_epochs 100 --patience 10 --trials_count 1 --initial_trial 10 --model_names MCD DANN CDAN CORAL MMD --lr 0.00005 --gamma 0.99 --num_workers 2 --batch_size 32 --vishook_frequency 10 > log.txt

# python main.py --max_epochs 100 --patience 10 --trials_count 1 --initial_trial 11 --model_names MCD DANN CDAN CORAL MMD --lr 0.0001 --gamma 0.9 --num_workers 2 --batch_size 32 --vishook_frequency 10 > log_09_2.txt

# VIS
# CUDA_VISIBLE_DEVICES=0 python main.py --max_epochs 20 --patience 5 --trials_count 1 --source amazon --target webcam --model_names DANN --num_workers 2 --batch_size 32 --vishook_frequency 3 --lr 0.0001 --gamma 0.8

# TEST
# CUDA_VISIBLE_DEVICES=0 python main.py --max_epochs 2 --patience 1 --trials_count 1 --initial_trial 100 --model_names CORAL --num_workers 1 --batch_size 64


# ALL 2
# CUDA_VISIBLE_DEVICES=0 python main.py --max_epochs 100 --patience 10 --trials_count 1 --initial_trial 20 --model_names MCD DANN CDAN CORAL MMD --lr 0.0001 --gamma 0.1 --num_workers 1 --batch_size 64 --vishook_frequency 10


# 1
# CUDA_VISIBLE_DEVICES=0 python main.py --max_epochs 1 --patience 1 --trials_count 1 --initial_trial 100 --model_names DANN --lr 0.0001 --gamma 0.1 --num_workers 1 --batch_size 64 --vishook_frequency 10


# SOURCE
## TEST
# CUDA_VISIBLE_DEVICES=0 python main.py --max_epochs 1 --patience 1 --trials_count 1 --initial_trial 100 --model_names SOURCE --lr 0.0005 --gamma 0.99 --num_workers 1 --batch_size 64 --vishook_frequency 10
# CUDA_VISIBLE_DEVICES=1 python main.py --max_epochs 1 --patience 1 --trials_count 1 --initial_trial 100 --model_names DANN --lr 0.0005 --gamma 0.99 --num_workers 1 --batch_size 64 --vishook_frequency 10


## REAL
# CUDA_VISIBLE_DEVICES=0 python main.py --max_epochs 60 --patience 3 --trials_count 3 --initial_trial 1000 --model_names SOURCE --lr 0.0005 --gamma 0.99 --num_workers 2 --batch_size 64 --vishook_frequency 10
### HP tune a2w
# CUDA_VISIBLE_DEVICES=0 python main.py --max_epochs 10 --patience 3 --trials_count 1 --initial_trial 999 --model_names DANN CDAN MMD --num_workers 2 --batch_size 32 --vishook_frequency 10 --source amazon --target webcam --hp_tune True --source_checkpoint_trial_number 1000
# CUDA_VISIBLE_DEVICES=1 python main.py --max_epochs 10 --patience 3 --trials_count 1 --initial_trial 999 --model_names MCD CORAL MMD --num_workers 2 --batch_size 32 --vishook_frequency 10 --source amazon --target webcam --hp_tune True --source_checkpoint_trial_number 1000
### HP tune d2a
# CUDA_VISIBLE_DEVICES=0 python main.py --max_epochs 10 --patience 3 --trials_count 1 --initial_trial 999 --model_names DANN CDAN MMD --num_workers 2 --batch_size 32 --vishook_frequency 10 --source dslr --target amazon --hp_tune True --source_checkpoint_trial_number 1000
# CUDA_VISIBLE_DEVICES=1 python main.py --max_epochs 10 --patience 3 --trials_count 1 --initial_trial 999 --model_names MCD CORAL MMD --num_workers 2 --batch_size 32 --vishook_frequency 10 --source dslr --target amazon --hp_tune True --source_checkpoint_trial_number 1000
### train
# CUDA_VISIBLE_DEVICES=0 python main.py --max_epochs 60 --patience 10 --trials_count 5 --initial_trial 1000 --model_names DANN CDAN MMD --num_workers 2 --batch_size 32 --vishook_frequency 10 --source_checkpoint_trial_number 1000
# CUDA_VISIBLE_DEVICES=1 python main.py --max_epochs 60 --patience 10 --trials_count 5 --initial_trial 1000 --model_names MCD CORAL --num_workers 2 --batch_size 32 --vishook_frequency 10 --source_checkpoint_trial_number 1000



## REAL
# CUDA_VISIBLE_DEVICES=0 python main.py --max_epochs 10 --patience 3 --trials_count 3 --initial_trial 2000 --model_names SOURCE --lr 0.0005 --gamma 0.99 --num_workers 2 --batch_size 128 --vishook_frequency 10
### HP tune a2w
# CUDA_VISIBLE_DEVICES=0 python main.py --max_epochs 10 --patience 3 --trials_count 1 --initial_trial 1999 --model_names DANN CDAN MMD --num_workers 2 --batch_size 32 --vishook_frequency 10 --source amazon --target webcam --hp_tune True --source_checkpoint_trial_number 2001
# CUDA_VISIBLE_DEVICES=1 python main.py --max_epochs 10 --patience 3 --trials_count 1 --initial_trial 1999 --model_names MCD CORAL MMD --num_workers 2 --batch_size 32 --vishook_frequency 10 --source amazon --target webcam --hp_tune True --source_checkpoint_trial_number 2001
### HP tune d2a
# CUDA_VISIBLE_DEVICES=1 python main.py --max_epochs 10 --patience 3 --trials_count 1 --initial_trial 1999 --model_names MCD DANN CDAN MMD CORAL --num_workers 2 --batch_size 32 --vishook_frequency 10 --source dslr --target amazon --hp_tune True --source_checkpoint_trial_number 2001 
### train
# CUDA_VISIBLE_DEVICES=0 python main.py --max_epochs 60 --patience 10 --trials_count 5 --initial_trial 2000 --model_names MCD DANN CDAN --num_workers 2 --batch_size 32 --vishook_frequency 10 --source_checkpoint_trial_number 2001
# CUDA_VISIBLE_DEVICES=1 python main.py --max_epochs 60 --patience 10 --trials_count 5 --initial_trial 2000 --model_names MMD CORAL --num_workers 2 --batch_size 32 --vishook_frequency 10 --source_checkpoint_trial_number 2001
