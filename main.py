import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf

from data_loader import load_data

train_ds, val_ds, target_ds = load_data("amazon", "dslr")
