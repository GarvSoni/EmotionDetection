import argparse
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from model_history import plot_model_history
import config
from test import test
from train import train
from model import create_model
from dataset import dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# command line argument
ap = argparse.ArgumentParser()
ap.add_argument("--whattodo",help="train_it/test_it")
whattodo = ap.parse_args().whattodo


model=create_model()
if whattodo == "test_it":
    
    test(model)


if whattodo == "train_it":

    num_train = 28709
    num_val = 7178
    batch_size = 256
    num_epoch = 20


    train_dir = config.train_dir
    val_dir = config.val_dir
    train_generator, validation_generator = dataset(train_dir, val_dir, batch_size)

    train(model, train_generator, num_train, batch_size, num_epoch,validation_generator, num_val, plot_model_history)

