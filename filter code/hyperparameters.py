"""
    Author: Mauro Mendez.
    Date: 02/11/2020.

    Hyperparameters for a run.
"""

parameters = {
    # Random Seed
    'seed': 200,

    # Data
    'train_data': '/home/amul/Downloads/radscholors/current code/csv/train',
    'test_data': '/home/amul/Downloads/radscholors/current code/csv/test',  # Path to the training and testing directories

    'img_size': 380,  # Image input size (this might change depending on the model)  # was 380
    'batch_size': 16,  # Input batch size for training (you can change this depending on your GPU ram)
    'data_mean': [0.10608228296041489, 0.10858751088380814, 0.11053004860877991],  # Mean values for each layer (RGB) (THIS CHANGE FOR EVERY DATASET)
    'data_std': [0.15130813419818878, 0.15402470529079437, 0.15593037009239197],  # Std Dev values for each layer (RGB) (THIS CHANGE FOR EVERY DATASET)
    'out_features': 1,  # For binary is 1

    # Model
    'model': 'efficientnet',  # Model to train (This name has to correspond to a model from models.py)
    'optimizer': 'ADAM',  # Optimizer to update model weights (Currently supported: ADAM or SGD)
    'criterion': 'BCEWithLogitsLoss',
    'lear_rate': 0.0001,  # Learning Rate to use
    'min_epochs': 10,  # Minimum number of epochs to train for
    'epochs': 100,  # Number of epochs to train for
    'precision': 16,  # Pytorch precision in bits
    'accumulate_grad_batches': 4,  # the number of batches to estimate the gradient from
    'num_workers': 0  # Number of CPU workers to preload the dataset in parallel
}
