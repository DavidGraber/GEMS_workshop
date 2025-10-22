import argparse
import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
# import wandb
import time

from torch_geometric.data import Dataset
from Dataset import *
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset
from GEMS18 import *


"""
This script trains a machine learning model using PyTorch and PyTorch Geometric. It includes functionalities for 
parsing command-line arguments and training the model with various configurations such as early stopping. 
The script also supports logging and tracking experiments using Weights and Biases (wandb).

Description:
    1. Splits the dataset into n stratified folds for n-fold cross-validation.
    2. Trains the model on the training set and evaluates it on the validation set.
    3. Logs the training and validation metrics for each epoch.
    4. Saves the best model based on the validation metric.
    5. Plots the predictions and residuals of the model at regular intervals. 

    
REQUIRED Command-line Arguments:
    --dataset_path:         REQUIRED - Path to the .pt file containing the dataset.
    --run_name:             REQUIRED - Name of the run for saving results and logs.

OPTIONAL Command-line Arguments (with default values):

    --save_dir:             OPTIONAL - The path for saving results and logs. Default: run_name/

    TRAIN-VALIDATION SPLIT
    --n_folds:              OPTIONAL - Number of stratified folds for n-fold cross-validation
    --fold_to_train:        OPTIONAL - Fold to be used for training
    --random_seed:          OPTIONAL - Random seed for dataset splitting.

    MODEL PARAMETERS
    --model:                OPTIONAL - Name of the model architecture to be used [GEMS18d, ...]
    --loss_func:            OPTIONAL - Loss function to be used ['MSE', 'RMSE', 'wMSE', 'L1', 'Huber'].
    --optim:                OPTIONAL - Optimizer to be used ['Adam', 'Adagrad', 'SGD'].
    --num_epochs:           OPTIONAL - Number of epochs for training.
    --batch_size:           OPTIONAL - Batch size for training.
    --learning_rate:        OPTIONAL - Learning rate for training.
    --weight_decay:         OPTIONAL - Weight decay parameter for training.
    --conv_dropout:         OPTIONAL - Dropout probability for convolutional layers.
    --dropout:              OPTIONAL - Dropout probability for dropout layer in fully-connected NN.

    EARLY STOPPING
    --early_stopping:       OPTIONAL - Whether to use early stopping.
    --early_stop_patience:  OPTIONAL - Patience for early stopping.

    W&B TRACKING
    --wandb:                OPTIONAL - Whether or not to stream the run to Weights and Biases.
    --project_name:         OPTIONAL - Project name for saving run data to Weights and Biases.


EXAMPLE USAGE
python train.py --dataset_path datasets/dataset_train.pt --fold_to_train 1 --run_name first_test early_stop_patience 5    
    
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Training Parameters and Input Dataset Control")

    # REQUIRED: Training Dataset and Run Name
    parser.add_argument("--dataset_path", required=True, help="The path to the .pt file containing the dataset")
    parser.add_argument("--run_name", required=True, help="Name of the Run")

    # Dataset splitting
    parser.add_argument("--n_folds", default=5, type=int, help="The number of stratified folds that should be generated (n-fold-CV)")
    parser.add_argument("--fold_to_train", default=0, type=int, help="Of the n_folds generated, on which fold should the model be trained")

    # Model type and save path
    parser.add_argument("--model", default="GEMS18d", help="The name of the model architecture")
    parser.add_argument("--save_dir", default=None, help="The path for saving results and logs. Default: run_name/")

    # Training Parameters
    parser.add_argument("--num_epochs", default=500, type=int, help="Number of Epochs the model should be trained (int)")
    parser.add_argument("--loss_func", default='RMSE', help="The loss function that will be used ['MSE', 'RMSE', 'wMSE', 'L1', 'Huber']")
    parser.add_argument("--optim", default='SGD', help="The optimizer that will be used ['Adam', 'Adagrad', 'SGD']")
    parser.add_argument("--batch_size", default=256, type=int, help="The Batch Size that should be used for training (int)")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="The learning rate with which the model should train (float)")
    parser.add_argument("--weight_decay", default=0.001, type=float, help="The weight decay parameter with which the model should train (float)")
    parser.add_argument("--conv_dropout", default=0, type=float, help="The dropout probability that should be applied in the convolutional layers")
    parser.add_argument("--dropout", default=0.5, type=float, help="The dropout probability that should be applied in the dropout layer")
    parser.add_argument("--random_seed", default=0, type=int, help="The random seed that should be used for the splitting of the dataset")
    
    # Early stopping
    parser.add_argument("--early_stopping",  default=True, type=lambda x: x.lower() in ['true', '1', 'yes'], help="If early stopping should be used to prevent overfitting")
    parser.add_argument("--early_stop_patience", default=20, type=int, help="For how many epochs the validation loss can cease to decrease without triggering early stop")
    
    # W&B Tracking
    parser.add_argument("--wandb", default=False, type=lambda x: x.lower() in ['true', '1', 'yes'], help="Wheter or not the run should be streamed to Weights and Biases")
    parser.add_argument("--project_name", default=None, help="Project Name for the saving of run data to Weights and Biases")

    return parser.parse_args()


# Function to count number of trainable parameters
def count_parameters(model, trainable=True):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable)


class wMSELoss(torch.nn.Module):
    def __init__(self):
        super(wMSELoss, self).__init__()
    def forward(self, output, targets):
        squared_errors = (output - targets) ** 2
        return torch.mean(squared_errors * (targets + 1))


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = torch.nn.MSELoss()
    def forward(self, output, targets):
        return torch.sqrt(self.mse(output, targets))

        
class EarlyStopper:
    def __init__(self, patience=1):
        self.patience = patience
        self.counter = 0
        self.best_validation_rmse = float('inf')

    def early_stop(self, val_rmse):
        if val_rmse < self.best_validation_rmse:
            self.best_validation_rmse = val_rmse
            self.counter = 0
        elif val_rmse > self.best_validation_rmse:
            self.counter += 1
            if self.counter >= self.patience:
                print(f'Early Stopping: Validation RMSE has not decreased for {self.patience} epochs')
                return True
        return False


# Training Function for 1 Epoch
#-------------------------------------------------------------------------------------------------------------------------------
def train(Model, loader, criterion, optimizer, device):
    Model.train()
        
    # Initialize variables to accumulate metrics
    total_loss = 0.0
    y_true = []
    y_pred = []
                
    for graphbatch in loader:
        graphbatch.to(device)
        targets = graphbatch.y

        # Forward pass
        optimizer.zero_grad()
        output = Model(graphbatch).view(-1)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        # Accumulate loss collect the true and predicted values for later use
        total_loss += loss.item()
        y_true.extend(targets.tolist())
        y_pred.extend(output.tolist())

    # Calculate evaluation metrics
    avg_loss = total_loss / len(loader)

    # Pearson Correlation Coefficient
    corr_matrix = np.corrcoef(y_true, y_pred)
    r = corr_matrix[0, 1]

    # R2 Score
    r2_score = 1 - np.sum((np.array(y_true) - np.array(y_pred)) ** 2) / np.sum((np.array(y_true) - np.mean(np.array(y_true))) ** 2)

    # RMSE in pK unit
    min=0
    max=16
    true_labels_unscaled = torch.tensor(y_true) * (max - min) + min
    predictions_unscaled = torch.tensor(y_pred) * (max - min) + min
    criter = RMSELoss()
    rmse = criter(predictions_unscaled, true_labels_unscaled)

    return avg_loss, r, rmse, r2_score, y_true, y_pred
#-------------------------------------------------------------------------------------------------------------------------------


# Evaluation Function
#-------------------------------------------------------------------------------------------------------------------------------
def evaluate(Model, loader, criterion, device):
    Model.eval()

    # Initialize variables to accumulate the evaluation results
    total_loss = 0.0
    y_true = []
    y_pred = []

    # Disable gradient calculation during evaluation
    with torch.no_grad():
        for graphbatch in loader:

            graphbatch.to(device)
            targets = graphbatch.y

            # Forward pass
            output = Model(graphbatch).view(-1)
            loss = criterion(output, targets)

            # Accumulate loss and collect the true and predicted values for later use
            total_loss += loss.item()
            y_true.extend(targets.tolist())
            y_pred.extend(output.tolist())


    # Calculate evaluation metrics
    eval_loss = total_loss / len(loader)

    # Pearson Correlation Coefficient
    corr_matrix = np.corrcoef(y_true, y_pred)
    r = corr_matrix[0, 1]

    # R2 Score
    r2_score = 1 - np.sum((np.array(y_true) - np.array(y_pred)) ** 2) / np.sum((np.array(y_true) - np.mean(np.array(y_true))) ** 2)

    # RMSE in pK unit
    min=0
    max=16
    true_labels_unscaled = torch.tensor(y_true) * (max - min) + min
    predictions_unscaled = torch.tensor(y_pred) * (max - min) + min
    criter = RMSELoss()
    rmse = criter(predictions_unscaled, true_labels_unscaled)

    return eval_loss, r, rmse, r2_score, y_true, y_pred
#-------------------------------------------------------------------------------------------------------------------------------


def plot_predictions(train_y_true, train_y_pred, val_y_true, val_y_pred, title):
    
    axislim = 1.1
    fig = plt.figure(figsize=(8, 8))  # Set the figure size as needed

    plt.scatter(train_y_true, train_y_pred, alpha=0.5, c='blue', label='Training Data')
    plt.scatter(val_y_true, val_y_pred, alpha=0.5, c='red', label='Validation Data')

    plt.plot([min(train_y_true + val_y_true), axislim], [min(train_y_true + val_y_true), axislim], color='red', linestyle='--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.ylim(-0.1, axislim)
    plt.xlim(-0.1, axislim)
    plt.axhline(0, color='grey', linestyle='--')
    plt.axvline(0, color='grey', linestyle='--')
    plt.title(title)
    
    # Adding manual legend items for colors
    legend_elements = []
    legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=8, label='Training Dataset'))
    legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=8, label='Validation Dataset'))

    plt.legend(handles=legend_elements, loc='upper left')
    return fig


def main():
    args = parse_args()

    # Training Parameters
    loss_function = args.loss_func
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    optim = args.optim
    batch_size = args.batch_size
    dropout_prob = args.dropout
    conv_dropout_prob = args.conv_dropout
    n_folds = args.n_folds
    fold_to_train = args.fold_to_train

    # Architecture and run settings
    model_arch = args.model
    dataset_path = args.dataset_path
    run_name = f'{args.run_name}_f{fold_to_train}' 
    
    # Set random seeds
    torch.manual_seed(args.random_seed)
    random_seed = args.random_seed
    
    # If no save directory is provided, save in the run_name directory
    save_dir = args.save_dir
    if not args.save_dir: 
        save_dir = f'{run_name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f'Saving Directory generated')

    # Weights and Biases Tracking
    wandb_tracking = args.wandb
    project_name = args.project_name
    wandb_dir = save_dir
    
    # Early Stopping
    early_stopping = args.early_stopping
    early_stop_patience = args.early_stop_patience

    if early_stopping:
        early_stopper = EarlyStopper(patience=early_stop_patience)


    # Load Dataset - 5-fold cross-validation splitting in a stratified way
    #----------------------------------------------------------------------------------------------------
    dataset = torch.load(dataset_path)
    node_feat_dim = dataset[0].x.shape[1]
    edge_feat_dim = dataset[0].edge_attr.shape[1]

    labels = [graph.y.item() for graph in dataset]

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=n_folds, random_state=random_seed, shuffle=True)
    group_assignment = np.array( [round(lab) for lab in labels] )

    train_indices = []
    val_indices = []
    for i, (train_index, val_index) in enumerate(skf.split(np.zeros(len(dataset)), group_assignment)):
        val_indices.append(val_index.tolist())
        train_indices.append(train_index.tolist())

    # Select the fold that should be used for the training
    train_idx = train_indices[fold_to_train]
    val_idx = val_indices[fold_to_train]

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    # Save split dictionary to json at save dir (if the dataset contains the key "id")
    if 'id' in train_dataset[0].keys():
        split = {}
        split['validation'] = [grph['id'] for grph in val_dataset]
        split['train'] = [grph['id'] for grph in train_dataset]
        with open(f'{save_dir}/train_val_split.json', 'w', encoding='utf-8') as json_file:
            json.dump(split, json_file, ensure_ascii=False, indent=4)

    print(f'Length Training Dataset: {len(train_dataset)}')
    print(f'Length Validation Dataset: {len(val_dataset)}')
    print(f'Example Graph: {train_dataset[0]}')

    train_loader = DataLoader(dataset = train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True)
    eval_loader_train = DataLoader(dataset = train_dataset, batch_size=512, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True)
    eval_loader_val = DataLoader(dataset = val_dataset, batch_size=512, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True)
    #----------------------------------------------------------------------------------------------------



    # Initialize Model, Optimizer and Loss Function
    #-------------------------------------------------------------------------------------------------------------------------------
    # Device Settings
    num_threads = int(os.environ.get('OMP_NUM_THREADS', torch.get_num_threads()))
    torch.set_num_threads(num_threads)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Initialize the model and optimizer
    model_class = getattr(sys.modules[__name__], args.model)
    Model = model_class(dropout_prob=dropout_prob, in_channels=node_feat_dim, edge_dim=edge_feat_dim, conv_dropout_prob=conv_dropout_prob).to(device)
    Model = Model.float()
    torch.save(Model, f'{save_dir}/model_configuration.pt')

    parameters = count_parameters(Model)
    print(f'Model architecture {model_arch} with {parameters} parameters')

    if optim == 'Adam': optimizer = torch.optim.Adam(list(Model.parameters()),lr=learning_rate, weight_decay=weight_decay)
    elif optim == 'Adagrad': optimizer = torch.optim.Adagrad(Model.parameters(), learning_rate, weight_decay=weight_decay)
    elif optim == 'SGD': optimizer = torch.optim.SGD(Model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    #-------------------------------------------------------------------------------------------------------------------------------



    # DEFINE LOSS FUNCTION ['MSE', 'wMSE', 'L1', 'Huber']
    #----------------------------------------------------------------------------------------------------
    if loss_function == 'Huber':
        criterion = torch.nn.HuberLoss(reduction='mean', delta=1.0)
        print(f'Loss Function: Huber Loss')
    elif loss_function == 'L1':
        criterion = torch.nn.L1Loss(size_average=None, reduce=None, reduction='mean')
        print(f'Loss Function: L1 Loss')
    elif loss_function == 'wMSE':
        criterion = wMSELoss()
        print(f'Loss Function: wMSE Loss')
    elif loss_function == 'RMSE':
        criterion = RMSELoss()
        print(f'Loss Function: RMSE Loss')
    else: 
        criterion = torch.nn.MSELoss()
        print(f'Loss Function: MSE Loss')
    #----------------------------------------------------------------------------------------------------



    # Initialize WandB tracking with config dictionary
    #-----------------------------------------------------------------------------------
    if wandb_tracking:
        config = {
            "Learning Rate": learning_rate,
            "Weight Decay": weight_decay,
            "Architecture": model_arch,
            "Epochs": num_epochs,
            "Optimizer": optim,
            "Early Stopping": early_stopping,
            "Early Stopping Patience": args.early_stop_patience,
            "Batch Size": batch_size,
            "Splitting Random Seed":random_seed,
            "Dropout Probability": dropout_prob,
            "Dropout Prob Convolutional Layers":conv_dropout_prob,
            "Number of Parameters": parameters,
            "Device": torch.cuda.get_device_name()
        }
        wandb.login()
        wandb.init(project=project_name, name = run_name, config=config, dir=wandb_dir)
        

    print(f'Model Architecture {model_arch} - Fold {fold_to_train} ({run_name})')
    print(f'Number of Parameters: {parameters}')
    print(f'Learning Rate: {learning_rate}')
    print(f'Weight Decay: {weight_decay}')
    print(f'Batch Size: {batch_size}')
    print(f'Loss Function: {loss_function}')    
    print(f'Number of Epochs: {num_epochs}')
    print(f'Model Training Output ({run_name})')


    # Training and Validation Set Performance BEFORE Training
    #-------------------------------------------------------------------------------------------------------------------------------
    epoch = 0
    train_loss, train_r, train_rmse, train_r2, train_y_true, train_y_pred = evaluate(Model, eval_loader_train, criterion, device)
    val_loss, val_r, val_rmse, val_r2, val_y_true, val_y_pred = evaluate(Model, eval_loader_val, criterion, device)

    log_string = f'Before Train: Train Loss: {train_loss:6.3f}|  Pearson:{train_r:6.3f}|  R2:{train_r2:6.3f}|  RMSE:{train_rmse:6.3f}|  -- Val Loss: {val_loss:6.3f}|  Pearson:{val_r:6.3f}|  R2:{val_r2:6.3f}|  RMSE:{val_rmse:6.3f}| '
    print(log_string)

    if wandb_tracking:
        wandb.log({
                "Epoch": epoch,
                "Learning Rate": optimizer.param_groups[0]['lr'],
                "Training Loss":train_loss,
                "Training Pearson Correlation": train_r,
                "Training RMSE": train_rmse,
                "Training R2": train_r2,
                "Validation Loss":val_loss,
                "Validation R2": val_r2,
                "Validation Pearson Correlation": val_r,
                "Validation RMSE": val_rmse
                })

    best_epoch = val_rmse
    best_metrics = {'val': (val_loss, val_r, val_rmse, val_r2, val_y_true, val_y_pred),
                    'train': (train_loss, train_r, train_rmse, train_r2, train_y_true, train_y_pred)}



    # Training Loop
    #-------------------------------------------------------------------------------------------------------------------------------
    plotted = []
    tic = time.time()
    last_saved_epoch = 0
    early_stop = False

    for epoch in range(epoch+1, num_epochs+1):

        train_loss, train_r, train_rmse, train_r2, train_y_true, train_y_pred = train(Model, train_loader, criterion, optimizer, device)
        val_loss, val_r, val_rmse, val_r2, val_y_true, val_y_pred = evaluate(Model, eval_loader_val, criterion, device)

        log_string = f'Epoch {epoch:05d}:  Train Loss: {train_loss:6.3f}|  Pearson:{train_r:6.3f}|  R2:{train_r2:6.3f}|  RMSE:{train_rmse:6.3f}|  -- Val Loss: {val_loss:6.3f}|  Pearson:{val_r:6.3f}|  R2:{val_r2:6.3f}|  RMSE:{val_rmse:6.3f}| '

        if wandb_tracking:
            wandb.log({
                    "Epoch": epoch,
                    "Learning Rate": optimizer.param_groups[0]['lr'],
                    "Training Loss":train_loss,
                    "Training Pearson Correlation": train_r,
                    "Training RMSE": train_rmse,
                    "Training R2": train_r2,
                    "Validation Loss":val_loss,
                    "Validation R2": val_r2,
                    "Validation Pearson Correlation": val_r,
                    "Validation RMSE": val_rmse
                    })
            
        # If the previous best val_rmse is beaten, save the model and update the best metrics dict
        if val_rmse < best_epoch: 
            torch.save(Model.state_dict(), f'{save_dir}/{run_name}_best_stdict.pt')
            log_string += ' Saved'
            last_saved_epoch = epoch

            best_epoch = val_rmse
            best_metrics['val'] = (val_loss, val_r, val_rmse, val_r2, val_y_true, val_y_pred)
            best_metrics['train'] = (train_loss, train_r, train_rmse, train_r2, train_y_true, train_y_pred)

        print(log_string, flush=True)

        if early_stopping: 
            early_stop = early_stopper.early_stop(val_rmse)


        # After regular intervals, plot the predictions of the current and the best model
        if epoch % 20 == 0 or epoch == num_epochs or early_stop:

            # If there has been a new best epoch in the last interval of epochs, plot the predictions of this model      
            if last_saved_epoch not in plotted:
                
                # Load the current best metrics from dict
                val_loss, val_r, val_rmse, val_r2, val_y_true, val_y_pred = best_metrics['val']
                train_loss, train_r, train_rmse, train_r2, train_y_true, train_y_pred = best_metrics['train']

                # Plot the predictions and the residuals plot
                title = f"{run_name}: Epoch {last_saved_epoch}\nTrain R = {train_r:.3f}, Validation R = {val_r:.3f}\nTrain RMSE = {train_rmse:.3f}, Validation RMSE = {val_rmse:.3f}"
                best_predictions = plot_predictions(train_y_true, train_y_pred, val_y_true, val_y_pred, title)
                                        
                best_predictions.savefig(f'{save_dir}/train_predictions.png')
                plotted.append(last_saved_epoch)

            plt.close('all')
                
        # Is it time for early stopping?
        if early_stop: 
            print("Early stopping triggered. Ending training.")
            break

    toc = time.time()
    training_time = (toc-tic)/60
    print(f"Time for Training: {training_time:5.1f} minutes - ({(training_time/num_epochs):5.2f} minutes/epoch)")
    if wandb_tracking: 
        wandb.finish()


if __name__ == "__main__":
    main()