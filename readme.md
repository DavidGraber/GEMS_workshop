# Graph Neural Network Training and Testing Instructions

This guide will walk you through the steps required to train and test a Graph Neural Network (GNN) for predicting protein-ligand binding affinities using the GEMS architecture.

## Requirements and Environment Setup

Before starting, create a new environment with **Python 3.10** and install the following dependencies:

```bash
# Install basic requirements
pip install matplotlib scikit-learn numpy==1.26.4
pip install torch==2.0.1
pip install torch_geometric

# To visualize the input graphs in interactive 3D plot
pip install plotly 
```

### With GPU accelaration:
Visit the pytorch geometric documentation page and run the installation command appropriate for your system:
https://pytorch-geometric.readthedocs.io/en/2.4.0/install/installation.html
```bash
# Example for Linux with Torch 2.0.1 and Cuda 11.7 
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
```


## Download Dataset

Download the training and testing dataset using the command below

**Linux**
```bash
wget https://g-95befe.765b9d.09d9.data.globus.org/GEMS_PLINDER_OOD_datasets.tar.gz
```
**Mac**
```bash
curl https://g-95befe.765b9d.09d9.data.globus.org/GEMS_PLINDER_OOD_datasets.tar.gz --output GEMS_PLINDER_OOD_datasets.tar.gz
```
**Windows**
```bash
curl https://g-95befe.765b9d.09d9.data.globus.org/GEMS_PLINDER_OOD_datasets.tar.gz --output GEMS_PLINDER_OOD_datasets.tar.gz
```

### Extract the datasets and inspect
```bash
tar -xzvf GEMS_PLINDER_OOD_datasets.tar.gz
```
This will extract the datasets and place them into a folder called "datasets". Each dataset (.pt file) contains a number of pytorch geometric Data() objects representing featurized graphs. Each object contains node features `x`, an `edge_index`, edge features (`edge_attr`), the datapoint label `y` and the coordinates of the graph nodes `pos` (for visualization only).
```bash
# Example Data() object representing a graph
Data(x=[62, 828], edge_index=[2, 478], edge_attr=[478, 20], y=0.3700000047683716, pos=[62, 3], n_nodes=[3], id='1a4h')
```
- **Feature matrix** `x`: Node features for all nodes in the graph
- **Edge index** `edge_index`: A tensor saving pairwise connections (which nodes are connected?)
- **Edge features** `edge_attr`: The features saved in the edges of the graph
- **Label** `y`: The binding affinity of this graph, scaled between 0 and 1


## Visualize Graphs in 3D
In order to visualize an example graph in an interative 3D-plot, make sure you have plotly installed (see above) and run the `visualize_graph.py` script with the path to a dataset (.pt file) as input:
```bash
python visualize_graph.py --dataset_path datasets/dataset_1nvq_ood_test.pt
```



## Train the GNN

The training script (`train.py`) trains a graph neural network on a provided dataset. It supports various command line arguments to customize the training process. Here's how to train your model:

### Basic Training Command

```bash
python train.py --dataset_path datasets/dataset_train.pt --run_name my_first_model
```

This will train the model with default parameters and save the results in a folder named `my_first_model_f0`.

### Advanced Training Options

You can customize your training with the following options:

```bash
python train.py \
  --dataset_path datasets/dataset_train.pt \
  --run_name my_advanced_model \
  --model GEMS18d \
  --n_folds 5 \
  --fold_to_train 0 \
  --num_epochs 300 \
  --batch_size 128 \
  --learning_rate 0.001 \
  --weight_decay 0.0005 \
  --dropout 0.3 \
  --conv_dropout 0.1 \
  --loss_func RMSE \
  --optim Adam \
  --early_stopping True \
  --early_stop_patience 15 \
  --random_seed 42 \
  --save_dir results/my_model
```

### Key Training Parameters

- **Required Parameters**:
  - `--dataset_path`: Path to the dataset (.pt file)
  - `--run_name`: Name for the training run (used for saving results)

- **Model Architecture and Saving**:
  - `--model`: Model architecture to use (default: "GEMS18d")
  - `--save_dir`: Directory to save results (default: run_name/)

- **Training-Validation Split**:
  - `--n_folds`: Number of folds for cross-validation (default: 5)
  - `--fold_to_train`: Which fold to use for training (default: 0)
  - `--random_seed`: Random seed for reproducibility (default: 0)

- **Training Hyperparameters**:
  - `--num_epochs`: Maximum number of training epochs (default: 500)
  - `--batch_size`: Batch size for training (default: 256)
  - `--learning_rate`: Learning rate (default: 0.001)
  - `--weight_decay`: Weight decay for regularization (default: 0.001)
  - `--dropout`: Dropout probability (default: 0.5)
  - `--conv_dropout`: Dropout probability for conv layers (default: 0)

- **Loss Function and Optimizer**:
  - `--loss_func`: Loss function ['MSE', 'RMSE', 'wMSE', 'L1', 'Huber'] (default: 'RMSE')
  - `--optim`: Optimizer ['Adam', 'Adagrad', 'SGD'] (default: 'SGD')

- **Early Stopping**:
  - `--early_stopping`: Whether to use early stopping (default: True)
  - `--early_stop_patience`: Patience for early stopping (default: 20)

### Training Output

The training script will output:
- **Training and validation metrics** for each epoch
- **Model checkpoint** for the best performing model saved as `{run_name}_f{fold}_best_stdict.pt`
- **True vs. predicted** scatterplot for the current best model
- **Learning Curves:** A line plot showing the development of the training and validation loss


## Test Your Model

After training, you can evaluate your model on test datasets using the `test.py` script:

### Basic Testing Command
This will test your model on the commonly used casf2016 benchmark dataset.
```bash
python test.py --stdicts my_model_f0/my_model_f0_best_stdict.pt --dataset_path datasets/dataset_casf2016.pt
```
 
### Testing on out-of-distribution datasets
You can test your model on any of the challenging out-of-distribution datasets and see how well it performs on protein-ligand complexes it has never seen before.
```bash
python test.py --stdicts my_model_f0/my_model_f0_best_stdict.pt --dataset_path datasets/dataset_1nvq_ood_test.pt
```

### Testing with an Ensemble of Models

You can test with an ensemble of models by providing multiple state dictionaries:

```bash
python test.py \
  --stdicts my_model_f0/my_model_f0_best_stdict.pt,my_model_f1/my_model_f1_best_stdict.pt,my_model_f2/my_model_f2_best_stdict.pt \
  --dataset_path datasets/dataset_casf2016.pt \
  --save_path results/ensemble_test
```

### Key Testing Parameters

- **Required Parameters**:
  - `--stdicts`: Comma-separated paths to model state dictionaries
  - `--dataset_path`: Path to the test dataset (.pt file)

- **Optional Parameters**:
  - `--model_arch`: Model architecture (default: "GEMS18d")
  - `--save_path`: Directory to save test results (default: same dir as stdict)

### Testing Output

The testing script will output:
- Performance metrics (RMSE, Pearson correlation, R2)
- A scatter plot of true vs. predicted values
- A JSON file with predictions for each molecule in the test set


## Advanced Usage and Tips

1. **Cross-Validation**: Train multiple folds and average results:
   ```bash
   for i in {0..4}; do
     python train.py --dataset_path datasets/dataset_train.pt --run_name cv_model --fold_to_train $i
   done
   ```

2. **Hyperparameter Tuning**: Adjust learning rate, batch size, and dropout for optimal performance:
   ```bash
   python train.py --dataset_path datasets/dataset_train.pt --run_name tuning_lr --learning_rate 0.0005
   ```

3. **Out-of-Distribution Testing**: Test your model on different protein families:
   ```bash
   python test.py --stdicts my_model_f0/my_model_f0_best_stdict.pt --dataset_path datasets/dataset_1nvq_ood_test.pt
   ```

4. **Ensemble Testing**: Combine multiple trained models for improved prediction accuracy.

## Common Issues and Solutions

- **Out of Memory Errors**: Reduce batch size or model complexity
- **Poor Validation Performance**: Try different learning rates, increase dropout, or use early stopping
- **Slow Training**: Use GPU acceleration if available, optimize batch size
- **Overfitting**: Increase dropout, add weight decay, use early stopping

## Additional Resources

- PyTorch Geometric Documentation: https://pytorch-geometric.readthedocs.io/
- Graph Neural Networks Overview: https://distill.pub/2021/gnn-intro/
- GEMS Model Paper: https://www.nature.com/articles/s42256-025-01124-5
