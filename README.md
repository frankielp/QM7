# Quantum Mechanical Calculation Estimation

This project demonstrates the training of a neural network model using PyTorch for a specific dataset.

## Description

The goal of this project is to train a neural network model on the QM7 dataset and evaluate its performance using mean absolute error (MAE) as the evaluation metric. The project includes the implementation of Machine Learning models, MLP, GCN, data processing functions, and training procedures.

## Dataset

The QM7 dataset contains quantum mechanical calculations of molecular energies for a set of organic molecules. The dataset is stored in the 'data/qm7.mat' file.

## Dependencies

The project requires the following dependencies:

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- tqdm

## Project Structure

The project consists of the following files and directories:

- `model/gnn.py`: Contains the GNN (Graph Neural Network) model implementation.
- `model/ml.py`: Contains the ML (Machine Learning) model implementations, including linear regression, Gaussian process regression, kernel ridge regression, and support vector regression.
- `utils.py`: Contains utility functions for data preprocessing and visualization.
- `data/qm7.mat`: The input dataset in MATLAB format.
- `exps/`: Output directory for experiment results, including weights and visualization plots.

## Usage

1. Clone the repository:

   ```
   git clone https://github.com/frankielp/QM7
   ```

2. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Prepare the data:

   - Place the QM7 dataset file ('qm7.mat') in the 'data/' directory.

4. Run the `train.py` script to train and evaluate the models.

   ```
   python train.py --model <model_name> --data <data_type>
   ```

   Replace `<model_name>` with the desired model: `linear_regression`, `gaussian_process_regression`, `kernel_ridge_regression`, `support_vector_regression`, or `mlp`.

   Replace `<data_type>` with the desired data type for the ML model: `eigenspectrum` (sorted data) or `random` (randomly shuffled data).

   Example:

   ```
   python train.py model=mlp data=eigenspectrum
   ```

   This command will train the MLP model on sorted eigenspectrum data.

5. The trained models and evaluation results will be saved in the `exps` directory. 

- The weights and loss plots will be saved in the 'model_exp_version/weight/' and 'model_exp_version/visualize/' directories, respectively.

- The experiment results, including the MAE for each fold and the training log, will be saved in the `exps/mae_log.txt` file.

## Notes

- Some models requires CUDA-compatible GPUs for faster training. If CUDA is not available, the model will be trained on the CPU.

- The MLP and GNN model training process may take a long time, depending on the size of the dataset and the number of iterations. The current configuration uses 5-fold cross-validation and 50000 iterations.

- The MAE (Mean Absolute Error) for each fold and the final MAE will be printed after the training process is complete.

## Acknowledgements

This project uses the [QM7](http://quantum-machine.org/datasets/) dataset. For more information about the dataset, refer to the original source.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more information.

