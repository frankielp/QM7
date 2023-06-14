# Quantum Mechanical Calculation Estimation

This project demonstrates the training of a neural network model using PyTorch for a specific dataset.

## Description

The goal of this project is to train a neural network model on the QM7 dataset and evaluate its performance using mean absolute error (MAE) as the evaluation metric. The project includes the implementation of a multi-layer perceptron (MLP) model, data processing functions, and training procedures.

## Dataset

The QM7 dataset contains quantum mechanical calculations of molecular energies for a set of organic molecules. The dataset is stored in the 'data/qm7.mat' file.

## Dependencies

The project requires the following dependencies:

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- tqdm

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

4. Train the MLP model:

   ```
   python train.py
   ```

   The training script will train the MLP model on the QM7 dataset using a step-like learning rate schedule. The trained models and loss plots will be saved in the 'model/weight/' and 'model/plot/' directories, respectively.

## Results

The results of the training process will be displayed during the training and saved in the 'model/plot/' directory. The MAE scores for each fold and the final MAE score will be printed to the console.

## License

This project is licensed under the [MIT License](LICENSE).

```

You can customize the content based on your project's specific details and requirements.
