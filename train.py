import os

import torch
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from tqdm import tqdm

from model.gnn import *
from model.ml import *
from utils import *

Y_SCALE_FACTOR = 2000.0


def train_ml():
    # Load data
    print("Load data")
    datadir = "data/qm7.mat"
    data = process_data_sorted(datadir)
    # Configure
    n_folds = 5
    mae = []
    for val_fold in tqdm(range(n_folds)):
        X_train, y_train, X_val, y_val = train_val_split(data, val_fold, "sorted")
        loss = kernel_ridge(X_train, y_train, X_val, y_val)
        mae.append(loss)
        print("Fold {:f} - MAE: {:f}".format(val_fold, loss))

    print("Final MAE: ", mae)


def train_mlp_sorted():
    # Load data
    print("Process data")
    datadir = "data/qm7.mat"
    data = process_data_sorted(datadir)

    # Init model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_folds = 5
    # For fold results
    results = {}

    # Output folder
    output_path = "exps"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    folders = os.listdir(output_path)
    new_id = 0
    if len(folders) > 0:
        for folder in folders:
            if not folder.startswith("mlp_exp_"):
                continue
            new_id = max(new_id, int(folder.split("mlp_exp_")[-1]))
        new_id += 1
    output_path = os.path.join(output_path, f"mlp_exp_{new_id}")
    os.makedirs(output_path)
    weights_path = os.path.join(output_path, "weights")
    os.mkdir(weights_path)
    visualize_path = os.path.join(output_path, "visualize")
    os.mkdir(visualize_path)
    output_path = output_path + "/"
    visualize_path = visualize_path + "/"
    weights_path = weights_path + "/"

    # Set fixed random number seed
    torch.manual_seed(42)

    # Create the log file
    log_file_path = os.path.join(output_path, "mae_log.txt")
    log_file = open(log_file_path, "w")

    for val_fold in range(n_folds):
        print("Fold ", val_fold)
        X_train, y_train, X_val, y_val = train_val_split(data, val_fold, "sorted")

        # num_iterations = 1000000
        # num_iterations = 50000
        num_iterations = 300
        mb = 256  # mini-batch size
        eval_step = 10

        input_size = X_train.shape[1]  # Get the number of features from X_train
        hidden_sizes = [400, 100]
        output_size = 1

        # Init the neural network
        mlp = MLP(input_size, hidden_sizes, output_size).to(device)
        mlp.apply(reset_weights)

        # Initialize optimizer
        optimizer = torch.optim.AdamW(mlp.parameters(), lr=0.001, weight_decay=0.01)
        criterion = nn.L1Loss()  # MAE

        # Initialize the step scheduler
        # scheduler = StepLR(optimizer, step_size=500, gamma=0.3333)
        milestones = [500, 1000, 2500, 12500]
        gamma = 0.3333  # No change in learning rate initially
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

        best_mae = float("inf")
        best_model = None

        train_mae_plot = []
        val_mae_plot = []

        for epoch in tqdm(range(1, num_iterations + 1)):
            mlp.train()

            train_mae = []
            for batch_start in range(0, len(X_train), mb):
                optimizer.zero_grad()
                batch_end = min(batch_start + mb, len(X_train) - 1)
                X_batch = torch.Tensor(X_train[batch_start:batch_end]).to(device)
                y_batch = (
                    torch.Tensor(y_train[batch_start:batch_end]).unsqueeze(1).to(device)
                )

                output = mlp(X_batch)
                loss = criterion(output * Y_SCALE_FACTOR, y_batch * Y_SCALE_FACTOR)
                loss.backward()
                optimizer.step()
                scheduler.step()  # Update learning rate
                train_mae.append(float(loss))

            train_mae = np.mean(train_mae)
            train_mae_plot.append(train_mae)

            if epoch % eval_step == 0:
                mlp.eval()
                val_mae = []
                for batch_start in range(0, len(X_val), mb):
                    batch_end = min(batch_start + mb, len(X_val) - 1)
                    X_batch = torch.Tensor(X_val[batch_start:batch_end]).to(device)
                    y_batch = (
                        torch.Tensor(y_val[batch_start:batch_end])
                        .unsqueeze(1)
                        .to(device)
                    )

                    output = mlp(X_batch)
                    val_mae.append(
                        float(
                            criterion(output * Y_SCALE_FACTOR, y_batch * Y_SCALE_FACTOR)
                        )
                    )
                val_mae = np.mean(val_mae)
                val_mae_plot.append(val_mae)
                plot_loss(
                    train_mae_plot,
                    val_mae_plot,
                    os.path.join(visualize_path, f"fold_{val_fold}.png"),
                )
                torch.save(
                    mlp.state_dict(),
                    os.path.join(weights_path, f"mlp_model_last_{val_fold}.pt"),
                )

                # Check if current model has the best MAE
                if val_mae < best_mae:
                    best_mae = val_mae
                    best_model = mlp.state_dict()
                    torch.save(
                        best_model,
                        os.path.join(weights_path, f"mlp_model_best_{val_fold}.pt"),
                    )
        print("Fold {:f} - MAE: {:f}".format(val_fold, float(np.mean(val_mae_plot))))
        # Write the MAE for the current fold to the log file
        log_file.write(
            "Fold {:f} - MAE| Train - {:f} | Val - {:f}\n".format(
                val_fold, float(np.mean(val_mae_plot)), float(np.mean(train_mae_plot))
            )
        )
        print("Fold {:f} - MAE: {:f}".format(val_fold, float(np.mean(val_mae_plot))))
    log_file.close()
    print("Training complete. MAE log file saved at: ", log_file_path)


def train_mlp_random():
    # Load data
    print("Process data")
    datadir = "data/qm7.mat"
    data = process_data_random(datadir)

    # Init model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_folds = 5
    # For fold results
    results = {}

    # Output folder
    output_path = "exps"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    folders = os.listdir(output_path)
    new_id = 0
    if len(folders) > 0:
        for folder in folders:
            if not folder.startswith("mlp_exp_"):
                continue
            new_id = max(new_id, int(folder.split("mlp_exp_")[-1]))
        new_id += 1
    output_path = os.path.join(output_path, f"mlp_exp_{new_id}")
    os.makedirs(output_path)
    weights_path = os.path.join(output_path, "weights")
    os.mkdir(weights_path)
    visualize_path = os.path.join(output_path, "visualize")
    os.mkdir(visualize_path)
    output_path = output_path + "/"
    visualize_path = visualize_path + "/"
    weights_path = weights_path + "/"

    # Set fixed random number seed
    torch.manual_seed(42)

    # Hyperparameter
    num_iterations = 50000
    mb = 256  # mini-batch size
    eval_step = 10
    milestones = [500, 1000, 2500, 12500]
    gamma = 0.3333  # No change in learning rate initially

    input_size = None  # Get the number of features from X_train
    hidden_sizes = [400, 200, 100]
    output_size = 1

    # Create the log file
    log_file_path = os.path.join(output_path, "mae_log.txt")
    log_file = open(log_file_path, "w")
    log_file.write(f"Epoch: {num_iterations}\n")
    log_file.write(f"Bach size: {mb}\n")
    log_file.write(f"Evaluation step: {eval_step}\n")
    log_file.write(f"StepLR milestones: {str(milestones)}\n")
    log_file.write(f"StepLR gamma: {gamma}\n")
    log_file.write("\n")
    log_file.write(f"MLP Layer\n")
    log_file.write(f"Input size: X_train.shape[1]\n")
    log_file.write(f"Hidden sizes: {hidden_sizes}\n")
    log_file.write(f"Output size: {output_size}\n")
    log_file.write("\n")

    for val_fold in range(n_folds):
        print("Fold ", val_fold)
        X_train, y_train, X_val, y_val = train_val_split(data, val_fold, "random")

        input_size = X_train.shape[1]  # Get the number of features from X_train
        hidden_sizes = [400, 100]
        output_size = 1

        # Init the neural network
        mlp = MLP(input_size, hidden_sizes, output_size).to(device)
        mlp.apply(reset_weights)

        # Initialize optimizer
        optimizer = torch.optim.AdamW(mlp.parameters(), lr=0.001)
        criterion = nn.L1Loss()  # MAE

        # Initialize the step scheduler
        # scheduler = StepLR(optimizer, step_size=500, gamma=0.3333)
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

        best_mae = float("inf")
        best_model = None

        train_mae_plot = []
        val_mae_plot = []

        for epoch in tqdm(range(1, num_iterations + 1)):
            mlp.train()

            train_mae = []
            for batch_start in range(0, len(X_train), mb):
                optimizer.zero_grad()
                batch_end = min(batch_start + mb, len(X_train) - 1)
                X_batch = torch.Tensor(X_train[batch_start:batch_end]).to(device)
                y_batch = (
                    torch.Tensor(y_train[batch_start:batch_end]).unsqueeze(1).to(device)
                )

                output = mlp(X_batch)
                loss = criterion(output * Y_SCALE_FACTOR, y_batch * Y_SCALE_FACTOR)
                loss.backward()
                optimizer.step()
                scheduler.step()  # Update learning rate
                train_mae.append(float(loss))

            train_mae = np.mean(train_mae)
            train_mae_plot.append(train_mae)

            if epoch % eval_step == 0:
                mlp.eval()
                val_mae = []
                for batch_start in range(0, len(X_val), mb):
                    batch_end = min(batch_start + mb, len(X_val) - 1)
                    X_batch = torch.Tensor(X_val[batch_start:batch_end]).to(device)
                    y_batch = (
                        torch.Tensor(y_val[batch_start:batch_end])
                        .unsqueeze(1)
                        .to(device)
                    )

                    output = mlp(X_batch)
                    val_mae.append(
                        float(
                            criterion(output * Y_SCALE_FACTOR, y_batch * Y_SCALE_FACTOR)
                        )
                    )
                val_mae = np.mean(val_mae)
                val_mae_plot.append(val_mae)
                plot_loss(
                    train_mae_plot,
                    val_mae_plot,
                    os.path.join(visualize_path, f"fold_{val_fold}.png"),
                )
                torch.save(
                    mlp.state_dict(),
                    os.path.join(weights_path, f"mlp_model_last_{val_fold}.pt"),
                )

                # Check if current model has the best MAE
                if val_mae < best_mae:
                    best_mae = val_mae
                    best_model = mlp.state_dict()
                    torch.save(
                        best_model,
                        os.path.join(weights_path, f"mlp_model_best_{val_fold}.pt"),
                    )
        print("Fold {:f} - MAE: {:f}".format(val_fold, float(best_mae)))
        # Write the MAE for the current fold to the log file
        log_file.write(
            "Fold {:f} - MAE| Train - {:f} | Val - {:f}\n".format(
                val_fold, float(train_mae), float(val_mae)
            )
        )
        print("Fold {:f} - MAE: {:f}".format(val_fold, float(best_mae)))
    log_file.close()
    print("Training complete. MAE log file saved at: ", log_file_path)


def train_gnn():
    # Set fixed random number seed
    torch.manual_seed(42)

    datadir = "data/qm7.mat"
    # Parameter
    mb = 100
    learning_rate = 1e-1
    num_iterations = 100
    eval_step = 5
    step_size = 30  # steplr
    gamma = 0.8  # steplr

    dataset, A_hat, D, folds, label = process_data_gnn(datadir)
    # Init model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_folds = 5
    # For fold results
    results = {}

    # Output folder
    output_path = "exps"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    folders = os.listdir(output_path)
    new_id = 0
    if len(folders) > 0:
        for folder in folders:
            if not folder.startswith("gcn_exp_"):
                continue
            new_id = max(new_id, int(folder.split("gcn_exp_")[-1]))
        new_id += 1
    output_path = os.path.join(output_path, f"gcn_exp_{new_id}")
    os.makedirs(output_path)
    weights_path = os.path.join(output_path, "weights")
    os.mkdir(weights_path)
    visualize_path = os.path.join(output_path, "visualize")
    os.mkdir(visualize_path)
    output_path = output_path + "/"
    visualize_path = visualize_path + "/"
    weights_path = weights_path + "/"

    # Create the log file
    log_file_path = os.path.join(output_path, "mae_log.txt")
    log_file = open(log_file_path, "w")
    log_file.write("\nHyperparameters:\n")
    log_file.write("Minibatch Size: {}\n".format(mb))
    log_file.write("Learning Rate: {}\n".format(learning_rate))
    log_file.write("Number of Iterations: {}\n".format(num_iterations))
    log_file.write("Evaluation Step: {}\n".format(eval_step))
    log_file.write("LR Step: {}\n".format(step_size))
    log_file.write("LR Gamma: {}\n".format(gamma))
    log_file.write("\n")

    for val_fold in range(n_folds):
        print("Fold ", val_fold)
        val_idx = folds[val_fold].flatten()
        train_idx = np.array(list(set(folds.flatten()) - set(val_idx)))

        X_train = dataset[train_idx]
        X_val = dataset[val_idx]
        y_train = label[0][train_idx]
        y_val = label[0][val_idx]
        A_hat_train = A_hat[train_idx]
        A_hat_val = A_hat[val_idx]
        D_train = D[train_idx]
        D_val = D[val_idx]

        gcn = GCN(mb, device)
        # Initialize optimizer
        optimizer = torch.optim.AdamW([gcn.W1, gcn.W2, gcn.W3], lr=learning_rate)
        criterion = nn.L1Loss()  # MAE

        # Initialize the step scheduler
        # scheduler = StepLR(optimizer, step_size=500, gamma=0.3333)

        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

        # Initialize the step scheduler
        best_mae = float("inf")
        best_model = None

        train_mae_plot = []
        val_mae_plot = []

        for epoch in tqdm(range(1, num_iterations + 1)):
            gcn.train()

            train_mae = []
            for batch_start in range(0, len(X_train), mb):
                if batch_start + mb > len(X_train) - 1:
                    break
                batch_end = batch_start + mb
                X_batch = torch.Tensor(X_train[batch_start:batch_end]).to(device)
                y_batch = (
                    torch.Tensor(y_train[batch_start:batch_end]).view(mb, 1).to(device)
                )
                A_hat_batch = torch.Tensor(A_hat_train[batch_start:batch_end]).to(
                    device
                )
                D_batch = torch.Tensor(D_train[batch_start:batch_end]).to(device)

                output = gcn(X_batch, A_hat_batch, D_batch)
                optimizer.zero_grad()
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                scheduler.step()
                train_mae.append(float(loss))

            train_mae = np.mean(train_mae)
            train_mae_plot.append(train_mae)

            if epoch % eval_step == 0:
                gcn.eval()
                val_mae = []
                for val_batch_start in range(0, len(X_val), mb):
                    with torch.no_grad():
                        if val_batch_start + mb > len(X_val) - 1:
                            break
                        val_batch_end = val_batch_start + mb
                        val_X_batch = torch.Tensor(
                            X_val[val_batch_start:val_batch_end]
                        ).to(device)
                        val_y_batch = (
                            torch.Tensor(y_val[val_batch_start:val_batch_end])
                            .view(mb, 1)
                            .to(device)
                        )
                        val_A_hat_batch = torch.Tensor(
                            A_hat_val[val_batch_start:val_batch_end]
                        ).to(device)
                        val_D_batch = torch.Tensor(
                            D_val[val_batch_start:val_batch_end]
                        ).to(device)
                        val_output = gcn(val_X_batch, val_A_hat_batch, val_D_batch)
                        val_mae.append(float(criterion(val_output, val_y_batch)))
                val_mae = np.mean(val_mae)
                val_mae_plot.append(val_mae)
                plot_loss(
                    train_mae_plot,
                    val_mae_plot,
                    os.path.join(visualize_path, f"fold_{val_fold}.png"),
                )
                torch.save(
                    gcn.state_dict(),
                    os.path.join(weights_path, f"gcn_model_last_{val_fold}.pt"),
                )

                # Check if current model has the best MAE
                if val_mae < best_mae:
                    best_mae = val_mae
                    best_model = gcn.state_dict()
                    torch.save(
                        best_model,
                        os.path.join(weights_path, f"gcn_model_best_{val_fold}.pt"),
                    )
        # Write the MAE for the current fold to the log file
        log_file.write(
            "Fold {:f} - MAE| Train - {:f} | Val - {:f}\n".format(
                val_fold, float(np.min(train_mae_plot)), float(best_mae)
            )
        )
        print("Fold {:f} - MAE: {:f}".format(val_fold, float(best_mae)))
    log_file.close()
    print("Training complete. MAE log file saved at: ", log_file_path)


if __name__ == "__main__":
    train_mlp_random()
