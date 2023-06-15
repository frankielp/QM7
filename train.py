import torch
from torch.optim.lr_scheduler import MultiStepLR, StepLR

from model.ml import *
from utils import plot_loss, process_data

Y_SCALE_FACTOR = 2000.0


def train_ml():
    # Load data
    print("Load data")
    datadir = "data/qm7.mat"
    data = process_data(datadir)
    # Configure
    n_folds = 5
    mae = []
    for val_fold in tqdm(range(n_folds)):
        X_train, y_train, X_val, y_val = train_val_split(data, val_fold)
        loss = kernel_ridge(X_train, y_train, X_val, y_val)
        mae.append(loss)
        print("Fold {:f} - MAE: {:f}".format(val_fold, loss))

    print("Final MAE: ", mae)


def train_mlp():
    # Load data
    print("Load data")
    datadir = "data/qm7.mat"
    data = process_data(datadir)

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
            if not folder.startswith("exp_"):
                continue
            new_id = max(new_id, int(folder.split("exp_")[-1]))
        new_id += 1
    output_path = os.path.join(output_path, f"exp_{new_id}")
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

    for val_fold in range(n_folds):
        print("Fold ", val_fold)
        X_train, y_train, X_val, y_val = train_val_split(data, val_fold)

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
            optimizer.zero_grad()
            train_mae = []
            for batch_start in range(0, len(X_train), mb):
                batch_end = max(batch_start + mb, len(X_train) - 1)
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
                    batch_end = max(batch_start + mb, len(X_val) - 1)
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


if __name__ == "__main__":
    train_mlp()
