from model.ml import *
from torch.optim.lr_scheduler import StepLR
import torch
Y_SCALE_FACTOR=2000.0


def train_ml():
	# Load data
    print('Load data')
    datadir='data/qm7.mat'
    data = process_data(datadir)
	# Configure
    n_folds = 5
    mae=[]
    for val_fold in tqdm(range(n_folds)):
        X_train,y_train,X_val,y_val=train_val_split(data,val_fold)
        loss=svr(X_train,y_train,X_val,y_val)
        mae.append(loss)
        print("Fold {:f} - MAE: {:f}".format(val_fold,loss))

    print('Final MAE: ',mae)

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()

def train_mlp():
	# Load data
    print('Load data')
    datadir='data/qm7.mat'
    data = process_data(datadir)

    # Init model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_folds = 5
    # For fold results
    results = {}
    
    # Set fixed random number seed
    torch.manual_seed(42)

    for val_fold in range(n_folds):
        print('Fold ',val_fold)
        X_train,y_train,X_val,y_val=train_val_split(data,val_fold)

        # num_iterations = 1000000
        num_iterations = 50000
        mb = 128  # mini-batch size
        eval_step=100

        input_size = X_train.shape[1]  # Get the number of features from X_train
        hidden_sizes = [400, 100]
        output_size = 1

        # Init the neural network
        mlp = MLP(input_size, hidden_sizes, output_size).to(device)
        mlp.apply(reset_weights)

        # Initialize optimizer
        optimizer = torch.optim.AdamW(mlp.parameters(), lr=0.001, weight_decay=0.0001)
        criterion = nn.L1Loss() # MAE

        # Initialize the step scheduler
        scheduler = StepLR(optimizer, step_size=500, gamma=0.1)

        best_mae = float('inf')
        best_model = None

        train_mae_plot=[]
        val_mae_plot=[]


        for epoch in tqdm(range(1, num_iterations + 1)):
            mlp.train()
            optimizer.zero_grad()
            train_mae=[]
            for batch_start in range(0, len(X_train), mb):
                batch_end = max(batch_start + mb,len(X_train)-1)
                X_batch = torch.Tensor(X_train[batch_start:batch_end]).to(device)
                y_batch = torch.Tensor(y_train[batch_start:batch_end]).unsqueeze(1).to(device)
            
                output = mlp(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                scheduler.step()  # Update learning rate
                train_mae.append(float(loss*Y_SCALE_FACTOR))
                
            train_mae=np.mean(train_mae)
            train_mae_plot.append(train_mae)
            
            
            
        
            
            if epoch % eval_step == 0:
                mlp.eval()
                val_mae=[]
                for batch_start in range(0, len(X_val), mb):
                    batch_end = max(batch_start + mb,len(X_val)-1)
                    X_batch = torch.Tensor(X_val[batch_start:batch_end]).to(device)
                    y_batch = torch.Tensor(y_val[batch_start:batch_end]).unsqueeze(1).to(device)
                
                    output = mlp(X_batch)
                    val_mae.append(float(criterion(output, y_batch)*Y_SCALE_FACTOR))
                val_mae=np.mean(val_mae)
                val_mae_plot.append(val_mae)
                print(train_mae_plot,val_mae_plot)
                plot_loss(train_mae_plot,val_mae_plot, f"model/plot/{val_fold}.png")
                torch.save(mlp.state_dict(), f'model/weight/mlp_model_last_{val_fold}.pt')

                # Check if current model has the best MAE
                if val_mae < best_mae:
                    best_mae = val_mae
                    best_model = mlp.state_dict()
                    torch.save(best_model, f'model/weight/mlp_model_best_{val_fold}.pt')


def plot_loss(mae_scores_train, mae_scores_val, save_dir):
    # Generate the iteration numbers based on the number of MAE scores
    iterations = np.arange(1,len(mae_scores_train)+ 1)

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the training MAE curve
    axes[0].plot(iterations, mae_scores_train)
    axes[0].set_xlabel('Iterations')
    axes[0].set_ylabel('Training MAE')
    axes[0].set_title('Training Loss Curve')
    axes[0].grid(True)

    iterations = np.arange(1,len(mae_scores_val)+ 1)

    # Plot the validation MAE curve
    axes[1].plot(iterations, mae_scores_val)
    axes[1].set_xlabel('Iterations')
    axes[1].set_ylabel('Validation MAE')
    axes[1].set_title('Validation Loss Curve')
    axes[1].grid(True)

    # Adjust the layout to prevent overlapping of subplots
    fig.tight_layout()

    # Save the plot to the specified directory
    plt.savefig(save_dir)
    plt.close()


if __name__=="__main__":
    train_mlp()