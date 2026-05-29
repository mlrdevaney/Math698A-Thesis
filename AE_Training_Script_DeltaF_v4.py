print("Script Started...")
##########################################################################################################
# This PyTorch version replicates the TensorFlow-based autoencoder trainer from Math 689 Thesis work.
# It constructs an autoencoder with 1, 3, or 5 hidden layers,
# adds noise if desired, trains the model on velocity solution data, and saves logs and weights.
##########################################################################################################

import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from AE_DeltaF_Class_v4 import Autoencoder  # Assumes class name is Autoencoder
import Utilities  # Custom module for loading settings, data, and handling model folders


# Extract key training settings
import argparse

# ================================
# Command-line argument parser
# ================================
parser = argparse.ArgumentParser(description="Train Autoencoder on Boltzmann data")

parser.add_argument("--daily_count", type=int, default=0)
parser.add_argument("--truncation", type=int, default=0)
parser.add_argument("--threshold", type=float, default=1e-3)
parser.add_argument("--bias", type=int, default=0)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--earlystop", type=int, default=1)
parser.add_argument("--val_split", type=float, default=0.0)
parser.add_argument("--hidden_layers", type=int, default=3)
parser.add_argument("--code_len", type=int, default=4**3)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--min_delta", type=float, default=1e-4)
parser.add_argument("--patience", type=int, default=15)
parser.add_argument("--noise", type=float, default=0)
parser.add_argument("--lr", type=float, default=0.002)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--eps", type=float, default=1e-6)
parser.add_argument("--momdecay", type=float, default=4e-3)
parser.add_argument("--train_path", type=str, default="Data/Non-norm_BBBTsD_delta_f_data_MM_41_MT_0_CT_1.0.pkl")
parser.add_argument("--test_path", type=str, default="Data/11xs_delta_f_data_MM_41_MT_0_CT_1.0.pkl")

args = parser.parse_args()

# ================================
# Extract key training settings
# ================================
epochs_num = args.epochs
DC = args.daily_count
truncation = bool(args.truncation)
bias = bool(args.bias)
Early_Stopping = bool(args.earlystop)
hidden_layer_num = args.hidden_layers
code_len = args.code_len
batch_size = args.batch_size
min_delta = args.min_delta
patience = args.patience
noise = args.noise
lr = args.lr
dropout = args.dropout
eps = args.eps
train_path = args.train_path
test_path = args.test_path
val_fraction = args.val_split
momentum_decay = args.momdecay

# === Utility Callback ===
class SaveEpochInfo:
    def __init__(self, outfile, sol_data_train, solsize, batch_size):
        self.outfile = outfile
        self.outfile.write(f"training samples: {sol_data_train.shape[0]}, solution size: {solsize}, batch size: {batch_size}\n\n")
        self.outfile.write(f"epoch\ttrain_loss\tval_loss\ttest_loss\n")
        self.outfile.flush()

    def log(self, epoch, train_loss, test_loss, val_loss):
        infoStr = f"{epoch:03d}\t{train_loss:.10e}\t{val_loss:.10e}\t{test_loss:.10e}\n"
        self.outfile.write(infoStr)
        self.outfile.flush()

# Choose training device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading Training and Testing Data...")
# === Load training solution data ===
sol_data_train = Utilities.LoadPickleSolData(train_path) # shape (B=29249, MM**3) - 2d
solsize = sol_data_train.shape[1]  # Number of velocity space points = MM**3

# === Load separate test solution data ===
sol_data_test = Utilities.LoadPickleSolData(test_path) # shape (B=4228, MM**3) - 2d

# === Optional noise injection ===
if noise != 0:
    sol_data_train = sol_data_train * (1 + (noise / 100.0) * np.random.rand(*sol_data_train.shape))
    sol_data_test  = sol_data_test  * (1 + (noise / 100.0) * np.random.rand(*sol_data_test.shape))

if truncation:
    print(f"Original training samples: {sol_data_train.shape[0]}")
    threshold = args.threshold

    # Compute L2 norm per sample (axis=1 since shape is (B, MM**3))
    train_norms = np.linalg.norm(sol_data_train, axis=1)

    # Keep only samples with norm >= threshold
    train_mask = train_norms >= threshold

    sol_data_train = sol_data_train[train_mask]

    print(f"Filtered training samples (remaining): {sol_data_train.shape[0]}")

# === Convert to PyTorch tensors ===
train_tensor = torch.tensor(sol_data_train, dtype=torch.float64) 
test_tensor  = torch.tensor(sol_data_test,  dtype=torch.float64) 

# --- Train/Validation Split ---
train_dataset = TensorDataset(train_tensor, train_tensor)   # AE targets = inputs
test_dataset  = TensorDataset(test_tensor,  test_tensor)

# ---- Validation Split ----

num_total = len(train_dataset)
num_val = int(num_total * val_fraction)
num_train = num_total - num_val

train_dataset_split, val_dataset_split = torch.utils.data.random_split(
    train_dataset, [num_train, num_val]
)

# ---- DataLoaders ----
train_loader = DataLoader(train_dataset_split, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset_split, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# === Initialize model, loss function, optimizer ===
model = Autoencoder(solsize, code_len, hidden_layer_num, dropout=dropout, bias=bias).to(device)
model = model.double()
def relative_mse(pred, target, eps=1e-12):
    num = torch.sum((pred - target)**2, dim=1)
    den = torch.sum(target**2, dim=1) + eps
    return torch.mean(num / den)

criterion = relative_mse
optimizer = optim.NAdam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=eps, momentum_decay=momentum_decay)

print("Creating Model Path...")
# === Setup logging ===
ae_model_stamp = f"{datetime.now().strftime('%m_%d_%Y-%H_%M_%S')}-DC{DC}"
savedModelPath = f"AEs/delf-AE-{ae_model_stamp}"
Utilities.RemoveSavedModels(savedModelPath)
os.makedirs(savedModelPath, exist_ok=True)

outfile = open(os.path.join(savedModelPath, "output.txt"), "w")

saveEpochInfo = SaveEpochInfo(outfile, train_tensor, solsize, batch_size)
# === Early stopping setup ===
best_val_loss = float('inf')
lowest_val_loss = float('inf')
patience_counter = 0
best_model_state_dict = None
lowest_model_state_dict = None

print("Starting Training Loop...")
# === Training loop ===
for epoch in range(epochs_num):
    model.train()
    train_loss = 0
    for df_inp, df_true in train_loader:
        df_inp, df_true = df_inp.to(device), df_true.to(device)
        optimizer.zero_grad()
        pred = model(df_inp)
        loss = criterion(pred, df_true)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * df_inp.size(0)
    train_loss /= len(train_loader.dataset)

    # === Evaluate on val set for early stopping ===
    model.eval()
    val_loss = 0
    test_loss = 0
    with torch.no_grad():
        for df_inp, df_true in test_loader:
            df_inp, df_true = df_inp.to(device), df_true.to(device)
            pred = model(df_inp)
            loss = criterion(pred, df_true)
            test_loss += loss.item() * df_inp.size(0)
    test_loss /= len(test_loader.dataset)
    if val_fraction > 0.0:
        with torch.no_grad():
            for df_inp, df_true in val_loader:
                df_inp, df_true = df_inp.to(device), df_true.to(device)
                pred = model(df_inp)
                loss = criterion(pred, df_true)
                val_loss += loss.item() * df_inp.size(0)
        val_loss /= len(val_loader.dataset)
    else:
        val_loss = test_loss
    
    # Logging
    outfile.write(f"{epoch}\t{train_loss:.6f}\t{val_loss:.6f}\t{test_loss:.6f}\n")
    outfile.flush()
    
    
    # Always track the absolute lowest val loss
    if val_loss < lowest_val_loss:
        lowest_val_loss = val_loss
        lowest_model_state_dict = model.state_dict()

    if val_loss < best_val_loss - min_delta:
        best_val_loss = val_loss
        best_model_state_dict = model.state_dict()  # Save best model state
        patience_counter = 0
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, "
              f"Val Loss = {val_loss:.6f}, "
              f"Current Best Val Loss = {best_val_loss:.6f}")
    else:
        patience_counter += 1
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, "
              f"Val Loss = {val_loss:.6f}")
        if Early_Stopping:
            if patience_counter >= patience:
                # If we early stop, make sure we save the model corresponding
                # to the absolute lowest val loss observed
                print(f"Early stopping at epoch {epoch+1}. "
                      f"Last Calculated best Val loss: {best_val_loss:.6f}, "
                      f"Lowest overall Val loss: {lowest_val_loss:.6f}")

                # Save the absolute lowest model if it's different
                if lowest_val_loss < best_val_loss:
                    print("Saving model at lowest Val loss instead of min_delta best.")
                    best_model_state_dict = lowest_model_state_dict

                break

if Early_Stopping:
    # === Save best model ===
    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)
        torch.save(model.state_dict(), os.path.join(savedModelPath, "LearnSolWeights.pt"))
        torch.save(model, os.path.join(savedModelPath, "LearnSolModel.pt"))
if not Early_Stopping:
    torch.save(model.state_dict(), os.path.join(savedModelPath, "LearnSolWeights.pt"))
    torch.save(model, os.path.join(savedModelPath, "LearnSolModel.pt"))


# === Final evaluation on test set ===
model.eval()


saveEpochInfo.log(epoch, train_loss, test_loss, val_loss)
print(f"Model saved as {savedModelPath}")
# === Save run configuration ===
run_config_path = os.path.join(savedModelPath, "run_config.txt")
with open(run_config_path, "w") as f:
    f.write("==============================================\n")
    f.write("AUTOENCODER TRAINING RUN CONFIGURATION\n")
    f.write("==============================================\n\n")

    f.write(f"Timestamp: {ae_model_stamp}\n")
    f.write(f"Device: {device}\n\n")

    f.write("=== MODEL SETTINGS ===\n")
    f.write(f"Model_Class: Autoencoder\n")
    f.write(f"Hidden_Layers: {hidden_layer_num}\n")
    f.write(f"Code_Length: {code_len}\n")
    f.write(f"Input_Dimension (solsize): {solsize}\n")
    f.write("Activation: ReLU (inside AutoEncoder_Class)\n")
    f.write("Loss_Function: L1Loss\n\n")

    f.write("=== HYPERPARAMETERS ===\n")
    f.write(f"Bias: {bias}\n")
    f.write(f"Epochs: {epochs_num}\n")
    f.write(f"Batch_Size: {batch_size}\n")
    f.write(f"Optimizer: NAdam\n")
    f.write(f"Validation_split: {val_fraction}\n")
    f.write(f"Learning_Rate: {lr}\n")
    f.write(f"Beta_1: 0.9\n")
    f.write(f"Beta_2: 0.999\n")
    f.write(f"Epsilon: {eps}\n")
    f.write(f"Schedule_Decay: None (not implemented in PyTorch)\n")
    f.write(f"Weight_Decay: 0.0\n")
    f.write(f"Dropout_Rate: {dropout}\n")
    f.write(f"Patience: {patience}\n")
    f.write(f"Noise: {noise}%\n\n")

    f.write("=== EARLY STOPPING ===\n")
    f.write(f"Enabled: {Early_Stopping}\n")
    f.write(f"Min_Delta (early stop): {min_delta}\n")
    f.write(f"Patience: {patience} epochs\n")
    f.write(f"Best_Val_Loss: {best_val_loss:.6e}\n")
    f.write(f"Lowest_Val_Loss: {lowest_val_loss:.6e}\n")

    f.write("=== OPTIMIZER ===\n")
    f.write(f"Optimizer: {optimizer.__class__.__name__}\n")
    for k, v in optimizer.defaults.items():
        f.write(f"  {k}: {v}\n")

    f.write("=== DATA SETTINGS ===\n")
    f.write(f"Train_Data_Path: {train_path}\n")
    f.write(f"Test_Data_Path: {test_path}\n")
    f.write(f"Train_Samples: {train_tensor.shape[0]}\n")
    f.write(f"Test_Samples: {test_tensor.shape[0]}\n")
    f.write(f"Solution_Size: {solsize}\n\n")
    f.write("\n==============================================\n")
outfile.close()
