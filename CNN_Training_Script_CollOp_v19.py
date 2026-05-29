print("Script Launched...")
import numpy as np
import os
import math
import re
import Utilities
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
import time
import pickle
import matplotlib.pyplot as plt
import my_readwrite
from CNN_CollOp_Class_v16 import LearnColOpCNN
import argparse
import inspect
import my_utils
from michaels_utils import parse_ae_metadata, build_ksis, huber_torch

parser = argparse.ArgumentParser(description="Boltzmann Collision CNN Training")

# Boolean/toggle features
parser.add_argument("--SVD_Cons", type=int, default=0)
parser.add_argument("--Herm_Cons", type=int, default=1)
parser.add_argument("--Bias", type=int, default=0)
parser.add_argument("--L1", type=int, default=0)
parser.add_argument("--L2", type=int, default=0)
parser.add_argument("--Use_AE", type=int, default=1)
parser.add_argument("--Weak_Projection", type=int, default=0)
parser.add_argument("--Lyapunov_Constraints", type=int, default=1)
parser.add_argument("--Stationarity", type=int, default=0)
parser.add_argument("--Isotropization", type=int, default=0)
parser.add_argument("--Entropy", type=int, default=1)

# Hyperparameters
parser.add_argument("--l1_val", type=float, default=1e-15)
parser.add_argument("--l2_val", type=float, default=1e-15)
parser.add_argument("--stat_val", type=float, default=1e-15)
parser.add_argument("--wpc", type=float, default=2.5e-16)
parser.add_argument("--lypc", type=float, default=1e-13)
parser.add_argument("--m_gpt", type=float, default=1.0)
parser.add_argument("--entropy_const", type=float, default=1e-9)
parser.add_argument("--iso_const", type=float, default=1e-15)
parser.add_argument("--daily_model_count", type=int, default=0)
parser.add_argument("--epochs_num", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--loss_fn_name", type=str, default="MAE")
parser.add_argument("--lr", type=float, default=2e-3)
parser.add_argument("--eps", type=float, default=1e-8)
parser.add_argument("--d_huber", type=float, default=1.0)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--val_split", type=float, default=0.0)
parser.add_argument("--early_stop", type=int, default=1)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--min_delta", type=float, default=1e-5)
parser.add_argument("--train_delf_path", type=str, default="Data/08xs_delta_f_data_MM_41_MT_0_CT_1.0.pkl")
parser.add_argument("--train_coll_path", type=str, default="Data/08xs_coll_data_MM_41_MT_0_CT_1.0.pkl")
parser.add_argument("--train_maxwellian_path", type=str, default="Data/08xs_maxwellian_data_MM_41_MT_0_CT_1.0.pkl")
parser.add_argument("--test_delf_path", type=str, default="Data/11xs_delta_f_data_MM_41_MT_0_CT_1.0.pkl")
parser.add_argument("--test_coll_path", type=str, default="Data/11xs_coll_data_MM_41_MT_0_CT_1.0.pkl")
parser.add_argument("--test_maxwellian_path", type=str, default="Data/11xs_maxwellian_data_MM_41_MT_0_CT_1.0.pkl")
parser.add_argument("--ae_path", type=str, default="AEs/delf-AE-02_16_2026-18_58_35-DC6")
args = parser.parse_args()

# === Assign arguments ===
SVD_Cons = bool(args.SVD_Cons)
Herm_Cons = bool(args.Herm_Cons)
Bias = bool(args.Bias)
L1 = bool(args.L1)
L2 = bool(args.L2)
Use_AE = bool(args.Use_AE)
Weak_Projection = bool(args.Weak_Projection)
Lyapunov_Constraints = bool(args.Lyapunov_Constraints)
Stationarity = bool(args.Stationarity)
Isotropization = bool(args.Isotropization)
Entropy = bool(args.Entropy)
Early_Stopping = bool(args.early_stop)

l1_val = args.l1_val
l2_val = args.l2_val
stat_val = args.stat_val
wpc = args.wpc
lypc = args.lypc
m_gpt = args.m_gpt
entropy_const = args.entropy_const
iso_const = args.iso_const
epochs_num = args.epochs_num
batch_size = args.batch_size
loss_fn_name = args.loss_fn_name
optim_lr = args.lr
eps = args.eps
d_huber = args.d_huber
dropout = args.dropout
train_delf_path = args.train_delf_path
train_coll_path = args.train_coll_path
train_maxwellian_path = args.train_maxwellian_path
test_delf_path = args.test_delf_path
test_coll_path = args.test_coll_path
test_maxwellian_path = args.test_maxwellian_path
ae_path = args.ae_path
val_fraction = args.val_split
patience = args.patience
min_delta = args.min_delta
daily_model_count = args.daily_model_count

# === Utility Callback ===
class SaveEpochInfo:
    def __init__(self, outfile, sol_data_train, solsize, batch_size):
        self.outfile = outfile
        self.outfile.write(f"training samples: {sol_data_train.shape[0]}, solution size: {solsize}, batch size: {batch_size}\n\n")
        self.outfile.write(f"epoch\ttrain_loss\t\ttest_loss\t\tval_loss\t\ttrain_l2_max_error\t\ttest_l2_max_error\t\tval_l2_max_error\n")
        self.outfile.flush()

    def log(self, epoch, train_loss, test_loss, val_loss, train_l2errmax, test_l2errmax, val_l2errmax):
        infoStr = f"{epoch:03d}\t{train_loss:.10e}\t{test_loss:.10e}\t{val_loss:.10e}\t{train_l2errmax:.6e}\t\t\t{test_l2errmax:.6e}\t\t\t{val_l2errmax:.6e}\n"
        self.outfile.write(infoStr)
        self.outfile.flush()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################################################################################
########################################################################################################
########################################################################################################

nodes_path = 'CollTrnDta180_1su1sv1sw41MuUU41MvVU41MwWU_nodes.dat'
nodes_u, nodes_v, nodes_w, nodes_gwts = my_readwrite.read_nodes(nodes_path)
nodes_u = torch.tensor(nodes_u, dtype=torch.float64).to(device)
nodes_v = torch.tensor(nodes_v, dtype=torch.float64).to(device)
nodes_w = torch.tensor(nodes_w, dtype=torch.float64).to(device)
nodes_gwts = torch.tensor(nodes_gwts, dtype=torch.float64).to(device)
maxwellian_data = Utilities.LoadPickleSolData(train_maxwellian_path) # shape (B=2899, MM*3) - 2d
maxwellian_data_test = Utilities.LoadPickleSolData(test_maxwellian_path) # shape (B=2899, MM*3) - 2d
maxwell = torch.tensor(maxwellian_data, dtype=torch.float64).to(device)
maxwell_test = torch.tensor(maxwellian_data_test, dtype=torch.float64).to(device)
# Load and freeze Autoencoder
if Use_AE:
    print("Preparing AE Model...")
    from AE_DeltaF_Class_v4 import Autoencoder
    sol_data_raw = Utilities.LoadPickleSolData(train_delf_path) # shape (B=29249, MM*3) - 2d
    coll_data = Utilities.LoadPickleSolData(train_coll_path) # shape (B=29249, MM*3) - 2d
    sol_data_test = Utilities.LoadPickleSolData(test_delf_path) # shape (B=29249, MM*3) - 2d
    coll_data_test = Utilities.LoadPickleSolData(test_coll_path) # shape (B=29249, MM*3) - 2d
    ae_model_path = f"{ae_path}/LearnSolWeights.pt"
    ae_info = parse_ae_metadata(ae_path)
    code_len = ae_info["code_len"]
    hidden_layer_num = ae_info["hidden_layers"]
    ae_dropout = ae_info["dropout"]
    solsize = sol_data_raw.shape[1] # = MM**3 for untrimmed, (MM-2*Mtrim)**3 for trimmed
    dms = int(np.cbrt(solsize)) # = MM for untrimmed, (MM-2*Mtrim) for trimmed
    dms_cnn = int(round(code_len ** (1/3)))
    assert dms_cnn**3 == code_len, f"code_len={code_len} must be a perfect cube"
    ae_model = Autoencoder(solsize, code_len, hidden_layer_num, ae_dropout, bias=False).to(device)
    ae_model.load_state_dict(torch.load(ae_model_path, map_location=device))
    ae_model = ae_model.double()
    ae_model.eval()
    for param in ae_model.parameters():
        param.requires_grad = False
else:
    sol_data_raw = Utilities.LoadPickleSolData(train_delf_path) # shape (B=29249, MM*3) - 2d
    coll_data = Utilities.LoadPickleSolData(train_coll_path) # shape (B=29249, MM*3) - 2d
    sol_data_test = Utilities.LoadPickleSolData(test_delf_path) # shape (B=29249, MM*3) - 2d
    coll_data_test = Utilities.LoadPickleSolData(test_coll_path) # shape (B=29249, MM*3) - 2d
    solsize = sol_data_raw.shape[1] # = MM**3 for untrimmed, (MM-2*Mtrim)**3 for trimmed
    dms = int(np.cbrt(solsize)) # = MM for untrimmed, (MM-2*Mtrim) for trimmed
    dms_cnn = dms  # raw solution shape cube root
    hidden_layer_num = 0
    code_len = 0
    ae_model = None

# === Unified Input Prep ===
print("Preparing Input...")
def prepare_input(sol_data_raw, use_ae, ae_model, solsize, dms_cnn):
    sol_tensor = torch.tensor(sol_data_raw, dtype=torch.float64).to(device)
    if use_ae:
        with torch.no_grad():
            encoded = ae_model.encoder(sol_tensor)
            return encoded.view(-1, 1, dms_cnn, dms_cnn, dms_cnn), sol_tensor
    else:
        return sol_tensor.view(-1, 1, dms_cnn, dms_cnn, dms_cnn), sol_tensor
cnn_input, sol_tensor = prepare_input(sol_data_raw, Use_AE, ae_model, solsize, dms_cnn) 
coll_tensor = torch.tensor(coll_data, dtype=torch.float64)  # shape (B, MM*3) - 2d if untrimmed
cnn_input_test, un_enc_df_test = prepare_input(sol_data_test, Use_AE, ae_model, solsize, dms_cnn) 
coll_tensor_test = torch.tensor(coll_data_test, dtype=torch.float64)
un_enc_delf = sol_tensor - maxwell

train_dataset = TensorDataset(cnn_input, coll_tensor, maxwell, un_enc_delf)
test_dataset = TensorDataset(cnn_input_test, coll_tensor_test, maxwell_test, un_enc_df_test)

class_file = inspect.getfile(LearnColOpCNN)
match = re.search(r'CNN_CollOp_Class_v(\d+).py', class_file)
if match:
    class_version = int(match.group(1))
else:
    raise ValueError("Could not determine CNN class version.")

# --- Train/Validation Split ---
num_total = len(train_dataset)
num_val = int(num_total * val_fraction)
num_train = num_total - num_val

train_dataset_split, val_dataset_split = torch.utils.data.random_split(
    train_dataset, [num_train, num_val]
)

train_loader = DataLoader(train_dataset_split, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset_split, batch_size=batch_size, shuffle=False)

# --- Check data magnitudes before training ---
sample_Qs = []
for batch in train_loader:
    _, Q, _, _ = batch
    sample_Qs.append(Q.flatten())
    if len(sample_Qs) > 5: 
        break

Q_concat = torch.cat(sample_Qs)
print(f"Q mean: {Q_concat.mean().item():.3e}, std: {Q_concat.std().item():.3e}, "
      f"min: {Q_concat.min().item():.3e}, max: {Q_concat.max().item():.3e}")

# ----- Compute or load singular vectors ----------
if SVD_Cons or Weak_Projection:
    print("Loading Singular Vectors")
    nodes_path = 'CollTrnDta180_1su1sv1sw41MuUU41MvVU41MwWU_nodes.dat'
    match = re.search(r'sw(\d+)MuUU', nodes_path)
    if match:
        M_val = int(match.group(1))
        SVD_pkl_file = f'SVD_U_M{M_val}_double.pkl'
        if os.path.exists(SVD_pkl_file) and os.path.getsize(SVD_pkl_file) > 0:
            print(f"Found precomputed SVD for M={M_val}. Loading from {SVD_pkl_file}...")
            with open(SVD_pkl_file, 'rb') as svd_file:
                U = pickle.load(svd_file)
            U = U.to(device).double()
        else:
            if os.path.exists(SVD_pkl_file):
                print(f"{SVD_pkl_file} exists but is empty. Recomputing SVD...")
            else:
                print("Pickle file for number of velocity nodes not found. Computing SVD...")
            nodes_u, nodes_v, nodes_w, nodes_gwts = my_readwrite.read_nodes(nodes_path)
            Mat_Moments = np.zeros((nodes_u.shape[1], 5))
            Mat_Moments[:, 0] = nodes_gwts[0,:]
            Mat_Moments[:, 1] = nodes_gwts[0,:] * nodes_u[0, :]
            Mat_Moments[:, 2] = nodes_gwts[0,:] * nodes_v[0,:]
            Mat_Moments[:, 3] = nodes_gwts[0,:] * nodes_w[0,:]
            scrp_array = (nodes_u ** 2 + nodes_v ** 2 + nodes_w ** 2)
            Mat_Moments[:, 4] = nodes_gwts[0,:] * scrp_array[0,:]
            M = torch.tensor(Mat_Moments, dtype=torch.float64, device=device)
            U, S, V = torch.linalg.svd(M, full_matrices=False) #U-shape: (M^3, 5); S-shape: (5); V-shape: (5, 5)
            U = U.to(device).double()
            with open(SVD_pkl_file, 'wb') as f:
                pickle.dump(U, f)
            print(f"Saved SVD matrix to {SVD_pkl_file}")
    else:
        raise ValueError("Could not extract M from nodes_path")

# === CNN Setup ===
if not L2:
    l2_val = 0
model = LearnColOpCNN(dms=dms_cnn, output_dim=solsize, bias=Bias, dropout_rate=dropout).to(device)
model = model.double()
criterion = nn.L1Loss() if loss_fn_name == "MAE" else nn.MSELoss()
optimizer = optim.Adamax(model.parameters(), lr=optim_lr, betas=(0.9, 0.999), eps=eps, weight_decay=l2_val) #Nguyen uses lr=0.001, eps=0.0

# === Output Directory Setup ===
flags = [f"SVD{int(SVD_Cons)}", f"Herm{int(Herm_Cons)}"]
flags_str = "-".join(flags)
cnn_model_datetime = datetime.now().strftime('%m_%d_%Y-%H_%M_%S')
savedModelPath = f"CNNs/collop-CNN{cnn_model_datetime}-DC{daily_model_count}-{flags_str}"
print("Model Folder Name:", savedModelPath)
Utilities.RemoveSavedModels(savedModelPath)
os.makedirs(savedModelPath, exist_ok=True)
outfile = open(os.path.join(savedModelPath, "output.txt"), "w")
saveEpochInfo = SaveEpochInfo(outfile, cnn_input, solsize, batch_size)


best_monitored_loss = float('inf')
lowest_monitored_loss = float('inf')
best_test_loss = float('inf')
lowest_test_loss = float('inf')
best_val_loss = float('inf')
lowest_val_loss = float('inf')
patience_counter = 0
best_model_state_dict = None
l2errmax_per_epoch = []


def hermite_projection(pred, fM, ksis, nodes_gwts):
    # ksis: (B, N, 5), fM: (B, N), gwts: (1, N)
    fM = fM.view(fM.shape[0], fM.shape[1], 1)  # (B, N, 1)
    gwts = nodes_gwts.view(1, nodes_gwts.shape[1], 1)      # (1, N, 1)
    for j in range(5):
        kj = ksis[:, :, j:j+1]                 # (B, N, 1)
        ip = torch.sum(pred.unsqueeze(2) * kj * gwts, dim=1, keepdim=True)  # (B, 1, 1)
        inp = pred - (fM.squeeze(-1) * kj.squeeze(-1) * ip.squeeze(-1))  # (B, N)
    return inp

print("Starting Training Loop...")
for epoch in range(epochs_num):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        delta_f, Q, fM, delf_arc = batch
        delta_f, Q, fM, delf_arc = delta_f.to(device), Q.to(device), fM.detach().to(device), delf_arc.detach().to(device)
        ksis = build_ksis(nodes_u, nodes_v, nodes_w, nodes_gwts, fM)
        optimizer.zero_grad()
        pred = model(delta_f)

        if SVD_Cons:
            inp = pred - (pred @ U) @ U.T

        elif Herm_Cons:
            inp = hermite_projection(pred, fM, ksis, nodes_gwts * math.sqrt(2/0.2))
            
        else:
            inp = pred
            
        loss = criterion(inp, Q)
        
        if Stationarity:
            Q_M = model(torch.zeros_like(delta_f))
            loss += stat_val * torch.mean(Q_M**2)

        if Entropy:
            f = delf_arc + fM
            f = torch.clamp(f, min=1e-12)
            Qlogf = torch.clamp(torch.sum(pred * torch.log(f)), min=0.0)
            loss += entropy_const * Qlogf
            
        if Weak_Projection:
            loss += wpc * torch.linalg.norm(pred @ U)

        if L1:
            l1_norm = sum(torch.sum(torch.abs(param)) for param in model.parameters())
            loss += l1_val * l1_norm

        if Lyapunov_Constraints:
            f = delf_arc + fM
            logf = torch.log(torch.clamp(f, min=1e-16))
            logfM = torch.log(fM)
            Qlogf = torch.sum(pred * logf)
            flogf = torch.sum(f * logf)
            fMlogfM = torch.sum(fM * logfM)
            dflogdif = torch.sum(delf_arc * (logf - logfM))
            loss += lypc * huber_torch(Qlogf + m_gpt * (flogf - fMlogfM), d_huber) / (dflogdif + eps) #NOTE

        if Isotropization:
            sol = delf_arc + fM
            Tx, Ty, Tz = my_utils.get_directional_temperatures_torch(sol, nodes_u, nodes_v, nodes_w, nodes_gwts)
            iso_loss = torch.mean(
                (Tx - Ty)**2 + (Ty - Tz)**2 + (Tx - Tz)**2
            )

            loss += iso_const * iso_loss
                
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * delta_f.size(0)

    avg_train_loss = total_loss / len(train_loader.dataset)
    
    # Max L2 error
    model.eval()
    train_l2errmax = 0.0
    with torch.no_grad():
        for batch in train_loader:
            delta_f, Q, fM, delf_arc = batch
            delta_f, Q, fM, delf_arc = delta_f.to(device), Q.to(device), fM.detach().to(device), delf_arc.detach().to(device)
            ksis = build_ksis(nodes_u, nodes_v, nodes_w, nodes_gwts, fM)
            pred = model(delta_f) # shape: [B, M**3 = N]
            if SVD_Cons:
                pred = pred - (pred @ U) @ U.T
            elif Herm_Cons:
                pred = hermite_projection(pred, fM, ksis, nodes_gwts * math.sqrt(2/0.2))
            else:
                pred = pred
            train_l2errmax = max(train_l2errmax, max(torch.norm(pred - Q, dim=1)).item())
    l2errmax_per_epoch.append(train_l2errmax)

    # === Validation & Testing Stage ===
    model.eval()
    val_loss = 0.0
    val_l2errmax = 0.0
    test_loss = 0.0
    test_l2errmax = 0.0
    with torch.no_grad():
        for batch in val_loader:
            delta_f, Q, fM, delf_arc = batch
            delta_f, Q, fM, delf_arc = delta_f.to(device), Q.to(device), fM.detach().to(device), delf_arc.detach().to(device)
            ksis = build_ksis(nodes_u, nodes_v, nodes_w, nodes_gwts, fM)
            pred = model(delta_f)
            if SVD_Cons:
                pred = pred - (pred @ U) @ U.T
            elif Herm_Cons:
                pred = hermite_projection(pred, fM, ksis, nodes_gwts * math.sqrt(2/0.2))
            else:
                pred = pred
            val_loss += criterion(pred, Q).item() * delta_f.size(0)
        val_l2errmax = max(val_l2errmax, max(torch.norm(pred - Q, dim=1)).item())
        for batch in test_loader:
            delta_f, Q, fM, df_raw = batch
            delta_f, Q, fM, df_raw = delta_f.to(device), Q.to(device), fM.detach().to(device), df_raw.to(device)
            ksis = build_ksis(nodes_u, nodes_v, nodes_w, nodes_gwts, fM)
            pred = model(delta_f)
            if SVD_Cons:
                pred = pred - (pred @ U) @ U.T
            elif Herm_Cons:
                pred = hermite_projection(pred, fM, ksis, nodes_gwts * math.sqrt(2/0.2))
            else:
                pred = pred
            test_loss += criterion(pred, Q).item() * delta_f.size(0)
        test_l2errmax = max(test_l2errmax, max(torch.norm(pred - Q, dim=1)).item())

    test_loss /= len(test_loader.dataset)
    if len(val_loader.dataset) > 0:
        val_loss /= len(val_loader.dataset)
        monitored_loss = val_loss
    else:
        monitored_loss = test_loss

    # Correct: track lowest observed value of monitored metric
    lowest_monitored_loss = min(lowest_monitored_loss, monitored_loss)
    if len(val_loader.dataset) > 0:
        lowest_val_loss = lowest_monitored_loss
    else:
        lowest_test_loss = lowest_monitored_loss

    saveEpochInfo.log(
        epoch, avg_train_loss, test_loss, val_loss,
        train_l2errmax, test_l2errmax, val_l2errmax
    )

    # ---------- Check for improvement ----------
    if monitored_loss < best_monitored_loss - min_delta:
        # Improvement
        best_monitored_loss = monitored_loss
        best_model_state_dict = model.state_dict()
        patience_counter = 0

        if len(val_loader.dataset) > 0:
            print(f"Epoch {epoch+1}: Train={avg_train_loss:.6f}, "
                  f"Val={val_loss:.6f}, Best Val={best_monitored_loss:.6f}, "
                  f"Test={test_loss:.6f}")
            best_val_loss = best_monitored_loss
            
        else:
            print(f"Epoch {epoch+1}: Train={avg_train_loss:.6f}, "
                  f"Test={test_loss:.6f}, Best Test={best_monitored_loss:.6f}")
            best_test_loss = best_monitored_loss

    else:
        # No improvement
        patience_counter += 1
        if len(val_loader.dataset) > 0:
            print(f"Epoch {epoch+1}: Train={avg_train_loss:.6f}, "
                  f"Val={val_loss:.6f}, Test={test_loss:.6f}")
        else:
            print(f"Epoch {epoch+1}: Train={avg_train_loss:.6f}, "
                  f"Test={test_loss:.6f}")

        # ---------- Early Stopping ----------
        if Early_Stopping and patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}. "
                  f"Best monitored loss = {best_monitored_loss:.6f}")
            break


if Early_Stopping:
    # === Save best model ===
    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)
        torch.save(model.state_dict(), os.path.join(savedModelPath, "LearnColOpCNNWeights.pt"))
        torch.save(model, os.path.join(savedModelPath, "LearnColOpCNNModel.pt"))
if not Early_Stopping:
    torch.save(model.state_dict(), os.path.join(savedModelPath, "LearnColOpCNNWeights.pt"))
    torch.save(model, os.path.join(savedModelPath, "LearnColOpCNNModel.pt"))
    
# === Plot L2 Error ===
plt.figure(figsize=(8,6))
plt.plot(range(len(l2errmax_per_epoch)), l2errmax_per_epoch, marker='o', label='Max L2 Error per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Max L2 Error')
plt.title('Training Max L2 Error vs Epoch')
plt.grid(True)
plt.legend()
plot_path = os.path.join(savedModelPath, 'l2_max_error_per_epoch.png')
plt.savefig(plot_path)
plt.close()

# === Log all configuration parameters ===
config_path = os.path.join(savedModelPath, "run_config.txt")
with open(config_path, "w") as cfg:
    cfg.write("==============================================\n")
    cfg.write("CONVOLUTIONAL NEURAL NETWORK TRAINING RUN CONFIGURATION\n")
    cfg.write("==============================================\n\n")
    
    cfg.write(f"Timestamp: {cnn_model_datetime}\n")
    cfg.write(f"Device: {device}\n")
    cfg.write(f"Model class source file: {class_file}\n")
    cfg.write(f"Class_Version: {class_version}\n\n")

    cfg.write("=== MODEL SETTINGS ===\n")
    cfg.write(f"SVD_Cons: {SVD_Cons}\n")
    cfg.write(f"Herm_Cons: {Herm_Cons}\n")
    cfg.write(f"Bias: {Bias}\n")
    cfg.write(f"L1: {L1}\n")
    cfg.write(f"L2: {L2}\n")
    cfg.write(f"Use_AE: {Use_AE}\n")
    cfg.write(f"Weak_Projection: {Weak_Projection}\n")
    cfg.write(f"Lyapunov_Constraints: {Lyapunov_Constraints}\n")
    cfg.write(f"Stationarity: {Stationarity}\n")
    cfg.write(f"Isotropization: {Isotropization}\n")
    cfg.write(f"Entropy: {Entropy}\n\n")

    cfg.write("=== HYPERPARAMETERS ===\n")
    cfg.write(f"l1_val: {l1_val}\n")
    cfg.write(f"l2_val: {l2_val}\n")
    cfg.write(f"wpc: {wpc}\n")
    cfg.write(f"lypc: {lypc}\n")
    cfg.write(f"entropy_const: {entropy_const}\n")
    cfg.write(f"stat_val: {stat_val}\n")
    cfg.write(f"iso_const: {iso_const}\n")
    cfg.write(f"validation_split: {val_fraction}\n")
    cfg.write(f"epochs_num: {epochs_num}\n")
    cfg.write(f"batch_size: {batch_size}\n")
    cfg.write(f"loss_fn_name: {loss_fn_name}\n")
    cfg.write(f"dropout_rate: {dropout}\n\n")
    cfg.write(f"Min_Delta (early stop): {min_delta}\n")
    cfg.write(f"Patience: {patience}\n\n")

    cfg.write("\n=== EARLY STOPPING ===\n")
    cfg.write(f"Enabled: {Early_Stopping}\n")
    cfg.write(f"Criterion: test_loss improves by > {min_delta}\n")
    cfg.write(f"Patience: {patience} epochs\n")
    if len(val_loader.dataset) > 0:
        cfg.write(f"Best_Val_Loss: {best_val_loss:.6e}\n")
        cfg.write(f"Lowest_Val_Loss: {lowest_val_loss:.6e}\n\n")
    else:
        cfg.write(f"Best_Test_Loss: {best_test_loss:.6e}\n")
        cfg.write(f"Lowest_Test_Loss: {lowest_test_loss:.6e}\n\n")

    cfg.write("=== OPTIMIZER ===\n")
    cfg.write(f"Optimizer: {optimizer.__class__.__name__}\n")
    for k, v in optimizer.defaults.items():
        cfg.write(f"  {k}: {v}\n")
    cfg.write("\n")

    cfg.write("=== AUTOENCODER ===\n")
    cfg.write(f"AE model path: {ae_path if Use_AE else 'N/A'}\n")
    cfg.write(f"Code length: {code_len}\n")
    cfg.write(f"Hidden layers: {hidden_layer_num}")
    cfg.write("\n==============================================\n")

    cfg.write("=== DATA SETTINGS ===\n")
    cfg.write(f"Train_Delf_Path: {train_delf_path}\n")
    cfg.write(f"Test_Delf_Path: {test_delf_path}\n")
    cfg.write(f"Train_Coll_Path: {train_coll_path}\n")
    cfg.write(f"Test_Coll_Path: {test_coll_path}\n")
    cfg.write(f"Train_Maxwell_Path: {train_maxwellian_path}\n")
    cfg.write(f"Test_Maxwell_Path: {test_maxwellian_path}\n")
    cfg.write(f"Train_Delf_Samples: {cnn_input.shape[0]}\n")
    cfg.write(f"Test_Delf_Samples: {cnn_input_test.shape[0]}\n")
    cfg.write(f"Train_Coll_Samples: {coll_tensor.shape[0]}\n")
    cfg.write(f"Test_Coll_Samples: {coll_tensor_test.shape[0]}\n")
    cfg.write(f"Train_Maxwell_Samples: {maxwell.shape[0]}\n")
    cfg.write(f"Test_Maxwell_Samples: {maxwell_test.shape[0]}\n")
    cfg.write(f"Solution_Size: {solsize}\n\n")
    cfg.write(f"dms: {dms}\n")
    cfg.write(f"dms_cnn: {dms_cnn}\n")
    cfg.write("\n============================\n")
outfile.close()
