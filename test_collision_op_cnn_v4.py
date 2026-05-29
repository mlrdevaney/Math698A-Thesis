import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
import re
import my_readwrite
import Utilities
from AE_DeltaF_Class_v4 import Autoencoder
import pickle
from michaels_utils import parse_ae_metadata, parse_run_config, str2bool
import importlib
from my_distributions import maxwellian
from pathlib import Path


device = torch.device("cpu")
##home_folder = "/pscratch/sd/m/mdevaney/Boltzmann/Nguyen_1"
home_folder = "/Volumes/Devaney SSD/AFIT:ARCH/Nguyen_1"
datatype = "111TrD" # or "111TrD" | "222TsD"
folder = os.path.join(home_folder, "CNNs/AAA test folder")
test_folder = Path(folder)
cnn_models = [p.name for p in test_folder.iterdir() if p.is_dir()]
batch_size = 1
# === Load grid and node data for VTK ===
##filename_grids = '/Volumes/Devaney SSD/BoltzmannData/RUNS M=41/take3_M41/good/080/CollTrnDta180_1su1sv1sw41MuUU41MvVU41MwWU_grids.dat'
filename_grids = os.path.join(home_folder, 'CollTrnDta180_1su1sv1sw41MuUU41MvVU41MwWU_grids.dat')
grids_cap_u, grids_cap_v, grids_cap_w, grids_u_tmp, grids_v_tmp, grids_w_tmp = my_readwrite.read_grids(filename_grids)

##filename_nodes = '/Volumes/Devaney SSD/BoltzmannData/RUNS M=41/take3_M41/good/080/CollTrnDta180_1su1sv1sw41MuUU41MvVU41MwWU_nodes.dat'
filename_nodes = os.path.join(home_folder, 'CollTrnDta180_1su1sv1sw41MuUU41MvVU41MwWU_nodes.dat')
nodes_u, nodes_v, nodes_w, ngwts = my_readwrite.read_nodes(filename_nodes)

mu = grids_cap_u[0, 0]
grids_u = grids_u_tmp[:, 0:mu]
grids_v = grids_v_tmp[:, 0:mu]
grids_w = grids_w_tmp[:, 0:mu]

# === Load test data ===
sol_pkl = f"Data/Non-norm_{datatype}_delta_f_data_MM_41_MT_0_CT_1.0.pkl"
coll_pkl = f"Data/Non-norm_{datatype}_coll_data_MM_41_MT_0_CT_1.0.pkl"
sol_data_test = Utilities.LoadPickleSolData(sol_pkl)
coll_data_test = Utilities.LoadPickleSolData(coll_pkl)
filetype = "zeros_test"
match = re.search(r'Non-norm_(\d+)T', sol_pkl)
if match:
    filenum = int(match.group(1))
    if filenum == 111:
        filetype = "training"
    elif filenum == 222:
        filetype = "testing"
    else:
        filetype = None
else:
    filetype = "zeros_test"
    
for CNN_model in cnn_models:
    ##model_dir = "/Volumes/Devaney SSD/AFIT:ARCH/Nguyen_1/CNNs/AAA test folder/collop-CNN02_02_2026-22_08_25-DC2-SVD0-Herm1"
    model_dir = f"{folder}/{CNN_model}"
    model_path = os.path.join(model_dir, "LearnColOpCNNWeights.pt")
    config_path = os.path.join(os.path.dirname(model_path), "run_config.txt")
    config = parse_run_config(config_path)
    ##ae_model_dir = "/Volumes/Devaney SSD/AFIT:ARCH/Nguyen_1/AEs/delf-AE-01_13_2026-09_59_07"
    ae_model_dir = os.path.join(home_folder, f"{config.get('AE_model_path')}")
    AE_model_path = os.path.join(ae_model_dir, "LearnSolWeights.pt")
    bias_cnn = str2bool(config.get("Bias", "0"))
    cnn_class_version = config.get("Class_Version") #NOTE
    if cnn_class_version is None:
        cnn_class_version = 16
    module_name = f"CNN_CollOp_Class_v{cnn_class_version}"
    module = importlib.import_module(module_name)
    LearnColOpCNN = module.LearnColOpCNN

    solsize = sol_data_test.shape[1]
    dms = int(np.cbrt(solsize))
    ae_config = parse_run_config(os.path.join(ae_model_dir, "run_config.txt"))
    if "Code_Length" in ae_config and "Hidden_Layers" in ae_config and "Dropout_Rate" in ae_config:
        code_len = int(ae_config["Code_Length"])
        hidden_layers = int(ae_config["Hidden_Layers"])
        ae_dropout = float(ae_config["Dropout_Rate"])
        

    ### Convert to torch tensors
    sol_data_tensor = torch.tensor(sol_data_test.reshape(-1, 1, dms, dms, dms), dtype=torch.float32)
    coll_data_tensor = torch.tensor(coll_data_test, dtype=torch.float32)

    AE_model = Autoencoder(solsize, code_len, hidden_layers, dropout=ae_dropout)
    AE_model.load_state_dict(torch.load(AE_model_path, map_location=device))
    AE_model.to(device)
    AE_model.eval()

    print("Encoding Now:")
    # Pass sol_data through AE encoder
    with torch.no_grad():
        sol_tensor = torch.tensor(sol_data_test, dtype=torch.float32).to(device)
        encoded = AE_model.encoder(sol_tensor)  # shape: (B, code_len) - 2d

    # Reshape encoded vector to match CNN input (reshape to 1xDxDxD if code_len = D^3)
    dms_enc = int(round(code_len ** (1/3)))
    assert dms_enc**3 == code_len, f"code_len={code_len} must be a perfect cube for CNN 3D input."
    encoded_reshaped = encoded.view(-1, 1, dms_enc, dms_enc, dms_enc)  # Shape: (B=1, C=1, D, D, D) - 5d

    coll_tensor = torch.tensor(coll_data_test, dtype=torch.float32) # shape (B, MM*3) - 2d
    test_dataset = TensorDataset(encoded_reshaped.cpu(), coll_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True) # Shape: (B=32, C=1, D, D, D) - 5d

    model = LearnColOpCNN(dms_enc, solsize, bias=bias_cnn).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    eps = 1e-12

    print("Predicting collision operator...")
    out_path = os.path.join(model_dir, f"df_vs_snr_{filetype}.txt")
    out_file = open(out_path, "w")
    out_file.write("||Q_true||_2   SNR_dB      L2_ERR\n")

    with torch.no_grad():
        for inp, coll in test_loader:
            inp, coll = inp.to(device), coll.to(device)   # coll shape: [B, N]
            pred = model(inp)                             # pred shape: [B, N]
            q_norm = torch.norm(coll)

            inp_flat = coll.view(coll.size(0), -1)
            pred_flat = pred.view(pred.size(0), -1)

            signal_energy = torch.sum(inp_flat**2*ngwts)
            noise_energy  = torch.sum((inp_flat - pred_flat)**2*ngwts)

            snr_db = 10 * torch.log10((signal_energy + eps) / (noise_energy + eps))
            L2 = torch.linalg.vector_norm(coll-pred)/(torch.linalg.vector_norm(coll)+eps)
            out_file.write(f"{q_norm:.8e} {snr_db:.8f} {L2:.8f}\n")
    out_file.close()
    print(f"Wrote {out_path}")
