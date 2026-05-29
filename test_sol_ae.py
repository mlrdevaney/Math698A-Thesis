import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import re
import my_vtk_tools
import my_readwrite
from my_readwrite import solution_untrim
import Utilities
##from AE_DeltaF_Class_v3 import Autoencoder
from AE_DeltaF_Class_v4 import Autoencoder
from pathlib import Path
from michaels_utils import parse_ae_metadata

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 1

# === Load test data ===
testing_file = "Data/Non-norm_111TrD_delta_f_data_MM_41_MT_0_CT_1.0.pkl"
sol_data_test = Utilities.LoadPickleSolData(testing_file)
##sol_data_test = np.zeros((1, 41**3), dtype=np.float32)
sol_tensor = torch.tensor(sol_data_test, dtype=torch.float32).to(device)
test_loader = DataLoader(TensorDataset(sol_tensor, sol_tensor), batch_size=batch_size)
solsize = sol_data_test.shape[1]

# === AE Model Setup ===
folder = "/Volumes/Devaney SSD/AFIT:ARCH/Nguyen_1/AEs/AAA test folder"
test_folder = Path(folder)
ae_models = [p.name for p in test_folder.iterdir() if p.is_dir()]
eps = 1e-12
for AE_model in ae_models:
    snr_values = []
    AE_model_dir = f"{folder}/{AE_model}"
    AE_model_path = os.path.join(AE_model_dir, "LearnSolWeights.pt")
    ae_info = parse_ae_metadata(AE_model_dir)
    code_len = ae_info["code_len"]
    hidden_layers = ae_info["hidden_layers"]
    AE = Autoencoder(solsize, code_len, hidden_layers)
    AE.load_state_dict(torch.load(AE_model_path, map_location=device))
    AE.to(device)
##    AE = AE.double()
    AE.eval()
    filetype = "zeros_test"
    match = re.search(r'Non-norm_(\d+)T', testing_file)
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
    out_path = os.path.join(AE_model_dir, f"df_vs_snr_{filetype}.txt")
    f_out = open(out_path, "w")
    f_out.write("||df_true||_2       SNR_dB      L2_ERR\n")
    
    print("Encoding Now:")
    
    with torch.no_grad():
        for inp, true in test_loader:
            inp, true = inp.to(device), true.to(device)
            pred = AE(inp)
            
            # flatten per sample (batch size = 1)
            df      = inp.view(-1)
            df_pred = pred.view(-1)
            signal_energy = torch.sum(df**2)
            noise_energy  = torch.sum((df - df_pred)**2)
            L2 = torch.linalg.vector_norm(true-pred)/(torch.linalg.vector_norm(true)+eps)
            df_norm = torch.sqrt(signal_energy).item()
            snr_db  = 10 * torch.log10((signal_energy + eps) / (noise_energy + eps))
            snr_db  = snr_db.item()
            f_out.write(f"{df_norm:.8e} {snr_db:.8f} {L2:.8f}\n")

    f_out.close()
    print(f"Wrote {out_path}")
