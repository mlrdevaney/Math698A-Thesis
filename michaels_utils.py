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
import argparse
import inspect
import my_utils
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib


def build_ksis(nodes_u, nodes_v, nodes_w, nodes_gwts, fM): # if fM has shape: (B, MM**3)
    import torch
    import math
    def H(n, x):
        if n == 0:
            return torch.ones_like(x)
        elif n == 1:
            return x
        else:
            raise ValueError("Hermite polynomials up to order 1 are supported.")
    scale = math.sqrt(2/(0.2))
    nodes_u = (nodes_u * scale).flatten().clone().detach()
    nodes_v = (nodes_v * scale).flatten().clone().detach()
    nodes_w = (nodes_w * scale).flatten().clone().detach()
    gwts = (nodes_gwts * scale).flatten().clone().detach()

    B, N = fM.shape

    # Verify orthog. between different Hermite polynomials
##    for y in range(2):
##        for z in range(2):
##            if y<z:
##                print(f"U Orthogonality of H{y} with H{z}", float(torch.sum(fM * gwts * H(y, nodes_u) * H(z, nodes_u))))
##                print(f"V Orthogonality of H{y} with H{z}", float(torch.sum(fM * gwts * H(y, nodes_v) * H(z, nodes_v))))
##                print(f"W Orthogonality of H{y} with H{z}", float(torch.sum(fM * gwts * H(y, nodes_w) * H(z, nodes_w))))


    
    ksi_list = []
    for k in range(2):
        for j in range(2):
            for i in range(2):
                if i + j + k <= 1: #if (i,j,k) then ksi_1 = (0,0,0); ksi_2 = (1,0,0); ksi_3 = (0,1,0); ksi_4 = (0,0,1)
                    Hi = H(i, nodes_u)
                    Hj = H(j, nodes_v)
                    Hk = H(k, nodes_w)
                    ksi = Hi * Hj * Hk
                    ksi_list.append(ksi)

    ksi_array = torch.stack(ksi_list, dim=0)  # shape: (4, N)

    # Normalize each ksi[i] w.r.t. fM for each batch
    ksi_array = ksi_array.unsqueeze(0).expand(B, -1, -1)  # (B, 4, N)
    fM_batched = fM.unsqueeze(1)  # (B, 1, N)
    gwts_batched = gwts.view(1, 1, N)   # (1, 1, N)

    norm = torch.sqrt(torch.sum(ksi_array**2 * fM_batched * gwts_batched, dim=2, keepdim=True))  # (B, 4, 1)
    ksi_array = ksi_array / norm  # (B, 4, N)

    # Compute ksi_5_hat = |v|^2 = u^2 + v^2 + w^2, shape (N,)
    ksi_5_hat = nodes_u**2 + nodes_v**2 + nodes_w**2  # (N,)
    ksi_5_hat = ksi_5_hat.view(1, 1, N).expand(B, -1, -1).clone()  # (B, 1, N)

    # Orthogonalize ksi_5_hat against first 4 basis functions
    for j in range(4):
        inner_prod = torch.sum(ksi_5_hat * ksi_array[:, j:j+1, :] * fM_batched * gwts_batched, dim=2, keepdim=True)  # (B, 1, 1)
        ksi_5_hat -= inner_prod * ksi_array[:, j:j+1, :]  # (B, 1, N)

    # Normalize ksi_5
    norm_5 = torch.sqrt(torch.sum(ksi_5_hat**2 * fM_batched * gwts_batched, dim=2, keepdim=True))  # (B, 1, 1)
    ksi_5 = ksi_5_hat / norm_5  # (B, 1, N)

    # Combine all 5 ksis: shape (B, 5, N)
    ksis = torch.cat([ksi_array, ksi_5], dim=1)  # (B, 5, N)
    ksis = ksis.permute(0, 2, 1)  # (B, N, 5)

##    for y in range(5):
##        for z in range(5):
##            if y<=z:
##                print(f"Orthogonality between ksi{y+1} and ksi{z+1}", float(torch.sum(ksis[:, :, y] * ksis[:, :, z] * fM_batched * gwts_batched)))


    return ksis

def build_ksis_np(nodes_u, nodes_v, nodes_w, nodes_gwts, fM):
    import numpy as np
    import math

    def H(n, x):
        if n == 0:
            return np.ones_like(x)
        elif n == 1:
            return x
        else:
            raise ValueError("Hermite polynomials up to order 1 are supported.")

    scale = math.sqrt(2 / 0.2)

    nodes_u = (nodes_u * scale).reshape(-1)
    nodes_v = (nodes_v * scale).reshape(-1)
    nodes_w = (nodes_w * scale).reshape(-1)
    gwts    = (nodes_gwts * scale).reshape(-1)

    fM = fM.reshape(-1)
    N = fM.shape[0]

    ##############################
    # First 4 ksis
    ##############################
    ksi_list = []
    for k in range(2):
        for j in range(2):
            for i in range(2):
                if i + j + k <= 1:
                    ksi = H(i, nodes_u) * H(j, nodes_v) * H(k, nodes_w)
                    ksi_list.append(ksi)

    ksi_array = np.stack(ksi_list, axis=0)  # (4, N)

    ##############################
    # Normalize ksis
    ##############################
    norm = np.sqrt(np.sum(ksi_array**2 * fM * gwts, axis=1, keepdims=True))  # (4,1)
    ksi_array = ksi_array / norm

    ##############################
    # ksi_5 = |v|^2
    ##############################
    ksi_5 = nodes_u**2 + nodes_v**2 + nodes_w**2  # (N,)

    ##############################
    # Orthogonalize ksi_5
    ##############################
    for j in range(4):
        inner = np.sum(ksi_5 * ksi_array[j] * fM * gwts)
        ksi_5 -= inner * ksi_array[j]

    ##############################
    # Normalize ksi_5
    ##############################
    norm5 = np.sqrt(np.sum(ksi_5**2 * fM * gwts))
    ksi_5 = ksi_5 / norm5

    ##############################
    # Combine
    ##############################
    ksis = np.vstack([ksi_array, ksi_5])  # (5, N)
    ksis = ksis.T  # (N, 5)

    return ksis


def load_moments_file(filename):
    """
    Loads a moments txt file (produced or saved) and skips any header lines.
    Returns a numpy array of numeric data.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    # find first line that starts with a number
    start_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped[0].isdigit() or stripped[0] in ['-','.']:
            start_idx = i
            break

    data = np.loadtxt(lines[start_idx:], delimiter=',')
    return data


def plot_moments(produced_moments_txt, saved_moments_txt, moment1, moment2, run_ID, CNN_model, cons, lyap, lyap_entrop):
    matplotlib.use("TkAgg")   # Force Tk backend BEFORE importing pyplot

    import matplotlib.pyplot as plt
    import numpy as np
    plt.ion()
    """
    Compare any two selected moments between:
      - produced moments (model output)
      - ground truth saved moments (true solution)
    """

    # ----------------------------
    # Load files
    # ----------------------------
    f1_np = load_moments_file(produced_moments_txt)
    f2_np = load_moments_file(saved_moments_txt)

    # ----------------------------
    # Map column names
    # ----------------------------
    moment_map = {
        "Time": 0,
        "Density": 1,
        "U": 2, "V": 3, "W": 4,
        "T": 5, "Tx": 6, "Ty": 7, "Tz": 8,
        "mom3u": 9, "mom3v": 10, "mom3w": 11,
        "mom4u": 12, "mom4v": 13, "mom4w": 14,
        "mom5u": 15, "mom5v": 16, "mom5w": 17,
        "mom6u": 18, "mom6v": 19, "mom6w": 20
    }

    col1 = moment_map[moment1]
    col2 = moment_map[moment2]

    # ----------------------------
    # Extract
    # ----------------------------
    time1 = f1_np[:, 0]
    time2 = f2_np[:, 0]

    y1_1 = f1_np[:, col1]   # produced
    y1_2 = f1_np[:, col2]

    y2_1 = f2_np[:, col1]   # ground truth
    y2_2 = f2_np[:, col2]

    # ----------------------------
    # Plot
    # ----------------------------
    plt.figure(figsize=(10, 6))
    plt.suptitle(f"Run {run_ID}: {moment1} vs {moment2}", fontsize=14)

    # OPTIONAL: Set OS window title (works on macOS, Windows & Linux)
    plt.gcf().canvas.manager.set_window_title(f"Run {run_ID} for {CNN_model} with {moment1} vs {moment2} H{int(cons)}L{int(lyap)}LE{int(lyap_entrop)}")


    plt.plot(time1, y1_1, '--', label=f"Produced {moment1}")
    plt.plot(time1, y1_2, '--', label=f"Produced {moment2}")

    plt.plot(time2, y2_1, label=f"Ground Truth {moment1}")
    plt.plot(time2, y2_2, label=f"Ground Truth {moment2}")

    plt.xlabel("Time")
    plt.ylabel("Moment Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.pause(0.001)

def parse_run_config(config_path):
    """
    Parses a run_config.txt file into a dict of {param: value}.
    Supports lines like `key = value` or `key: value`.
    """
    params = {}
    try:
        with open(config_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("=") or line.startswith("-"):
                    continue
                # match either "key = value" or "key: value"
                match = re.match(r"([\w\s\(\)/%-]+?)\s*[:=]\s*(.+)", line)
                if match:
                    key, value = match.groups()
                    key = key.strip().replace(" ", "_").replace("(", "").replace(")", "")
                    value = value.strip()
                    # try to interpret numeric values
                    try:
                        if "." in value or "e" in value.lower():
                            params[key] = float(value)
                        else:
                            params[key] = int(value)
                    except ValueError:
                        params[key] = value
##        print(f"Parsed {len(params)} parameters from {config_path}")
    except FileNotFoundError:
        print(f"⚠️ No run_config.txt found at {config_path}")
    return params


def parse_ae_metadata(ae_folder):
    """
    Attempts to extract AE training metadata from either:
      (1) run_config.txt (new AE models)
      (2) folder name (old AE models)
    Returns a dict with keys: 'timestamp', 'hidden_layers', 'code_len'
    """
    info = {"timestamp": None, "hidden_layers": None, "code_len": None, "dropout": None}

    run_config_path = os.path.join(ae_folder, "run_config.txt")
    if os.path.exists(run_config_path):
        # === New AE model format ===
        with open(run_config_path, "r") as f:
            for line in f:
                if "Hidden_Layers:" in line:
                    info["hidden_layers"] = int(line.split(":")[1].strip())
                elif "Code_Length:" in line:
                    info["code_len"] = int(line.split(":")[1].strip())
                elif "Dropout_Rate:" in line:
                    info["dropout"] = float(line.split(":")[1].strip())
                elif "Timestamp:" in line:
                    info["timestamp"] = line.split(":", 1)[1].strip()
    else:
        # === Legacy folder name format: delf-AE-mm_dd_yyyy-hh_mm_ss ===
        folder_name = os.path.basename(ae_folder)
        match = re.search(r"(\d{2}_\d{2}_\d{4}-\d{2}_\d{2}_\d{2})", folder_name)
        if match:
            info["timestamp"] = match.group(1)
        # Attempt to infer CL/HL from any folder naming convention e.g. "CL64_HL3"
        if "CL" in folder_name:
            cl_match = re.search(r"CL(\d+)", folder_name)
            if cl_match:
                info["code_len"] = int(cl_match.group(1))
        if "HL" in folder_name:
            hl_match = re.search(r"HL(\d+)", folder_name)
            if hl_match:
                info["hidden_layers"] = int(hl_match.group(1))

    return info

def str2bool(s):
    if isinstance(s, bool):
        return s
    if isinstance(s, str):
        return s.strip().lower() in ("true", "1", "yes")
    return bool(s)

def prepare_input(sol_data_raw, ae_model, solsize, enc_dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df_tensor = torch.tensor(sol_data_raw, dtype=torch.float64).to(device)
    with torch.no_grad():
        encoded = ae_model.encoder(df_tensor)
    return encoded.view(-1, 1, enc_dim, enc_dim, enc_dim), df_tensor

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def str2bool(s):
    if isinstance(s, bool):
        return s
    if isinstance(s, str):
        return s.strip().lower() in ("true", "1", "yes")
    return bool(s)

def hermite_projection(pred, fM, ksis, nodes_gwts):
    # pred: (1, N)
    # fM:   (N, 1) or (N,)
    # ksis: (1, N, 5)
    # gwts: (N,) or (N,1)

    # Convert to torch tensors (float64 for Hermite precision)
    fM = torch.tensor(fM, dtype=torch.float64)
    gwts = torch.tensor(nodes_gwts, dtype=torch.float64)

    B, N = pred.shape

    # Fix shapes to unify batch dimension
    pred = pred.view(B, N, 1)      # (B, N, 1)
    fM = fM.view(1, N, 1)          # (1, N, 1)
    gwts = gwts.view(1, N, 1)      # (1, N, 1)

    inp = pred.clone()             # (B, N, 1)

    for j in range(5):
        kj = ksis[:, :, j:j+1]     # (B, N, 1)

        # Inner product over N
        ip = torch.sum(pred * kj * gwts, dim=1, keepdim=True)  # (B, 1, 1)

        # Projection term
        proj = fM * kj * ip        # (B, N, 1)

        # Subtract projection
        inp = inp - proj           # (B, N, 1)

    return inp.squeeze(-1)         # back to (B, N)

def hermite_projection_np(pred, fM, ksis, nodes_gwts):
    """
    pred: (N,)
    fM:   (N,)
    ksis: (N, 5)
    nodes_gwts: (N,)

    Returns:
        projected pred: (N,)
    """
    import numpy as np

    # Ensure 1D
    pred = pred.reshape(-1)
    fM   = fM.reshape(-1)
    gwts = nodes_gwts.reshape(-1)

    inp = pred.copy()

    for j in range(5):
        kj = ksis[:, j]   # (N,)

        # Inner product <pred, kj>
        ip = np.sum(pred * kj * gwts)

        # Projection term
        proj = fM * kj * ip

        # Subtract projection
        inp = inp - proj

    return inp

def relu(x):
    import numpy as np
    return(np.maximum(0, x))

def huber_np(x, d):
    x = np.asarray(x)
    return np.where(
        x <= 0,
        0,
        np.where(
            x < d,
            x**2 / (2*d),
            x - d/2
        )
    )

def huber_torch(y, d):
    """
    Implements:
    s(y) = 0                    if y <= 0
           y^2 / (2d)           if 0 < y < d
           y - d/2              if y >= d
    """
    zero = torch.zeros_like(y)

    quad = y**2 / (2.0 * d)
    linear = y - d / 2.0

    return torch.where(
        y <= 0,
        zero,
        torch.where(y < d, quad, linear)
    )

def get_encoded_delta_f(sol, maxwell, AE_model, Use_AE, dms_enc, dms, device):
    import torch
    df = sol - maxwell
    df = torch.tensor(df, dtype=torch.float64).to(device)
    if Use_AE:
        df = AE_model.encoder(df)
        return df.view(1,1,dms_enc,dms_enc,dms_enc)
    return df.view(1,1,dms,dms,dms)

def load_macro_file(filepath):
    data = []

    with open(filepath, "r") as f:
        import numpy as np
        for line in f:
            line = line.strip()

            # skip header / broken lines
            if not line:
                continue

            try:
                row = [float(x) for x in line.split(",")]
            except:
                continue

            # enforce consistent row length (21 columns expected)
            if len(row) == 21:
                data.append(row)

    if len(data) == 0:
        raise ValueError(f"No valid data rows found in {filepath}")

    return np.array(data)
