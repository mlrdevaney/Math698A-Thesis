import pickle
import numpy as np

# Load the singular values once
sv_file = "singular_values.pkl"
with open(sv_file, "rb") as f:
    S = np.array(pickle.load(f))
print("Number of Singular Values:", len(S))
denom = np.sum(S)  # total sum of singular values

# Loop over thresholds n = -2 through -6
for n in range(-2, -7, -1):  # -2, -3, -4, -5, -6
    threshold = 10**n
    K_found = None
    for K in range(len(S)):
        num = np.sum(S[K:])       # sum of tail singular values
        e = num / denom
        if e < threshold:
            K_found = K
            print(f"For threshold 10^{n}: first K = {K_found}, e = {e:.3e}")
            break
    if K_found is None:
        print(f"For threshold 10^{n}: no K satisfies e < {threshold}")

# plot
import matplotlib.pyplot as plt

plt.figure()
plt.plot(np.arange(1, len(S)+1), S, marker='o')
plt.xlabel("Index")
plt.ylabel("Singular Value")
plt.title("Decay of Singular Values")
plt.yscale("log")
plt.grid(True)
plt.show()
