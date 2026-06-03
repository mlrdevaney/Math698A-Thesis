import numpy as np
import pickle

sol_euler_data = []
for n in range(5):
    with open(f"Euler_Convergence_488_0p1_ts-{0.002/(2**n)}_H_1_L_0_LE_1.pkl", 'rb') as f:
        data = pickle.load(f)
        sol_euler_data.append(data)
sol_euler_data = np.array(sol_euler_data)
sol_euler_data = sol_euler_data.squeeze(1)

# Compute relative L2 differences vs finest solution (n=4)
f_out = open('Euler_Convergence_Results.txt', 'w')
f_out.write("n_2**n rel_err\n")
for i in range(4):
    print(sol_euler_data[i].shape)
    rel_err = np.linalg.norm(sol_euler_data[4] - sol_euler_data[i]) / np.linalg.norm(sol_euler_data[4])
    f_out.write(f"{i}      {rel_err:.8f}\n")
f_out.close()
print('Wrote Euler_Convergence_Results.txt')
