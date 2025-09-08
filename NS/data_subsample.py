import numpy as np

K = 16
nx = 100

vorticity = np.load(f"./data/Random_NS_trunc_omega_{K}_{nx}.npy")
vorticity_T = vorticity[-1, :, :, :]

np.save(f"./data/Random_NS_trunc_omegaT_{K}_{nx}.npy", vorticity_T)