import numpy as np

# Load the matrices from the .npy files
tvec_marker_difference = np.load("difference_mtx/difference_tvec_matrix.npy")
rvec_marker_difference = np.load("difference_mtx/difference_rvec_matrix.npy")

# Reshape the 3D arrays into 2D arrays
tvec_marker_difference_reshaped = tvec_marker_difference.reshape(-1, tvec_marker_difference.shape[-1])
rvec_marker_difference_reshaped = rvec_marker_difference.reshape(-1, rvec_marker_difference.shape[-1])

# Write the matrices to a final.txt file
with open("final.txt", "w") as f:
    f.write("Translation Vector Difference (tvec_marker_difference):\n")
    np.savetxt(f, tvec_marker_difference_reshaped, fmt='%.6f')
    f.write("\nRotation Vector Difference (rvec_marker_difference):\n")
    np.savetxt(f, rvec_marker_difference_reshaped, fmt='%.6f')


print("Matrices have been written to final.txt.")
