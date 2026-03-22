# READ TIMESTAMPED DATA

import numpy as np
import csv

def load_data(file_name):
    # Read labels (first row, skip first column)
    with open(file_name, "r") as f:
        reader = csv.reader(f)
        labels = next(reader)[1:] # (handles quoted commas, reads only brain regions)

    # Read numeric data (skip header automatically here)
    data = np.loadtxt(file_name, delimiter=",", skiprows=1)

    # Split into time and matrix
    timestamp = data[:, 0]
    data_matrix = data[:, 1:]

    return labels, timestamp, data_matrix

labels_2D, timestamp_2D, data_matrix_2D = load_data("0s_to_600.024s_2D_Matrix.txt")
labels_2D_wbc, timestamp_2D_wbc, data_matrix_2D_wbc = load_data("0s_to_600.024s_2D_Signals_baselinecorrect_nelkul.txt")
labels_2D_f, timestamp_2D_f, data_matrix_2D_f = load_data("0s_to_600.024s_Filtered2D_Matrix.txt")
labels_2D_f_wbc, timestamp_2D_f_wbc, data_matrix_2D_f_wbc = load_data("0s_to_600.024s_Filtered2D_Signals_baselinecorrect_nelkul.txt")
labels_4D_sbs, timestamp_4D_sbs, data_matrix_4D_sbs = load_data("0s_to_600.416s_4D_Matrix_slicebysliceinterp_slice.txt")
labels_4D_f_sbs, timestamp_4D_f_sbs, data_matrix_4D_f_sbs = load_data("0s_to_600.416s_Filtered4D_Matrix_slicebysliceinterp_slice.txt")

# FORM CORRELATION MATRIX