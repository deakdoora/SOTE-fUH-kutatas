# IMPORTS

import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# READ TIMESTAMPED DATA

def load_data(file_name):
    # List variable
    data_list = []

    # Read labels (first row, skip first column)
    with open(file_name, "r") as f:
        reader = csv.reader(f)
        labels = next(reader)[1:] # (handles quoted commas, reads only brain regions)

        # Read numeric data rows (skip header)
        for row in reader:
            processed_row = []
            for value in row:
                if value.strip() == "":   # empty field
                    processed_row.append(np.nan)
                else:
                    processed_row.append(float(value))
            data_list.append(processed_row)

    # Convert to numpy array
    data = np.array(data_list)

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

corr_matrix_2D = pd.DataFrame(data = data_matrix_2D, columns = labels_2D).corr()
corr_matrix_2D_wbc = pd.DataFrame(data = data_matrix_2D_wbc, columns = labels_2D_wbc).corr()
corr_matrix_2D_f = pd.DataFrame(data = data_matrix_2D_f, columns = labels_2D_f).corr()
corr_matrix_2D_f_wbc = pd.DataFrame(data = data_matrix_2D_f_wbc, columns = labels_2D_f_wbc).corr()
corr_matrix_4D_sbs = pd.DataFrame(data = data_matrix_4D_sbs, columns = labels_4D_sbs).corr()
corr_matrix_4D_f_sbs = pd.DataFrame(data = data_matrix_4D_f_sbs, columns = labels_4D_f_sbs).corr()

'''
# Heatmap for visualizing correlation matrix
plt.figure(figsize=(5,5))
sns.heatmap(corr_matrix_2D, cmap="turbo", vmin=-1, vmax=1, linewidths=0.5)
plt.title("Correlation matrix")
plt.show()
'''