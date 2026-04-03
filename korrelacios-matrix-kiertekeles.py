# IMPORTS

import csv
import matplotlib.pyplot as plt
from nilearn.connectome import ConnectivityMeasure
import numpy as np
import pandas as pd
from scipy import signal
import seaborn as sns
from sklearn.cluster import KMeans

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

# Heatmap for visualizing correlation matrix
def heatmap(corr_matrix):
    plt.figure(figsize=(5,5))
    sns.heatmap(corr_matrix, cmap="jet", vmin=-1, vmax=1)
    plt.title("Correlation matrix")
    plt.show()
'''
heatmap(corr_matrix_2D)
'''

# K-MEANS CLUSTERING

# K-means clustering is an unsupervised machine learning algorithm used in neuroimaging to categorize data into
# distinct, non-overlapping sets based on similarity. In the context of functional ultrasound (fUS), it is primarily
# applied to classify response patterns and perform automatic brain parcellation. This allows for a data-driven
# definition of brain structures rather than relying solely on anatomical atlases.

def k_means_clustering(corr_matrix, labels, num_clusters, filename):

    # Initialize K-means model
    kmeans = KMeans(n_clusters = num_clusters, random_state = 42)
    # Fit K-means model (to rows)
    clusters = kmeans.fit_predict(corr_matrix)

    # Open file to save data in
    file = open(filename, "w")

    # Create 2D array storing labels by clusters
    labels_by_clusters = []
    for i in range(num_clusters):
        labels_i = []
        for j in range(len(clusters)):
            if clusters[j] == i:
                labels_i.append(labels[j])
        labels_by_clusters.append(labels_i)

        # Write to file
        file.write(str(i+1))
        for j in range(len(labels_by_clusters[i])):
            file.write('\t' + labels_by_clusters[i][j])
        file.write('\n')

    # Close file
    file.close()
'''
print('\nName file (K-means Clustering) :')
filename = str(input()) + '.txt'
k_means_clustering(corr_matrix_2D, labels_2D, 4, filename)
'''

# SPECTRAL COHERENCE ANALYSIS

# Spectral coherence analysis is a frequency-domain method used to evaluate the consistency of the relationship
# between two signals — specifically, how well they correlate at specific frequencies. In the context of
# functional ultrasound (fUS), it is used to investigate resting-state functional connectivity by determining if
# different brain regions share synchronized fluctuations in cerebral blood volume (CBV).

def spectral_coherence_analysis(data_matrix, regionA, regionB, sampling_freq = 15000000):
    f, Cxy = signal.coherence(data_matrix[:,regionA], data_matrix[:,regionB], fs = sampling_freq, nperseg = 256) # nperseg defines the frequency resolution

    return f, Cxy

# Visualize result
def spectral_coherence_analysis_plot(regionA, regionB, f, Cxy, labels):
    plt.semilogy(f, Cxy) # logarithmic y axis

    title = 'Spectral Coherence between Brain Regions: ' + labels[regionA] + ', ' + labels[regionB]
    plt.title(title)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Coherence')
    plt.grid()
    plt.show()

'''
regionA = 0
regionB = 6
f, Cxy = spectral_coherence_analysis(data_matrix_2D, regionA, regionB)
spectral_coherence_analysis_plot(regionA, regionB, f, Cxy, labels_2D)
'''

# USER INTERFACE

def runtime():
    
    # Choice
    print('\nPick one of the following options:')

    print('\n1. Provide timestamped data file')
    print('2. Quit\n')
    
    userinput = 0
    while (userinput == 0):
        userinput = int(input())

        if (userinput == 1):
            continue
        elif (userinput == 2):
            return
        else:
            userinput = 0

    # Get fUS data file to work with
    f = None
    while (f == None):
        print("\nEnter FILE NAME of .txt file with timestamped fUS data:")
        filename = str(input()) + '.txt'

        try:
            f = open(filename, "r")
            f.close()
        except FileNotFoundError:
            print("File does not exist")
        except IOError:
            print("Error opening file")

runtime()