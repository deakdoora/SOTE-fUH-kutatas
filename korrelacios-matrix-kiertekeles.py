# IMPORTS

import csv
import matplotlib.pyplot as plt
import networkx as nx
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
print('\nNumber of clusters:')
num_clusters = int(input())

k_means_clustering(corr_matrix_2D, labels_2D, num_clusters, filename)
'''

# SPECTRAL COHERENCE ANALYSIS

# Spectral coherence analysis is a frequency-domain method used to evaluate the consistency of the relationship
# between two signals — specifically, how well they correlate at specific frequencies. In the context of
# functional ultrasound (fUS), it is used to investigate resting-state functional connectivity by determining if
# different brain regions share synchronized fluctuations in cerebral blood volume (CBV).

def spectral_coherence_analysis(data_matrix, regionA, regionB, sampling_freq = 15000000):
    f, Cxy = signal.coherence(data_matrix[:,regionA], data_matrix[:,regionB], fs = sampling_freq, nperseg = 256) # nperseg defines the frequency resolution

    return f, Cxy

def spectral_coherence_analysis_file(filename, f, Cxy):
    # Open file
    file = open(filename, "w")

    # Write to file
    file.write('f\tCxy\n')
    for j in range(len(f)):
        file.write(str(f[j]) + '\t' + str(Cxy[j]) + '\n')

    # Close file
    file.close()

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

# GRAPH

def graph(corr_matrix, thr):
    # Base for graph
    network_graph = nx.Graph()

    # Nodes & edges
    for i in corr_matrix.columns:
        for j in corr_matrix.columns:
            if (i != j and abs(corr_matrix.loc[i, j]) > thr):
                network_graph.add_edge(i, j, weight=corr_matrix.loc[i, j])

    return network_graph

def graph_plot(graph, thr):
    # Visualization
    nodes = nx.spring_layout(graph, seed = 42)
    edges = graph.edges(data = True)
    edge_widths = [abs(data['weight'])*3 for _, _, data in edges]

    nx.draw(graph, nodes, with_labels = True, node_color = 'skyblue', node_size = 2000, width = edge_widths)
    plt.show()

# USER INTERFACE

def runtime():
    
    userinput = 0
    while (userinput != 6):

        # Choice
        print('\nPick one of the following options:')

        print('\n1. Provide timestamped data file')
        print('2. Quit\n')
        
        userinput = 0
        while (userinput == 0):
            userinput = int(input())
            match userinput:
                case 1:
                    pass
                case 2:
                    return
                case _:
                    print("Choose from above:")
                    userinput = 0

        # Get fUS data file to work with
        f = None
        while (f == None):
            print("\nEnter FILE NAME of .txt file with timestamped fUS data:")
            filename = str(input()) + '.txt'    # use 0s_to_600.024s_2D_Matrix for testing

            try:
                f = open(filename, "r")
                f.close()
            except FileNotFoundError:
                print("File does not exist")
            except IOError:
                print("Error opening file")

        # Load data & form correlation matrix
        labels, timestamp, data_matrix = load_data(filename)
        corr_matrix = pd.DataFrame(data = data_matrix, columns = labels).corr()

        # Choice
        userinput = 0
        options = True
        while (userinput != 5 and userinput != 6):

            if (options == True):
                print('\nPick one of the following options:')

                print('\n1. Correlation Matrix')
                print('2. K-means Clustering')
                print('3. Spectral Coherence Analysis')
                print('4. Graph')
                print('5. Provide new data file')
                print('6. Quit\n')

                options = False

            userinput = int(input())
            match userinput:
                case 1:    # Heatmap

                    heatmap(corr_matrix)

                    options = True

                case 2:    # K-means Clustering

                    print('\nName file to write in:')
                    filename_kmeans = str(input()) + '.txt'
                    print('\nNumber of clusters:')
                    num_clusters = int(input())

                    k_means_clustering(corr_matrix, labels, num_clusters, filename_kmeans)

                    options = True

                case 3:    # Spectral Coherence Analysis
                    
                    print('\nRegion A:')
                    regionA = -1
                    while (regionA < 0 or regionA >= len(labels)):
                        regionA = int(input())
                    print('\nRegion B :')
                    regionB = -1
                    while (regionB < 0 or regionB >= len(labels)):
                        regionB = int(input())
                    print('\nName file to write in:')
                    filename_spectral = str(input()) + '.txt'

                    f, Cxy = spectral_coherence_analysis(data_matrix, regionA, regionB)
                    spectral_coherence_analysis_file(filename_spectral, f, Cxy)
                    spectral_coherence_analysis_plot(regionA, regionB, f, Cxy, labels)

                    options = True
                
                case 4:    # Graph
                    
                    print('\nThreshold:')
                    thr = 10
                    while (thr < -1 or thr > 1): # absolute correlation
                        thr= float(input())

                    network_graph = graph(corr_matrix, thr)
                    graph_plot(network_graph, thr)

                    options = True

                case 5:    # New data file
                    pass
                case 6:    # Quit
                    return
                case _:
                    print("Choose from above:")
                    userinput = 0

runtime()    # use 0s_to_600.024s_2D_Matrix for testing