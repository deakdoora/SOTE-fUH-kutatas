# IMPORTS

from collections import Counter
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
    sns.heatmap(corr_matrix, square = True, cmap="jet", vmin=-1, vmax=1)
    plt.title("Correlation matrix")
    #plt.subplots_adjust(bottom = 0.5)
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
            if (i != j and corr_matrix.loc[i, j] > thr): #abs(corr_matrix.loc[i, j])
                network_graph.add_edge(i, j, weight=corr_matrix.loc[i, j])

    return network_graph

def graph_plot(graph):
    # Visualization
    nodes = nx.spring_layout(graph, seed = 42)
    edges = graph.edges(data = True)
    edge_widths = [abs(data['weight'])*3 for _, _, data in edges]

    nx.draw(graph, nodes, with_labels = True, node_color = 'skyblue', node_size = 2000, width = edge_widths)
    plt.show()

# GRAPH PARAMETRES

#node_ID = list(network_graph.nodes)[int(node)]
#node_name = network_graph.nodes[node_ID].get("name", str(node_ID))

# Basic structural parametres
def graph_nodes(network_graph):
    n = network_graph.number_of_nodes()
    print('Number of nodes:', n)
def graph_edges(network_graph):
    n = network_graph.number_of_edges()
    print('Number of edges:', n)
def graph_density(network_graph): # present / possible edges
    N = network_graph.number_of_nodes()
    E = network_graph.number_of_edges()
    n = 2*E / (N * (N-1))
    print('Graph density:', n)
# Node level metrics
def node_degree(network_graph): # edges of given node
    node = list(network_graph.nodes())
    degree = []
    for i in range(network_graph.number_of_nodes()):
        degree.append(network_graph.degree(node[i]))

    plt.scatter(node, degree)
    plt.title('Degree of nodes')
    plt.xlabel('Node')
    plt.ylabel('Degree')
    plt.xticks(rotation = 90)
    for i in range(len(node)): # add y value to each point
        plt.annotate(f"{degree[i]:.2f}", (node[i], degree[i]), textcoords="offset points", xytext=(0,5), ha='center')
    plt.ylim(top = np.max(degree) + (np.max(degree)-np.min(degree))*0.1) # for y values to fit on plot
    plt.subplots_adjust(bottom=0.5) # for x labels to fit on screen
    plt.show()
def degree_distribution(network_graph): # probability of a node having given number of edges
    n = network_graph.number_of_nodes()
    degree = np.linspace(0,n-1,n)
    probability = []
    for i in degree:
        node_num_with_degree_i = 0
        for j in range(n):
            node_ID = list(network_graph.nodes)[j]
            node_name = network_graph.nodes[node_ID].get("name", str(node_ID))
            if (network_graph.degree(node_name) == i):
                node_num_with_degree_i += 1
        probability.append(node_num_with_degree_i / n)
    
    plt.scatter(degree, probability)
    plt.title('Degree distribution')
    plt.xlabel('Degree')
    plt.ylabel('Probability')
    for i in range(len(degree)): # add y value to each point
        plt.annotate(f"{probability[i]:.2f}", (degree[i], probability[i]), textcoords="offset points", xytext=(0,5), ha='center')
    plt.show()
            
    #degrees = [i for n, i in network_graph.degree()]
    #num_nodes_by_degree = Counter(degrees)
    #dd = {k: v / n for k, v in num_nodes_by_degree.items()}
    #print(dd)
def clustering_coeff(network_graph): # present / possible edges of neighbours of given node
    node = list(network_graph.nodes())
    cc = []
    for i in range(network_graph.number_of_nodes()):
        cc.append(nx.clustering(network_graph, node[i]))

    plt.scatter(node, cc)
    plt.title('Clustering coefficient of nodes')
    plt.xlabel('Node')
    plt.ylabel('Clustering coefficient')
    plt.xticks(rotation = 90)
    for i in range(len(node)): # add y value to each point
        plt.annotate(f"{cc[i]:.2f}", (node[i], cc[i]), textcoords="offset points", xytext=(0,5), ha='center')
    plt.ylim(top = np.max(cc) + (np.max(cc)-np.min(cc))*0.1) # for y values to fit on plot
    plt.subplots_adjust(bottom=0.5) # for x labels to fit on screen
    plt.show()
def degree_centrality(network_graph): # popularity, normalized degree of given node
    dict = nx.degree_centrality(network_graph)
    node = list(dict.keys())
    dc = list(dict.values())

    plt.scatter(node, dc)
    plt.title('Degree centrality of nodes')
    plt.xlabel('Node')
    plt.ylabel('Degree centrality')
    plt.xticks(rotation = 90)
    for i in range(len(node)): # add y value to each point
        plt.annotate(f"{dc[i]:.2f}", (node[i], dc[i]), textcoords="offset points", xytext=(0,5), ha='center')
    plt.ylim(top = np.max(dc) + (np.max(dc)-np.min(dc))*0.1) # for y values to fit on plot
    plt.subplots_adjust(bottom=0.5) # for x labels to fit on screen
    plt.show()
def betweenness_centrality(network_graph): # control over information flow, how many shortest paths between nodes contain given node
    dict = nx.betweenness_centrality(network_graph)
    node = list(dict.keys())
    bc = list(dict.values())

    plt.scatter(node, bc)
    plt.title('Betweenness centrality of nodes')
    plt.xlabel('Node')
    plt.ylabel('Betweenness centrality')
    plt.xticks(rotation = 90)
    for i in range(len(node)): # add y value to each point
        plt.annotate(f"{bc[i]:.2f}", (node[i], bc[i]), textcoords="offset points", xytext=(0,5), ha='center')
    plt.ylim(top = np.max(bc) + (np.max(bc)-np.min(bc))*0.1) # for y values to fit on plot
    plt.subplots_adjust(bottom=0.5) # for x labels to fit on screen
    plt.show()
def closeness_centrality(network_graph): # speed of communication, how quickly are other nodes reachable from given
    dict = nx.closeness_centrality(network_graph)
    node = list(dict.keys())
    cc = list(dict.values())

    plt.scatter(node, cc)
    plt.title('Closeness centrality of nodes')
    plt.xlabel('Node')
    plt.ylabel('Closeness centrality')
    plt.xticks(rotation = 90)
    for i in range(len(node)): # add y value to each point
        plt.annotate(f"{cc[i]:.2f}", (node[i], cc[i]), textcoords="offset points", xytext=(0,5), ha='center')
    plt.ylim(top = np.max(cc) + (np.max(cc)-np.min(cc))*0.1) # for y values to fit on plot
    plt.subplots_adjust(bottom=0.5) # for x labels to fit on screen
    plt.show()
def eigenvector_centrality(network_graph): # well-connectedness, amount of inluential neighbouring nodes
    dict = nx.eigenvector_centrality(network_graph)
    node = list(dict.keys())
    ec = list(dict.values())

    plt.scatter(node, ec)
    plt.title('Eigenvector centrality of nodes')
    plt.xlabel('Node')
    plt.ylabel('Eigenvector centrality')
    plt.xticks(rotation = 90)
    for i in range(len(node)): # add y value to each point
        plt.annotate(f"{ec[i]:.2f}", (node[i], ec[i]), textcoords="offset points", xytext=(0,5), ha='center')
    plt.ylim(top = np.max(ec) + (np.max(ec)-np.min(ec))*0.1) # for y values to fit on plot
    plt.subplots_adjust(bottom=0.5) # for x labels to fit on screen
    plt.show()
# Path based metrics
def shortest_path_length(network_graph):
    l = []
    for i in range(network_graph.number_of_nodes()):
        l_i = []
        for j in range(network_graph.number_of_nodes()):
            s_ID = list(network_graph.nodes)[i]
            s_name = network_graph.nodes[s_ID].get("name", str(s_ID))
            t_ID = list(network_graph.nodes)[j]
            t_name = network_graph.nodes[t_ID].get("name", str(t_ID))

            l_i.append(nx.shortest_path_length(network_graph, source = s_name, target = t_name))
        l.append(l_i)
    l_matrix = pd.DataFrame(l, index = list(network_graph.nodes), columns = list(network_graph.nodes))

    sns.heatmap(l_matrix, square = True, annot = True, cmap = "Greys", cbar = False)
    plt.title("Shortest path lengths")
    #plt.subplots_adjust(bottom = 0.5)
    plt.show()
def weighted_shortest_path_length(network_graph):
    l = []
    for i in range(network_graph.number_of_nodes()):
        l_i = []
        for j in range(network_graph.number_of_nodes()):
            s_ID = list(network_graph.nodes)[i]
            s_name = network_graph.nodes[s_ID].get("name", str(s_ID))
            t_ID = list(network_graph.nodes)[j]
            t_name = network_graph.nodes[t_ID].get("name", str(t_ID))

            l_i.append(nx.shortest_path_length(network_graph, source = s_name, target = t_name, weight = 'weight'))
        l.append(l_i)
    l_matrix = pd.DataFrame(l, index = list(network_graph.nodes), columns = list(network_graph.nodes))

    sns.heatmap(l_matrix, square = True, annot = True, cmap = "Greys")
    plt.title("Shortest path lengths")
    #plt.subplots_adjust(bottom = 0.5)
    plt.show()
def shortest_path(network_graph, s, t):
    s_ID = list(network_graph.nodes)[s]
    s_name = network_graph.nodes[s_ID].get("name", str(s_ID))
    t_ID = list(network_graph.nodes)[t]
    t_name = network_graph.nodes[t_ID].get("name", str(t_ID))

    path = nx.shortest_path(network_graph, source = s_name, target = t_name)
    print(path)
# define new method here

# TEST RUNTIME

network_graph_2D = graph(corr_matrix_2D, 0.5)

# Basic structural parametres
#graph_nodes(network_graph_2D)
#graph_edges(network_graph_2D)
#graph_density(network_graph_2D)

# Node level metrics
#node_degree(network_graph_2D)
#degree_distribution(network_graph_2D)
#clustering_coeff(network_graph_2D)
#degree_centrality(network_graph_2D)
#betweenness_centrality(network_graph_2D)
#closeness_centrality(network_graph_2D)
#eigenvector_centrality(network_graph_2D)

# Path based metrics
#shortest_path_length(network_graph_2D)
#weighted_shortest_path_length(network_graph_2D)
#shortest_path(network_graph_2D, 4, 11)
# run new method here

# Choice
'''
print('\nPick one of the following options:')
print('\n1. Show graph')
print('2. Quit\n')
userinput = 0
while (userinput == 0):
    userinput = int(input())
    match userinput:
        case 1:
            graph_plot(network_graph_2D)
        case 2:
            pass
        case _:
            print("Choose from above:")
            userinput = 0
'''

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
                    while (thr < 0 or thr > 1): # absolute correlation
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
#def analysis(filename): # complete analysis of a measurement

#runtime()    # use 0s_to_600.024s_2D_Matrix for testing

# TO DO LIST

'''
- save everything to file
- integrate graph parametres into the runtime function
- full analysis function
- test for 4D data
'''