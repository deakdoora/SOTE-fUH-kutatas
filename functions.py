# IMPORTS °°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°

from collections import Counter
import csv
import datetime
import matplotlib.pyplot as plt
import networkx as nx
import networkx.algorithms.community as nxac
from nilearn.connectome import ConnectivityMeasure
import numpy as np
import pandas as pd
from pathlib import Path
import random
from scipy import signal
import seaborn as sns
from sklearn.cluster import KMeans
import glob

# FUNCTIONS °°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°

# DATA HANDLING .................................................................................................

def load_data(filepath):
    '''
    Recieves: filepath (signals of ROIs in time)
    Returns: labels (names of ROIs)
             timestamp (time instances)
             data_matrix (signal values of all ROIs in time)
    '''
    
    # List variable
    data_list = []

    # Read labels (first row, skip first column)
    with open(filepath, "r") as f:
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
'''
labels, timestamp, data_matrix = load_data('sub-1663/*fus2D.txt')
'''

def correlation_matrix(data_matrix, labels):
    '''
    Recieves: data_matrix (signal values of all ROIs in time)
              labels (names of ROIs)
    Returns: corr_matrix (correlation matrix)
    '''
    
    corr_matrix = pd.DataFrame(data = data_matrix, columns = labels).corr()
    return corr_matrix
def k_means_clustering(corr_matrix, labels):
    '''
    Recieves: corr_matrix (correlation matrix)
              labels (names of ROIs)
              num_clusters (# of clusters to distribute ROIs into)
    Returns: k_means_clusters (names of ROIs serated into n # of clusters where 1 < n < # of ROIs)
    '''

    k_means_clusters = []
    for num_clusters in range(2, len(labels)-1):
        # Initialize K-means model
        kmeans = KMeans(n_clusters = num_clusters, random_state = 42)
        # Fit K-means model (to rows)
        clusters = kmeans.fit_predict(corr_matrix)

        # Create 2D array storing labels by clusters
        labels_by_clusters = []
        for i in range(num_clusters):
            labels_i = []
            for j in range(len(clusters)):
                if clusters[j] == i:
                    labels_i.append(labels[j])
            labels_by_clusters.append(labels_i)
        k_means_clusters.append(labels_by_clusters)

    return k_means_clusters
def spectral_coherence_analysis(data_matrix, labels, sampling_freq = 15000000):
    '''
    Recieves: data_matrix (signal values of all ROIs in time)
              labels (names of ROIs)
              sampling_freq (sampling frequency of fUS device)
    Returns: ROI_pair_s (names of ROI pairs)
             f_s (frequencies)
             Cxy_s (coherence)
    '''
    
    ROI_pair_s = []
    f_s = []
    Cxy_s = []
    for i in range(len(data_matrix[0])-1):
        for j in range(i+1, len(data_matrix[0])):
            ROI_pair_s.append(str(labels[i]) + ', ' + str(labels[j]))

            f, Cxy = signal.coherence(data_matrix[:,i], data_matrix[:,j], fs = sampling_freq, nperseg = 256) # nperseg defines the frequency resolution
            f_s.append(f)
            Cxy_s.append(Cxy)

    return ROI_pair_s, f_s, Cxy_s
def graph(corr_matrix, thr):
    '''
    Recieves: corr_matrix (correlation matrix)
              thr (minimum threshold for edge weights)
    Returns: network_graph (network graph)
    '''

    # Base for graph
    network_graph = nx.Graph()

    # Nodes & edges
    for i in corr_matrix.columns:
        for j in corr_matrix.columns:
            if (i != j and corr_matrix.loc[i, j] > thr): #abs(corr_matrix.loc[i, j])
                network_graph.add_edge(i, j, weight=corr_matrix.loc[i, j])

    return network_graph
'''
corr_matrix = correlation_matrix(data_matrix, labels)
k_means_clusters = k_means_clustering(corr_matrix, labels)
ROI_pair_s, f_s, Cxy_s = spectral_coherence_analysis(data_matrix, labels, sampling_freq = 15000000)
network_graph = graph(corr_matrix, thr)
'''

# GRAPH PROPERTIES ..............................................................................................
# node_ID = list(network_graph.nodes)[int(node)]
# node_name = network_graph.nodes[node_ID].get("name", str(node_ID))

# Basic structural parametres

def graph_nodes(network_graph):
    '''
    Recieves: network_graph (network graph)
    Returns: n (# of nodes)
    '''

    n = network_graph.number_of_nodes()
    return n
def graph_edges(network_graph):
    '''
    Recieves: network_graph (network graph)
    Returns: e (# of edges)
    '''

    e = network_graph.number_of_edges()
    return e
def graph_density(network_graph): # present / possible edges
    '''
    Recieves: network_graph (network graph)
    Returns: d (graph density)
    '''
    
    N = network_graph.number_of_nodes()
    E = network_graph.number_of_edges()
    d = 2*E / (N * (N-1))
    return d
'''
n = graph_nodes(network_graph)
e = graph_edges(network_graph)
d = graph_density(network_graph)
'''

# Node level metrics

def node_degree(network_graph): # edges of given node
    '''
    Recieves: network_graph (network graph)
    Returns: nodes (names of nodes)
             degrees (# of connections of nodes)
    '''
    
    nodes = list(network_graph.nodes())
    degrees = []
    for i in range(network_graph.number_of_nodes()):
        degrees.append(network_graph.degree(nodes[i]))

    return nodes, degrees
def degree_distribution(network_graph): # probability of a node having given number of edges
    '''
    Recieves: network_graph (network graph)
    Returns: degrees (# of connections)
             probabilities (probability of a node having given degree)
    '''
    
    n = network_graph.number_of_nodes()
    degrees = np.linspace(0,n-1,n)
    probabilities = []
    for i in degrees:
        node_num_with_degree_i = 0
        for j in range(n):
            node_ID = list(network_graph.nodes)[j]
            node_name = network_graph.nodes[node_ID].get("name", str(node_ID))
            if (network_graph.degree(node_name) == i):
                node_num_with_degree_i += 1
        probabilities.append(node_num_with_degree_i / n)

    return degrees, probabilities
def clustering_coeff(network_graph): # present / possible edges of neighbours of given node
    '''
    Recieves: network_graph (network graph)
    Returns: node (names of nodes)
             cc (clustering coefficients)
    '''
    
    node = list(network_graph.nodes())
    cc = []
    for i in range(network_graph.number_of_nodes()):
        cc.append(nx.clustering(network_graph, node[i]))

    return node, cc
def degree_centrality(network_graph): # popularity, normalized degree of given node
    '''
    Recieves: network_graph (network graph)
    Returns: node (names of nodes)
             dc (degree centralities)
    '''
    
    dict = nx.degree_centrality(network_graph)
    node = list(dict.keys())
    dc = list(dict.values())

    return node, dc
def betweenness_centrality(network_graph): # control over information flow, how many shortest paths between nodes contain given node
    '''
    Recieves: network_graph (network graph)
    Returns: node (names of nodes)
             bc (betweenness centralities)
    '''
    
    dict = nx.betweenness_centrality(network_graph)
    node = list(dict.keys())
    bc = list(dict.values())

    return node, bc
def closeness_centrality(network_graph): # speed of communication, how quickly are other nodes reachable from given
    '''
    Recieves: network_graph (network graph)
    Returns: node (names of nodes)
             cc (closeness centralities)
    '''
    
    dict = nx.closeness_centrality(network_graph)
    node = list(dict.keys())
    cc = list(dict.values())

    return node, cc
def eigenvector_centrality(network_graph): # well-connectedness, amount of inluential neighbouring nodes
    '''
    Recieves: network_graph (network graph)
    Returns: node (names of nodes)
             ec (eigenvector centralities)
    '''
    
    dict = nx.eigenvector_centrality(network_graph)
    node = list(dict.keys())
    ec = list(dict.values())

    return node, ec
'''
nodes, degrees = node_degree(network_graph)
degrees, probabilities = degree_distribution(network_graph)
node, cc = clustering_coeff(network_graph)
node, dc = degree_centrality(network_graph)
node, bc = betweenness_centrality(network_graph)
node, cc = closeness_centrality(network_graph)
node, ec = eigenvector_centrality(network_graph)
'''

# Path based metrics

def shortest_path_length(network_graph):
    '''
    Recieves: network_graph (network graph)
    Returns: l_matrix (matrix of shortest path lengths)
    '''
    
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

    return l_matrix
def weighted_shortest_path_length(network_graph):
    '''
    Recieves: network_graph (network graph)
    Returns: wl_matrix (matrix of weighted shortest path lengths)
    '''
    
    wl = []
    for i in range(network_graph.number_of_nodes()):
        wl_i = []
        for j in range(network_graph.number_of_nodes()):
            s_ID = list(network_graph.nodes)[i]
            s_name = network_graph.nodes[s_ID].get("name", str(s_ID))
            t_ID = list(network_graph.nodes)[j]
            t_name = network_graph.nodes[t_ID].get("name", str(t_ID))

            wl_i.append(nx.shortest_path_length(network_graph, source = s_name, target = t_name, weight = 'weight'))
        wl.append(wl_i)
    wl_matrix = pd.DataFrame(wl, index = list(network_graph.nodes), columns = list(network_graph.nodes))

    return wl_matrix
def shortest_path(network_graph): # actual path
    '''
    Recieves: network_graph (network graph)
    Returns: path (node names in order on the shortest path from source to target)
    '''
    
    path = []
    for s in range(len(network_graph.nodes)):
        for t in range(len(network_graph.nodes)):
            if (s != t):
                s_ID = list(network_graph.nodes)[s]
                s_name = network_graph.nodes[s_ID].get("name", str(s_ID))
                t_ID = list(network_graph.nodes)[t]
                t_name = network_graph.nodes[t_ID].get("name", str(t_ID))

                path.append(nx.shortest_path(network_graph, source = s_name, target = t_name))

    return path
def ave_path_length(network_graph): # average of shortest path lengths
    '''
    Recieves: network_graph (network graph)
    Returns: np.average(l) (average shortest path length)
    '''
    
    l = []
    for i in range(network_graph.number_of_nodes()):
        for j in range(network_graph.number_of_nodes()-1-i):
            s_ID = list(network_graph.nodes)[i]
            s_name = network_graph.nodes[s_ID].get("name", str(s_ID))
            t_ID = list(network_graph.nodes)[j+1+i]
            t_name = network_graph.nodes[t_ID].get("name", str(t_ID))

            l.append(nx.shortest_path_length(network_graph, source = s_name, target = t_name))
    
    return np.average(l)
def ave_weighted_path_length(network_graph): # average of weighted shortest path lengths
    '''
    Recieves: network_graph (network graph)
    Returns: np.average(wl) (average weighted shortest path length)
    '''
    
    wl = []
    for i in range(network_graph.number_of_nodes()):
        for j in range(network_graph.number_of_nodes()-1-i):
            s_ID = list(network_graph.nodes)[i]
            s_name = network_graph.nodes[s_ID].get("name", str(s_ID))
            t_ID = list(network_graph.nodes)[j+1+i]
            t_name = network_graph.nodes[t_ID].get("name", str(t_ID))

            wl.append(nx.shortest_path_length(network_graph, source = s_name, target = t_name, weight='weight'))
    
    return np.average(wl)
def diameter(network_graph): # longest shortest path length
    '''
    Recieves: network_graph (network graph)
    Returns: np.max(l) (diameter)
    '''
    
    l = []
    for i in range(network_graph.number_of_nodes()):
        for j in range(network_graph.number_of_nodes()-1-i):
            s_ID = list(network_graph.nodes)[i]
            s_name = network_graph.nodes[s_ID].get("name", str(s_ID))
            t_ID = list(network_graph.nodes)[j+1+i]
            t_name = network_graph.nodes[t_ID].get("name", str(t_ID))

            l.append(nx.shortest_path_length(network_graph, source = s_name, target = t_name))
    
    return np.max(l)
def weighted_diameter(network_graph): # longest shortest weighted path length
    '''
    Recieves: network_graph (network graph)
    Returns: np.max(wl) (weighted diameter)
    '''
    
    wl = []
    for i in range(network_graph.number_of_nodes()):
        for j in range(network_graph.number_of_nodes()-1-i):
            s_ID = list(network_graph.nodes)[i]
            s_name = network_graph.nodes[s_ID].get("name", str(s_ID))
            t_ID = list(network_graph.nodes)[j+1+i]
            t_name = network_graph.nodes[t_ID].get("name", str(t_ID))

            wl.append(nx.shortest_path_length(network_graph, source = s_name, target = t_name, weight='weight'))
    
    return np.max(wl)
'''
l_matrix = shortest_path_length(network_graph)
wl_matrix = weighted_shortest_path_length(network_graph)
path = shortest_path(network_graph)
ave_l = ave_path_length(network_graph)
ave_wl = ave_weighted_path_length(network_graph)
dia = diameter(network_graph)
wdia = weighted_diameter(network_graph)
'''

# Global network properties

def connected_components(network_graph): # islands
    '''
    Recieves: network_graph (network graph)
    Returns: n (# of connected components / islands)
             conn_comp (lists of connected components / islands)
    '''
    
    n = nx.number_connected_components(network_graph)
    conn_comp = list(nx.connected_components(network_graph))
    return n, conn_comp
def giant_component(network_graph): # largest island
    '''
    Recieves: network_graph (network graph)
    Returns: giant (list of largest set of connected components / island)
    '''
    
    giant = max(nx.connected_components(network_graph), key = len)
    return giant
def modularity(network_graph): # how well the graph separates into islands (0: random, 0.3: meaningful structure, 0.5: strong islands)
    '''
    Recieves: network_graph (network graph)
    Returns: mod (modularity)
    '''
    
    islands = nxac.greedy_modularity_communities(network_graph)
    mod = nxac.modularity(network_graph, islands)
    return mod
def weighted_modularity(network_graph): # how well the weighted graph separates into islands
    '''
    Recieves: network_graph (network graph)
    Returns: wmod (weighted modularity)
    '''
    
    islands = nxac.greedy_modularity_communities(network_graph)
    wmod = nxac.modularity(network_graph, islands, weight = 'weight')
    return wmod
def assortativity(network_graph): # network mixing pattern, connectivity of similar nodes (-1 to 1, 0: random)
    '''
    Recieves: network_graph (network graph)
    Returns: a (assortativity)
    '''
    
    a = nx.degree_assortativity_coefficient(network_graph)
    return a
'''
n, conn_comp = connected_components(network_graph)
giant = giant_component(network_graph)
mod = modularity(network_graph)
wmod = weighted_modularity(network_graph)
a = assortativity(network_graph)
'''

# Flow and robustness

def network_efficiency(network_graph): # how easily and quickly information spreads
    '''
    Recieves: network_graph (network graph)
    Returns: ne (network efficiency)
    '''
    
    n = network_graph.number_of_nodes()

    ne = 1 / (n * (n-1))
    inv_d = 0
    for i in range(n):
        for j in range(n):
            if (i != j):
                s_ID = list(network_graph.nodes)[i]
                s_name = network_graph.nodes[s_ID].get("name", str(s_ID))
                t_ID = list(network_graph.nodes)[j]
                t_name = network_graph.nodes[t_ID].get("name", str(t_ID))

                inv_d += 1 / nx.shortest_path_length(network_graph, source = s_name, target = t_name)
    ne = ne * inv_d

    return ne
def weighted_network_efficiency(network_graph):
    '''
    Recieves: network_graph (network graph)
    Returns: wne (weighted network efficiency)
    '''
    
    n = network_graph.number_of_nodes()

    wne = 1 / (n * (n-1))
    w_inv_d = 0
    for i in range(n):
        for j in range(n):
            if (i != j):
                s_ID = list(network_graph.nodes)[i]
                s_name = network_graph.nodes[s_ID].get("name", str(s_ID))
                t_ID = list(network_graph.nodes)[j]
                t_name = network_graph.nodes[t_ID].get("name", str(t_ID))

                w_inv_d += 1 / nx.shortest_path_length(network_graph, source = s_name, target = t_name, weight = 'weight')
    wne = wne * w_inv_d

    return wne
def robustness_to_random_failure(network_graph, f): # resilience to failure (node / edge removal)
    '''
    Recieves: network_graph (network graph)
              f (output file variable)
    Returns: (saves random failure robustness test results to file)
    '''
    
    # Robustness test
    f.write('Robustness test to random failure')

    f.write('\n\nNode removal\n\n')
    f.write('network efficiency\tnumber of connected components\tsize of largest island\taverage of shortest path lengths\taverage of weighted shortest path lengths\n')

    ng = network_graph.copy()
    nodes = list(ng.nodes())
    random.shuffle(nodes)
    for node in nodes:
        ng.remove_node(node) # random failure
        if len(list(ng.nodes())) > 0:
            network_efficiency = nx.global_efficiency(ng)
            num_connected_components = nx.number_connected_components(ng)
            largest_island_size = len(max(nx.connected_components(ng), key = len))
            ng_sub = ng.subgraph(max(nx.connected_components(ng), key = len))
            ave_shortest_path_length = nx.average_shortest_path_length(ng_sub)
            ave_weighted_shortest_path_length = nx.average_shortest_path_length(ng_sub, weight = 'weight')

            f.write(str(network_efficiency) + '\t' + str(num_connected_components) + '\t' + str(largest_island_size) + '\t' + str(ave_shortest_path_length) + '\t' + str(ave_weighted_shortest_path_length) + '\n')

    f.write('\n\nEdge removal\n\n')
    f.write('network efficiency\tnumber of connected components\tsize of largest island\taverage of shortest path lengths\taverage of weighted shortest path lengths\n')

    ng = network_graph.copy()
    edges = list(ng.edges())
    random.shuffle(edges)
    for edge in edges:
        ng.remove_edge(edge[0], edge[1]) # random failure
        if len(list(ng.edges())) > 0:
            network_efficiency = nx.global_efficiency(ng)
            num_connected_components = nx.number_connected_components(ng)
            largest_island_size = len(max(nx.connected_components(ng), key = len))
            ng_sub = ng.subgraph(max(nx.connected_components(ng), key = len))
            ave_shortest_path_length = nx.average_shortest_path_length(ng_sub)
            ave_weighted_shortest_path_length = nx.average_shortest_path_length(ng_sub, weight = 'weight')

            f.write(str(network_efficiency) + '\t' + str(num_connected_components) + '\t' + str(largest_island_size) + '\t' + str(ave_shortest_path_length) + '\t' + str(ave_weighted_shortest_path_length) + '\n')
def robustness_to_targeted_attack(network_graph, f): # resilience to failure (node / edge removal)
    '''
    Recieves: network_graph (network graph)
              f (output file variable)
    Returns: (saves targeted failure robustness test results to file)
    '''
    
    # Robustness test
    f.write('Robustness test to targeted attack')

    f.write('\n\nHighest ranking node removal\n\n')
    f.write('network efficiency\tnumber of connected components\tsize of largest island\taverage of shortest path lengths\taverage of weighted shortest path lengths\n')

    ng = network_graph.copy()
    n = len(list(ng.nodes()))
    for i in range(n):
        node = max(ng.degree, key = lambda x: x[1])[0] # highest degree node
        ng.remove_node(node) # targetted attack
        if len(list(ng.nodes())) > 0:
            network_efficiency = nx.global_efficiency(ng)
            num_connected_components = nx.number_connected_components(ng)
            largest_island_size = len(max(nx.connected_components(ng), key = len))
            ng_sub = ng.subgraph(max(nx.connected_components(ng), key = len))
            ave_shortest_path_length = nx.average_shortest_path_length(ng_sub)
            ave_weighted_shortest_path_length = nx.average_shortest_path_length(ng_sub, weight = 'weight')

            f.write(str(network_efficiency) + '\t' + str(num_connected_components) + '\t' + str(largest_island_size) + '\t' + str(ave_shortest_path_length) + '\t' + str(ave_weighted_shortest_path_length) + '\n')

    f.write('\n\nMost important bridge removal\n\n')
    f.write('network efficiency\tnumber of connected components\tsize of largest island\taverage of shortest path lengths\taverage of weighted shortest path lengths\n')

    ng = network_graph.copy()
    e = len(list(ng.edges()))
    for j in range(e):
        eb = nx.edge_betweenness_centrality(ng) # edge betweenness
        edge = max(eb, key = eb.get)
        ng.remove_edge(edge[0], edge[1]) # targetted attack
        if len(list(ng.edges())) > 0:
            network_efficiency = nx.global_efficiency(ng)
            num_connected_components = nx.number_connected_components(ng)
            largest_island_size = len(max(nx.connected_components(ng), key = len))
            ng_sub = ng.subgraph(max(nx.connected_components(ng), key = len))
            ave_shortest_path_length = nx.average_shortest_path_length(ng_sub)
            ave_weighted_shortest_path_length = nx.average_shortest_path_length(ng_sub, weight = 'weight')

            f.write(str(network_efficiency) + '\t' + str(num_connected_components) + '\t' + str(largest_island_size) + '\t' + str(ave_shortest_path_length) + '\t' + str(ave_weighted_shortest_path_length) + '\n')
# define new method here
'''
ne = network_efficiency(network_graph)
wne = weighted_network_efficiency(network_graph)
robustness_to_random_failure(network_graph, f)
robustness_to_targeted_attack(network_graph, f)
'''

# SAVING TO FILE ................................................................................................

def save_corr_matrix(corr_matrix, file):
    '''
    Recieves: corr_matrix (correlation matrix)
              file (output file variable)
    Returns: (saves correlation matrix data to file)
    '''
    
    file.write('CORRELATION MATRIX\n\n')
    file.write(corr_matrix.to_string())
    file.write('\n\n')
def save_k_means_clustering(k_means_clusters, file):
    '''
    Recieves: corr_matrix (correlation matrix)
              labels (names of ROIs)
              num_clusters (# of clusters to distribute ROIs into)
    Returns: (saves k means clustering into file)
    '''

    file.write('K-MEANS CLUSTERING\n\n')
    for k in range(len(k_means_clusters)):
        file.write(str(k+2) + ' clusters:\n')
        file.write('Cluster #\tElements\n')
        for i in range(len(k_means_clusters[k])):
            file.write(str(i+1))
            for j in range(len(k_means_clusters[k][i])):
                file.write('\t' + str(k_means_clusters[k][i][j]))
            file.write('\n')
        file.write('\n')
def save_spectral_coherence_analysis(ROI_pair_s, f_s, Cxy_s, file):
    '''
    Recieves: ROI_pair_s (names of ROI pairs)
              f_s (frequencies)
              Cxy_s (coherence)
              file (output file variable)
    Returns: (saves spectral coherence analysis data for all ROI pairs to file)
    '''

    file.write('SPECTRAL COHERENCE ANALYSIS\n\n')

    # Title
    file.write('Frequency')
    for k in range(len(ROI_pair_s)):
        file.write('\t' + ROI_pair_s[k])
    file.write('\n')

    # Data
    for j in range(len(f_s[0])):
        file.write(str(f_s[0][j]))
        for k in range(len(ROI_pair_s)):
            file.write('\t' + str(Cxy_s[k][j]))
        file.write('\n')
    file.write('\n')
def save_graph(graph, file):
    '''
    Recieves: graph (network graph)
              file (output file variable)
    Returns: (saves the network graph to file edge by edge)
    '''
    
    file.write('GRAPH\n\n')

    for node1, node2, data in sorted(graph.edges(data=True)):
        weight = data.get("weight", 1)
        file.write(f"{node1} -- {node2} (weight={weight})\n")
    
    file.write('\n')

def save_one_liner(text, value, file):
    '''
    Recieves: text (definition of the value)
              value (computed quantity)
              file (output file variable)
    Returns: (saves one line of concatenaited string to file)
    '''
    if (file == None):
        return

    file.write(text + ': ' + str(value) + '\n\n')
def save_two_dim(cat1, cat2, data1, data2, file):
    '''
    Recieves: cat1, cat2 (cathegory name strings)
              data1, data2 (data arrays)
              file (output file variable)
    Returns: (saves two corresponding columns of data to file)
    '''
    
    if (file == None):
        return
    if (len(data1) != len(data2)):
        file.write('< saving failed >')
        return
    
    file.write(cat1 + '\t' + cat2 + '\n')
    for i in range(len(data1)):
        file.write(str(data1[i]) + '\t' + str(data2[i]) + '\n')
    file.write('\n')
def save_five_dim(cat1, cat2, cat3, cat4, cat5, data1, data2, data3, data4, data5, file):
    '''
    Recieves: cat1, ... cat5 (cathegory name strings)
              data1, ... data5 (data arrays)
              file (output file variable)
    Returns: (saves five corresponding columns of data to file)
    '''
    
    if (file == None):
        return
    if (not (len(data1) == len(data2) == len(data3) == len(data4) == len(data5))):
        file.write('< saving failed >')
        return
    
    file.write(cat1 + '\t' + cat2 + '\t' + cat3 + '\t' + cat4 + '\t' + cat5 + '\n')
    for i in range(len(data1)):
        file.write(str(data1[i]) + '\t' + str(data2[i]) + '\t' + str(data3[i]) + '\t' + str(data4[i]) + '\t' + str(data5[i]) + '\n')
    file.write('\n')
def save_heatmap(title, heatmap, file): # greys
    '''
    Recieves: title (title of heatmap)
              heatmap (pandas heatmap)
              file (output file variable)
    Returns: (saves all heatmap data to file)
    '''

    file.write(title + ':\n')
    file.write(heatmap.to_string())
    file.write('\n\n')
def save_path(path, file):
    '''
    Recieves: path (2D array containing shortest paths between each ordered node pair)
              file (output file variable)
    Returns: (saves all shortest paths to file)
    '''
    
    for i in range(len(path)):
        file.write('Shortest path from ' + str(path[i][0]) + ' to ' + str(path[i][-1]) + ':\n')
        file.write(str(path[i]))
        file.write('\n')
    file.write('\n')
def save_array(text, array, file):
    '''
    Recieves: text (what the array contains)
              array (array of data)
              file (output file variable)
    Returns: (saves text and array into separate lines of file)
    '''
    
    file.write(text + ':\n')
    file.write(str(array))
    file.write('\n\n')

# VISUALIZATION .................................................................................................

def show_corr_matrix(corr_matrix):
    '''
    Recieves: corr_matrix (correlation matrix)
    Returns: (heatmap of the correlation matrix)
    '''

    sns.heatmap(corr_matrix, square = True, cmap="jet", vmin=-1, vmax=1)
    plt.title("Correlation matrix")
    plt.show()
def show_spectral_coherence_analysis(regionA, regionB, labels, ROI_pair_s, f_s, Cxy_s):
    '''
    Recieves: regionA (index of first region)
              regionB (index of second region)
              labels (names of ROIs)
              ROI_pair_s (names of ROI pairs)
              f_s (frequencies)
              Cxy_s (coherence)
    Returns: (creates a plot of coherence with respect to frequency)
    '''
    
    n = ROI_pair_s.index(str(labels[regionA] + ', ' + labels[regionB]))
    
    plt.semilogy(f_s[n], Cxy_s[n]) # logarithmic y axis
    title = 'Spectral Coherence between Brain Regions: ' + ROI_pair_s[n]
    plt.title(title)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Coherence')
    plt.grid()
    plt.show()
def show_all_spectral_coherence_analysis(f_s, Cxy_s):
    '''
    Recieves: ROI_pair_s (names of ROI pairs)
              f_s (frequencies)
              Cxy_s (coherence)
    Returns: (creates a plot of coherence with respect to frequency)
    '''
    
    for n in range(len(Cxy_s)):
        plt.plot(f_s[n], Cxy_s[n], '.k', markeredgecolor = 'none', alpha = 0.5)

    plt.title('Spectral Coherence between Brain Regions with respect to Frequency')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Coherence')
    plt.ylim(0, 1)
    plt.grid()
    plt.show()
def show_graph(graph):
    '''
    Recieves: graph (network graph)
    Returns: (visualizes the network graph)
    '''

    nodes = nx.spring_layout(graph, seed = 42)
    edges = graph.edges(data = True)
    edge_widths = [abs(data['weight'])*3 for _, _, data in edges]

    nx.draw(graph, nodes, with_labels = True, node_color = 'skyblue', node_size = 2000, width = edge_widths)
    plt.show()
def show_heatmap(heatmap, title):
    '''
    Recieves: heatmap (pandas heatmap)
    Returns: (visualization of heatmap)
    '''

    sns.heatmap(heatmap, square = True, annot = True, cmap = "Greys", cbar = False)
    plt.title(title)
    plt.show()

def show_node_degree(nodes, degrees):
    '''
    Recieves: nodes (names of nodes)
              degrees (# of connections of nodes)
    Returns: (plots degrees with respect to nodes)
    '''
    
    plt.scatter(nodes, degrees)
    plt.title('Degree of nodes')
    plt.xlabel('Node')
    plt.ylabel('Degree')
    plt.xticks(rotation = 90)
    for i in range(len(nodes)): # add y value to each point
        plt.annotate(f"{degrees[i]:.2f}", (nodes[i], degrees[i]), textcoords="offset points", xytext=(0,5), ha='center')
    plt.ylim(top = np.max(degrees) + (np.max(degrees)-np.min(degrees))*0.1) # for y values to fit on plot
    plt.subplots_adjust(bottom=0.5) # for x labels to fit on screen
    plt.show()
def show_degree_distribution(degrees, probabilities):
    '''
    Recieves: degrees (# of connections)
              probabilities (probability of a node having given degree)
    Returns: (plots probabilities with respect to degrees)
    '''
    
    plt.scatter(degrees, probabilities)
    plt.title('Degree distribution')
    plt.xlabel('Degree')
    plt.ylabel('Probability')
    for i in range(len(degrees)): # add y value to each point
        plt.annotate(f"{probabilities[i]:.2f}", (degrees[i], probabilities[i]), textcoords="offset points", xytext=(0,5), ha='center')
    plt.show()
def show_clustering_coeff(node, cc):
    '''
    Recieves: node (names of nodes)
              cc (clustering coefficients)
    Returns: (plots clustering coefficient for each node)
    '''
    
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
def show_centrality(node, c, prefix):
    '''
    Recieves: node (names of nodes)
              c (some kind of centralities)
              prefix (distinguishable part of centrality name)
    Returns: (plots some kind of centrality for each node)
    '''
    
    plt.scatter(node, c)
    title_parts = [str(prefix), 'centrality of nodes']
    plt.title(' '.join(title_parts))
    plt.xlabel('Node')
    ylabel_parts = [str(prefix), 'centrality']
    plt.ylabel(' '.join(ylabel_parts))
    plt.xticks(rotation = 90)
    for i in range(len(node)): # add y value to each point
        plt.annotate(f"{c[i]:.2f}", (node[i], c[i]), textcoords="offset points", xytext=(0,5), ha='center')
    plt.ylim(top = np.max(c) + (np.max(c)-np.min(c))*0.1) # for y values to fit on plot
    plt.subplots_adjust(bottom=0.5) # for x labels to fit on screen
    plt.show()