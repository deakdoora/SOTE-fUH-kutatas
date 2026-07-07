# IMPORTS

import functions as func

# TEST INTERFACE

subject = '2581' #'fUS-2383-WT'
setting = '2D'

labels, timestamp, data_matrix = func.load_data('sub-' + subject + '/*_*_*_sub*-fus' + setting + '.txt')
data = func.group_var(subject, setting, labels, timestamp, data_matrix)

corr_matrix = func.correlation_matrix(data)
func.show_corr_matrix(data, corr_matrix)
corr_matrix_validentries = func.correlation_matrix_dropnan(corr_matrix)
func.show_corr_matrix(data, corr_matrix_validentries)

'''
file = None
while (file == None):
    try:
        file = open('k-Means-Clustering_2581_3D-sbsi-slice.txt', "w")
    except IOError:
        print("Error opening file")
k_means_clusters = func.k_means_clustering(corr_matrix_validentries, labels)
func.save_k_means_clustering(k_means_clusters, file)
file.close()
'''

'''
func.show_time_signals(data)
corr_matrix = func.correlation_matrix(data)
func.show_corr_matrix(data, corr_matrix)
#k_means_clusters = func.k_means_clustering(corr_matrix, labels)
network_graph = func.graph(corr_matrix, 0)
func.show_graph(data, network_graph)

file = None
while (file == None):
    try:
        file = open('test.txt', "w")
    except IOError:
        print("Error opening file")
sca = func.spectral_coherence_analysis(data)
func.save_spectral_coherence_analysis(sca, file)
func.show_all_spectral_coherence_analysis(data, sca)
file.close()
'''

# USER INTERFACE

'''
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
        labels, timestamp, data_matrix = load_data(sub_fUS_2383_WT)
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
                print('4. Network graph')
                print('5. Provide new data file')
                print('6. Quit\n')

                options = False

            userinput = int(input())
            match userinput:
                case 1:    # Heatmap

                    show_corr_matrix(corr_matrix)

                    options = True

                case 2:    # K-means Clustering

                    fk = None
                    print("\nName file to write in:")
                    filename_kmeans = str(input()) + '.txt'
                    fk = open(filename_kmeans, "w")
                    #print('\nNumber of clusters:')
                    #num_clusters = int(input())

                    k_means_clusters = k_means_clustering(corr_matrix, labels)
                    save_k_means_clustering(k_means_clusters, fk)

                    fk.close()

                    options = True

                case 3:    # Spectral Coherence Analysis
                    
                    userinput_sca = 0
                    while (userinput_sca != 3 and userinput_sca != 4 and userinput_sca != 5):
                    
                        sca = True
                        if (sca == True):
                            print('\nPick one of the following options:')

                            print('\n1. Spectral Coherence Analysis of two ROIs')
                            print('2. Spectral Coherence Analysis of all ROIs')
                            print('3. Quit Spectral Coherence Analysis')
                            print('4. Provide new data file')
                            print('5. Quit\n')

                            sca = False

                        userinput_sca = int(input())
                        match userinput_sca:
                            case 1:    # 2 ROIs

                                ROI_pair_s, f_s, Cxy_s = spectral_coherence_analysis(data_matrix, labels)
                                
                                print('\nRegion A:')
                                regionA = -1
                                while (regionA < 0 or regionA >= len(labels)):
                                    regionA = int(input())
                                print('\nRegion B :')
                                regionB = -1
                                while (regionB < 0 or regionB >= len(labels)):
                                    regionB = int(input())
                                
                                show_spectral_coherence_analysis(regionA, regionB, labels, ROI_pair_s, f_s, Cxy_s)

                                sca = True

                            case 2:    # all ROIs
                                
                                ROI_pair_s, f_s, Cxy_s = spectral_coherence_analysis(data_matrix, labels)
                                
                                fsca = None
                                print('\nName file to write in:')
                                filename_spectral = str(input()) + '.txt'
                                fsca = open(filename_spectral, "w")

                                save_spectral_coherence_analysis(ROI_pair_s, f_s, Cxy_s, fsca)
                                show_all_spectral_coherence_analysis(f_s, Cxy_s)

                                fsca.close()

                                sca = True

                            case 3:    # Quit SPA
                                options = True
                                break
                            
                            case 4:    # New data file
                                userinput = 5
                                options = False
                                break
                            
                            case 5:    # Quit
                                return
                            
                            case _:
                                print("Choose from above:")
                                userinput_sca = 0
                
                case 4:    # Network graph
                    
                    fng = None
                    print('\nName file to write in:')
                    filename_graph = str(input()) + '.txt'
                    fng = open(filename_graph, "w")
                    
                    print('\nThreshold:')
                    thr = 10
                    while (thr < 0 or thr > 1): # absolute correlation
                        thr= float(input())

                    network_graph = graph(corr_matrix, thr)
                    save_graph(network_graph, fng)

                    userinput_ng = 0
                    while (userinput_ng != 20 and userinput_ng != 21 and userinput_ng != 22):
                    
                        ng = True
                        if (ng == True):
                            print('\nPick one of the following options:')

                            print('\n1. Show network graph')
                            print('2. Number of nodes')
                            print('3. Number of edges')
                            print('4. Density')
                            print('5. Node degree')
                            print('6. Degree distribution')
                            print('7. Clustering coefficient')
                            print('8. Centralities')
                            print('9. Shortest path lengths')
                            print('10. Shortest paths')
                            print('11. Average path length')
                            print('12. Diameter')
                            print('13. Connected components')
                            print('14. Largest connected component')
                            print('15. Modularity')
                            print('16. Assortativity')
                            print('17. Network efficiency')
                            print('18. Robustness to random failure')
                            print('19. Robustness to targetted attack')
                            print('20. Quit Network Graph')
                            print('21. Provide new data file')
                            print('22. Quit\n')

                            ng = False

                        userinput_ng = int(input())
                        match userinput_ng:
                            case 1:    # show network graph

                                show_graph(network_graph)

                                ng = True

                            case 2:    # nodes

                                n = graph_nodes(network_graph)
                                print('\nNumber of nodes:', n)
                                save_one_liner('Number of nodes', n, fng)

                                ng = True

                            case 3:    # edges

                                e = graph_edges(network_graph)
                                print('\nNumber of edges:', e)
                                save_one_liner('Number of edges', e, fng)

                                ng = True

                            case 4:    # density

                                d = graph_density(network_graph)
                                print('\nGraph density:', d)
                                save_one_liner('Graph density', d, fng)

                                ng = True

                            case 5:    # node degree

                                nodes, degrees = node_degree(network_graph)
                                save_two_dim('Node', 'Degree', nodes, degrees, fng)
                                show_node_degree(nodes, degrees)

                                ng = True

                            case 6:    # degree distribution

                                degrees, probabilities = degree_distribution(network_graph)
                                save_two_dim('Degree', 'Probability', degrees, probabilities, fng)
                                show_degree_distribution(degrees, probabilities)

                                ng = True

                            case 7:    # clustering coeff

                                node, cc = clustering_coeff(network_graph)
                                save_two_dim('Node', 'Clustering coefficient', node, cc, fng)
                                show_clustering_coeff(node, cc)

                                ng = True

                            case 8:    # centralities

                                node, dc = degree_centrality(network_graph)
                                node, bc = betweenness_centrality(network_graph)
                                node, cc = closeness_centrality(network_graph)
                                node, ec = eigenvector_centrality(network_graph)

                                save_five_dim('Node', 'Degree centrality', 'Betweenness centrality', 'Closeness centrality', 'Eigenvector centrality', node, dc, bc, cc, ec, fng)

                                show_centrality(node, dc, 'Degree')
                                show_centrality(node, bc, 'Betweenness')
                                show_centrality(node, cc, 'Closeness')
                                show_centrality(node, ec, 'Eigenvector')

                                ng = True

                            case 9:    # shortest path lengths

                                l_matrix = shortest_path_length(network_graph)
                                save_heatmap('Shortest path lengths', l_matrix, fng)
                                show_heatmap(l_matrix, 'Shortest path lengths')

                                wl_matrix = weighted_shortest_path_length(network_graph)
                                save_heatmap('Weighted shortest path lengths', wl_matrix, fng)
                                show_heatmap(wl_matrix, 'Weighted shortest path lengths')

                                ng = True

                            case 10:    # shortest paths

                                path = shortest_path(network_graph)
                                save_path(path, fng)
                                print('\nShortest paths:\n', path)

                                ng = True

                            case 11:    # average path length

                                ave_l = ave_path_length(network_graph)
                                save_one_liner('Average shortest path length', ave_l, fng)
                                print('\nAverage shortest path length:', ave_l)

                                ave_wl = ave_weighted_path_length(network_graph)
                                save_one_liner('Average weighted shortest path length', ave_wl, fng)
                                print('Average weighted shortest path length:', ave_wl)

                                ng = True

                            case 12:    # diameter

                                d = diameter(network_graph)
                                save_one_liner('Diameter', d, fng)
                                print('\nDiameter:', d)

                                wd = weighted_diameter(network_graph)
                                save_one_liner('Weighted diameter', wd, fng)
                                print('Weighted diameter', wd)

                                ng = True

                            case 13:    # connected components

                                n, conn_comp = connected_components(network_graph)
                                fng.write('Number of islands: ' + str(n) + '\n')
                                fng.write(str(conn_comp) + '\n\n')
                                print('\nNumber of islands:', n)
                                print(str(conn_comp))

                                ng = True

                            case 14:    # largest connected component

                                giant = giant_component(network_graph)
                                save_array('Largest island', giant, fng)
                                print('\nLargest island:\n', giant)

                                ng = True

                            case 15:    # modularity

                                mod = modularity(network_graph)
                                save_one_liner('Modularity', mod, fng)
                                print('\nModularity:', mod)

                                wmod = weighted_modularity(network_graph)
                                save_one_liner('Weighted modularity', wmod, fng)
                                print('Weighted modularity:', wmod)

                                ng = True

                            case 16:    # assortativity

                                a = assortativity(network_graph)
                                save_one_liner('Assortativity', a, fng)
                                print('\nAssortativity:', a)

                                ng = True

                            case 17:    # network efficiency

                                ne = network_efficiency(network_graph)
                                save_one_liner('Network efficiency', ne, fng)
                                print('\nNetwork efficiency:', ne)

                                wne = weighted_network_efficiency(network_graph)
                                save_one_liner('Weighted network efficiency', wne, fng)
                                print('Weighted network efficiency:', wne)

                                ng = True

                            case 18:    # robustness to random failure

                                robustness_to_random_failure(network_graph, fng)
                                print('\n< Robustness to random failure has been saved to file >')

                                ng = True

                            case 19:    # robustness to targetted attack

                                robustness_to_targeted_attack(network_graph, fng)
                                print('\n< Robustness to targetted attack has been saved to file >')

                                ng = True

                            case 20:    # Quit network graph
                                options = True
                                break
                            
                            case 21:    # New data file
                                userinput = 5
                                options = False
                                break
                            
                            case 22:    # Quit
                                return
                            
                            case _:
                                print("Choose from above:")
                                userinput_ng = 0

                    fng.close()

                    options = True

                case 5:    # New data file
                    pass

                case 6:    # Quit
                    return
                
                case _:
                    print("Choose from above:")
                    userinput = 0
def analysis(input_filename, output_filename): # complete analysis of a measurement
    # ALL INPUT
    sampling_freq = 15000000
    thr = 0
    
    # OPEN FILE
    file = None
    while (file == None):
        try:
            file = open(output_filename, "w")
        except IOError:
            print("Error opening file")

    # ADMINISTRATION
    file.write('Date: ' + str(datetime.datetime.now()) + '\n')
    file.write('Data file: ' + str(input_filename) + '\n\n')

    # CORRELATION MATRIX
    labels, timestamp, data_matrix = load_data(input_filename) # splits data
    corr_matrix = pd.DataFrame(data = data_matrix, columns = labels).corr() # creates correlation matrix
    save_corr_matrix(corr_matrix, file)
    
    # K-MEANS CLUSTERING
    #k_means_clusters = k_means_clustering(corr_matrix, labels)
    #save_k_means_clustering(k_means_clusters, file)

    # SPECTRAL COHERENCE ANALYSIS
    ROI_pair_s, f_s, Cxy_s = spectral_coherence_analysis(data_matrix, labels, sampling_freq)
    save_spectral_coherence_analysis(ROI_pair_s, f_s, Cxy_s, file)
    
    # GRAPH
    network_graph = graph(corr_matrix, thr)
    save_graph(network_graph, file)

    # Basic structural parametres
    file.write('BASIC STRUCTURAL PARAMETRES\n\n')
    save_one_liner('Number of nodes', graph_nodes(network_graph), file)
    save_one_liner('Number of edges', graph_edges(network_graph), file)
    save_one_liner('Graph density', graph_density(network_graph), file)

    # Node level metrics
    file.write('NODE LEVEL METRICS\n\n')
    nodes, degrees = node_degree(network_graph)
    save_two_dim('Node', 'Degree', nodes, degrees, file)
    degrees, probabilities = degree_distribution(network_graph)
    save_two_dim('Degree', 'Probability', degrees, probabilities, file)
    node, cc = clustering_coeff(network_graph)
    save_two_dim('Node', 'Clustering coefficient', node, cc, file)
    node, dc = degree_centrality(network_graph)
    node, bc = betweenness_centrality(network_graph)
    node, cc = closeness_centrality(network_graph)
    node, ec = eigenvector_centrality(network_graph)
    save_five_dim('Node', 'Degree centrality', 'Betweenness centrality', 'Closeness centrality', 'Eigenvector centrality', node, dc, bc, cc, ec, file)

    # Path based metrics
    file.write('PATH BASED METRICS\n\n')
    l_matrix = shortest_path_length(network_graph)
    save_heatmap('Shortest path lengths', l_matrix, file)
    wl_matrix = weighted_shortest_path_length(network_graph)
    save_heatmap('Weighted shortest path lengths', wl_matrix, file)
    path = shortest_path(network_graph)
    save_path(path, file)
    ave_l = ave_path_length(network_graph)
    save_one_liner('Average shortest path length', ave_l, file)
    ave_wl = ave_weighted_path_length(network_graph)
    save_one_liner('Average weighted shortest path length', ave_wl, file)
    d = diameter(network_graph)
    save_one_liner('Diameter', d, file)
    wd = weighted_diameter(network_graph)
    save_one_liner('Weighted diameter', wd, file)

    # Global network properties
    file.write('GLOBAL NETWORK PROPERTIES\n\n')
    n, conn_comp = connected_components(network_graph)
    file.write('Number of islands: ' + str(n) + '\n')
    file.write(str(conn_comp) + '\n\n')
    giant = giant_component(network_graph)
    save_array('Largest island', giant, file)
    mod = modularity(network_graph)
    save_one_liner('Modularity', mod, file)
    wmod = weighted_modularity(network_graph)
    save_one_liner('Weighted modularity', wmod, file)
    a = assortativity(network_graph)
    save_one_liner('Assortativity', a, file)

    # Flow and robustness
    file.write('FLOW AND ROBUSTNESS\n\n')
    ne = network_efficiency(network_graph)
    save_one_liner('Network efficiency', ne, file)
    wne = weighted_network_efficiency(network_graph)
    save_one_liner('Weighted network efficiency', wne, file)
    robustness_to_random_failure(network_graph, file)
    robustness_to_targeted_attack(network_graph, file)

    # CLOSE FILE
    file.close()
'''
    
#runtime()  # test with 0s_to_600.024s_2D_Matrix
#analysis(sub_second, 'analysis_sub_second.txt')   # test with '0s_to_600.024s_2D_Matrix.txt' & 'analysis_test_i.txt'