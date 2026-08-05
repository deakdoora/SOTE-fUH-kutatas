# IMPORTS

import functions as func

# TEST INTERFACE

'''
# '1663-TG' / 'fUS-2383-WT'
subject = '1663-TG'
setting = '3D_sbsi-slice'
thr = 0.4

labels, timestamp, data_matrix = func.load_data('sub-' + subject + '/*_sub*-fus' + setting + '.txt')
data = func.group_var(subject, setting, labels, timestamp, data_matrix)
corr_matrix = func.correlation_matrix(data)
network_graph = func.graph(corr_matrix, thr)
'''

#file = None
#while (file == None):
    #try:
        #file = open('k-Means-Clustering_2581_3D-sbsi-slice.txt', "w")
    #except OSError:
        #print("Error opening file")
#file.close()

# USER INTERFACE

def runtime():
    
    userinput = 0
    while (userinput != 8):

        # Choice
        print('\nPick one of the following options:')

        print('\n1. Start analysis')
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

        # Provide subject & setting data
        print("\nEnter unique SUBJECT CODE:")
        subject = str(input())
        print("\nEnter SETTING abbreviation:")
        setting = str(input())

        # Load data & form correlation matrix & create network graph
        labels, timestamp, data_matrix = func.load_data('sub-' + subject + '/*_sub*-fus' + setting + '.txt')
        data = func.group_var(subject, setting, labels, timestamp, data_matrix)
        corr_matrix = func.correlation_matrix(data)

        # Choice
        userinput = 0
        options = True
        while (userinput != 7 and userinput != 8):

            if (options == True):
                print('\nPick one of the following options:')

                print('\n1. CBV Signals in Time')
                print('2. Correlation Matrix')
                print('3. K-means Clustering')
                print('4. Spectral Coherence Analysis')
                print('5. Network Graph')
                print('6. Infomap')
                print('7. Provide new data file')
                print('8. Quit\n')

                options = False

            userinput = int(input())
            match userinput:
                case 1:    # CBV Signals

                    func.show_time_signals(data)

                    options = True

                case 2:    # Correlation Matrix

                    userinput_cm = 0
                    while (userinput_cm != 3 and userinput_cm != 4 and userinput_cm != 5):
                    
                        cm = True
                        if (cm == True):
                            print('\nPick one of the following options:')

                            print('\n1. Correlation Matrix')
                            print('2. Correlation Matrix without NaN Values')
                            print('3. Quit Correlation Matrix')
                            print('4. Provide new data file')
                            print('5. Quit\n')

                            cm = False

                        userinput_cm = int(input())
                        match userinput_cm:
                            case 1:    # Matrix

                                func.show_corr_matrix(data, corr_matrix)

                                cm = True

                            case 2:    # Truncated Matrix
                                
                                corr_matrix_validentries = func.correlation_matrix_dropnan(corr_matrix)
                                func.show_corr_matrix(data, corr_matrix_validentries)

                                cm = True

                            case 3:    # Quit CM
                                options = True
                                break
                            
                            case 4:    # New data file
                                userinput = 7
                                options = False
                                break
                            
                            case 5:    # Quit
                                return
                            
                            case _:
                                print("Choose from above:")
                                userinput_cm = 0

                case 3:    # K-means Clustering

                    fk = None
                    print("\nName file to write in:")
                    filename_kmeans = str(input()) + '.txt'
                    fk = open(filename_kmeans, "w")

                    k_means_clusters = func.k_means_clustering(data, corr_matrix)
                    func.save_k_means_clustering(k_means_clusters, fk)

                    fk.close()

                    options = True

                case 4:    # Spectral Coherence Analysis1
                    
                    userinput_sca = 0
                    while (userinput_sca != 3 and userinput_sca != 4 and userinput_sca != 5):
                    
                        const_sca = True
                        if (const_sca == True):
                            print('\nPick one of the following options:')

                            print('\n1. Spectral Coherence Analysis of two ROIs')
                            print('2. Spectral Coherence Analysis of all ROIs')
                            print('3. Quit Spectral Coherence Analysis')
                            print('4. Provide new data file')
                            print('5. Quit\n')

                            const_sca = False

                        userinput_sca = int(input())
                        match userinput_sca:
                            case 1:    # 2 ROIs

                                sca = func.spectral_coherence_analysis(data)
                                
                                print('\nRegion A:')
                                regionA = -1
                                while (regionA < 0 or regionA >= len(labels)):
                                    regionA = int(input())
                                print('\nRegion B :')
                                regionB = -1
                                while (regionB < 0 or regionB >= len(labels)):
                                    regionB = int(input())
                                
                                func.show_spectral_coherence_analysis(regionA, regionB, data, sca)

                                const_sca = True

                            case 2:    # all ROIs
                                
                                sca = func.spectral_coherence_analysis(data)
                                
                                fsca = None
                                while (fsca == None):
                                    try:
                                        print('\nName file to write in:')
                                        filename_spectral = str(input()) + '.txt'
                                        fsca = open(filename_spectral, "w")
                                    except OSError:
                                        print("Error opening file")

                                func.save_spectral_coherence_analysis(sca, fsca)
                                func.show_all_spectral_coherence_analysis(data, sca)

                                fsca.close()

                                const_sca = True

                            case 3:    # Quit SPA
                                options = True
                                break
                            
                            case 4:    # New data file
                                userinput = 7
                                options = False
                                break
                            
                            case 5:    # Quit
                                return
                            
                            case _:
                                print("Choose from above:")
                                userinput_sca = 0
                
                case 5:    # Network graph

                    userinput_ng = 0
                    while (userinput_ng != 23 and userinput_ng != 24 and userinput_ng != 25):
                    
                        fng = None
                        while (fng == None):
                            try:
                                print('\nName file to write in:')
                                filename_graph = 'graph_subject-' + str(subject) + '_' + setting + '_' + str(input()) + '.txt'
                                fng = open(filename_graph, "w")
                            except OSError:
                                print("Error opening file")

                        print("\nThreshold:")
                        thr = -1
                        while (thr < 0 or thr > 1):
                            thr = float(input())

                        network_graph = func.graph(corr_matrix, thr)
                        func.save_graph(network_graph, fng)
                        adj_matrix = func.adjacency_matrix(network_graph)
                        func.save_heatmap('Adjacency matrix', adj_matrix, fng)

                        userinput_ng = 0
                        while (userinput_ng != 22 and userinput_ng != 23 and userinput_ng != 24 and userinput_ng != 25):
                        
                            ng = True
                            if (ng == True):
                                print('\nPick one of the following options:')

                                print('\n1. Show network graph')
                                print('2. Show adjacency matrix')
                                print('3. Number of nodes')
                                print('4. Number of edges')
                                print('5. Density')
                                print('6. Node degree')
                                print('7. Degree distribution')
                                print('8. Clustering coefficient')
                                print('9. Centralities')
                                print('10. Shortest path lengths')
                                print('11. Shortest paths')
                                print('12. Average path length')
                                print('13. Diameter')
                                print('14. Connected components')
                                print('15. Largest connected component')
                                print('16. Modularity')
                                print('17. Assortativity')
                                print('18. Network efficiency')
                                print('19. Robustness to random failure')
                                print('20. Robustness to targetted attack')
                                print('21. Percolation threshold')
                                print('22. New Network Graph')
                                print('23. Quit Network Graph')
                                print('24. Provide new data file')
                                print('25. Quit\n')

                                ng = False

                            userinput_ng = int(input())
                            match userinput_ng:
                                case 1:    # show network graph

                                    func.show_graph(data, network_graph, thr)

                                    ng = True

                                case 2:    # show adjacency matrix
                                    
                                    func.show_heatmap(adj_matrix, 'Adjacency Matrix', data)

                                    ng = True

                                case 3:    # nodes

                                    n = func.graph_nodes(network_graph)
                                    print('\nNumber of nodes:', n)
                                    func.save_one_liner('Number of nodes', n, fng)

                                    ng = True

                                case 4:    # edges

                                    e = func.graph_edges(network_graph)
                                    print('\nNumber of edges:', e)
                                    func.save_one_liner('Number of edges', e, fng)

                                    ng = True

                                case 5:    # density

                                    d = func.graph_density(network_graph)
                                    print('\nGraph density:', d)
                                    func.save_one_liner('Graph density', d, fng)

                                    ng = True

                                case 6:    # node degree

                                    nodes, degrees = func.node_degree(network_graph)
                                    func.save_two_dim('Node', 'Degree', nodes, degrees, fng)
                                    func.show_node_degree(nodes, degrees)

                                    ng = True

                                case 7:    # degree distribution

                                    degrees, probabilities = func.degree_distribution(network_graph)
                                    func.save_two_dim('Degree', 'Probability', degrees, probabilities, fng)
                                    func.show_degree_distribution(degrees, probabilities)

                                    ng = True

                                case 8:    # clustering coeff

                                    node, cc = func.clustering_coeff(network_graph)
                                    func.save_two_dim('Node', 'Clustering coefficient', node, cc, fng)
                                    func.show_clustering_coeff(data, node, cc)

                                    ng = True

                                case 9:    # centralities

                                    node, dc = func.degree_centrality(network_graph)
                                    node, bc = func.betweenness_centrality(network_graph)
                                    node, cc = func.closeness_centrality(network_graph)
                                    node, ec = func.eigenvector_centrality(network_graph)

                                    func.save_five_dim('Node', 'Degree centrality', 'Betweenness centrality', 'Closeness centrality', 'Eigenvector centrality', node, dc, bc, cc, ec, fng)

                                    func.show_centrality(data, node, dc, 'Degree')
                                    func.show_centrality(data, node, bc, 'Betweenness')
                                    func.show_centrality(data, node, cc, 'Closeness')
                                    func.show_centrality(data, node, ec, 'Eigenvector')

                                    ng = True

                                case 10:    # shortest path lengths

                                    l_matrix = func.shortest_path_length(network_graph)
                                    func.save_heatmap('Shortest path lengths', l_matrix, fng)
                                    func.show_heatmap(l_matrix, 'Shortest path lengths')

                                    wl_matrix = func.weighted_shortest_path_length(network_graph)
                                    func.save_heatmap('Weighted shortest path lengths', wl_matrix, fng)
                                    func.show_heatmap(wl_matrix, 'Weighted shortest path lengths')

                                    ng = True

                                case 11:    # shortest paths

                                    path = func.shortest_path(network_graph)
                                    func.save_path(path, fng)
                                    print('\nShortest paths:\n', path)

                                    ng = True

                                case 12:    # average path length

                                    ave_l = func.ave_path_length(network_graph)
                                    func.save_one_liner('Average shortest path length', ave_l, fng)
                                    print('\nAverage shortest path length:', ave_l)

                                    ave_wl = func.ave_weighted_path_length(network_graph)
                                    func.save_one_liner('Average weighted shortest path length', ave_wl, fng)
                                    print('Average weighted shortest path length:', ave_wl)

                                    ng = True

                                case 13:    # diameter

                                    d = func.diameter(network_graph)
                                    func.save_one_liner('Diameter', d, fng)
                                    print('\nDiameter:', d)

                                    wd = func.weighted_diameter(network_graph)
                                    func.save_one_liner('Weighted diameter', wd, fng)
                                    print('Weighted diameter', wd)

                                    ng = True

                                case 14:    # connected components

                                    n, conn_comp = func.connected_components(network_graph)
                                    fng.write('Number of islands: ' + str(n) + '\n')
                                    fng.write(str(conn_comp) + '\n\n')
                                    print('\nNumber of islands:', n)
                                    print(str(conn_comp))

                                    ng = True

                                case 15:    # largest connected component

                                    giant = func.giant_component(network_graph)
                                    func.save_array('Largest island', giant, fng)
                                    print('\nLargest island:\n', giant)

                                    ng = True

                                case 16:    # modularity

                                    mod = func.modularity(network_graph)
                                    func.save_one_liner('Modularity', mod, fng)
                                    print('\nModularity:', mod)

                                    wmod = func.weighted_modularity(network_graph)
                                    func.save_one_liner('Weighted modularity', wmod, fng)
                                    print('Weighted modularity:', wmod)

                                    ng = True

                                case 17:    # assortativity

                                    a = func.assortativity(network_graph)
                                    func.save_one_liner('Assortativity', a, fng)
                                    print('\nAssortativity:', a)

                                    ng = True

                                case 18:    # network efficiency

                                    ne = func.network_efficiency(network_graph)
                                    func.save_one_liner('Network efficiency', ne, fng)
                                    print('\nNetwork efficiency:', ne)

                                    wne = func.weighted_network_efficiency(network_graph)
                                    func.save_one_liner('Weighted network efficiency', wne, fng)
                                    print('Weighted network efficiency:', wne)

                                    ng = True

                                case 19:    # robustness to random failure

                                    func.robustness_to_random_failure(network_graph, fng)
                                    print('\n< Robustness to random failure has been saved to file >')

                                    ng = True

                                case 20:    # robustness to targetted attack

                                    func.robustness_to_targeted_attack(network_graph, fng)
                                    print('\n< Robustness to targetted attack has been saved to file >')

                                    ng = True

                                case 21:    # percolation threshold

                                    fr_r_n, gc_r_n, fr_r_e, gc_r_e = func.robustness_to_random_failure(network_graph, None)
                                    fr_t_n, gc_t_n, fr_t_e, gc_t_e = func.robustness_to_targeted_attack(network_graph, None)
                                    fractions, giant_components = func.group_robustness(fr_r_n, gc_r_n, fr_r_e, gc_r_e, fr_t_n, gc_t_n, fr_t_e, gc_t_e)
                                    
                                    func.show_percolation_threshold(data, fractions, giant_components)

                                    ng = True

                                case 22:    # New network graph
                                    pass

                                case 23:    # Quit network graph
                                    options = True
                                    break
                                
                                case 24:    # New data file
                                    userinput = 7
                                    options = False
                                    break
                                
                                case 25:    # Quit
                                    return
                                
                                case _:
                                    print("Choose from above:")
                                    userinput_ng = 0

                    fng.close()

                    #options = True

                case 6:    # Infomap

                    userinput_im = 0
                    while (userinput_im != 5 and userinput_im != 6 and userinput_im != 7):

                        print("\nThreshold:")
                        thr = -1
                        while (thr < 0 or thr > 1):
                            thr = float(input())

                        network_graph = func.graph(corr_matrix, thr)

                        print("\nNumber of trials:")
                        num_trials = 0
                        while (num_trials <= 0):
                            num_trials = int(input())
                        print("\nSeed:")
                        seed = int(input())

                        imap = func.my_infomap(network_graph, num_trials, seed)

                        userinput_im = 0
                        while (userinput_im != 4 and userinput_im != 5 and userinput_im != 6 and userinput_im != 7):
                        
                            im = True
                            if (im == True):
                                print('\nPick one of the following options:')

                                print('\n1. Number of communities')
                                print('2. Show infomap')
                                print('3. Save infomap')
                                print('4. New infomap')
                                print('5. Quit Infomap')
                                print('6. Provide new data file')
                                print('7. Quit\n')

                                im = False
                                
                            userinput_im = int(input())
                            match userinput_im:
                                case 1:    # Num of comm
                                
                                    print('\nNumber of top level communities: ', imap.num_top_modules)

                                    im = True

                                case 2:    # Show IM

                                    func.show_infomap(data, thr, network_graph, imap)

                                    im = True

                                case 3:    # Save IM

                                    fim = None
                                    while (fim == None):
                                        try:
                                            print('\nName file to write in:')
                                            filename_im = 'infomap_subject-' + str(subject) + '_' + setting + '_' + str(input()) + '.txt'
                                            fim = open(filename_im, "w")
                                        except OSError:
                                            print("Error opening file")

                                    func.save_infomap(imap, network_graph, fim)

                                    fim.close()

                                    im = True

                                case 4:    # New IM
                                    pass

                                case 5:    # Quit IM
                                    options = True
                                    break
                                
                                case 6:    # New data file
                                    userinput = 7
                                    options = False
                                    break
                                
                                case 7:    # Quit
                                    return
                                
                                case _:
                                    print("Choose from above:")
                                    userinput_im = 0
                                        
                case 7:    # New data file
                    pass

                case 8:    # Quit
                    return
                
                case _:
                    print("Choose from above:")
                    userinput = 0
def analysis(subject, setting, thr, unique_filename_part): # complete analysis of a measurement

    # SECTION SELECTION
    print('\nSelect the sections to include in the analysis:')
    run = False
    userinput = -1
    sections = []
    while (run == False):
        print('\n1. CBV signals in time', end = '')
        if 1 in sections:
            print(' < added >', end = '')
        print('\n2. Correlation Matrix', end = '')
        if 2 in sections:
            print(' < added >', end = '')
        print('\n3. Spectral coherence analysis', end = '')
        if 3 in sections:
            print(' < added >', end = '')
        print('\n4. Network graph', end = '')
        if 4 in sections:
            print(' < added >', end = '')
        print('\n5. Adjacency matrix', end = '')
        if 5 in sections:
            print(' < added >', end = '')
        print('\n6. Basic structural parameters', end = '')
        if 6 in sections:
            print(' < added >', end = '')
        print('\n7. Node level metrics', end = '')
        if 7 in sections:
            print(' < added >', end = '')
        print('\n8. Path based metrics', end = '')
        if 8 in sections:
            print(' < added >', end = '')
        print('\n9. Global network properties', end = '')
        if 9 in sections:
            print(' < added >', end = '')
        print('\n10. Flow and robustness', end = '')
        if 10 in sections:
            print(' < added >', end = '')
        print('\n11. Infomap', end = '')
        if 11 in sections:
            print(' < added >', end = '')

        print('\n\nPress 0 to run!\n')

        userinput = int(input())
        match userinput:
            case 0:    # run
                run = True

            case _ if 1 <= userinput <= 11:
                if userinput not in sections:
                    sections.append(int(userinput))

            case _:
                print("Choose from the list:")
                userinput = -1

    # CONSTANT
    sampling_freq = 15000000
    
    # OPEN FILE
    output_filename = 'subject-' + str(subject) + '_' + setting + '_' + unique_filename_part + '.txt'
    file = func.open_analysis_file(subject, setting, output_filename)
    print('< Output File opened >\n')

    # LOAD DATA
    labels, timestamp, data_matrix = func.load_data('sub-' + subject + '/*_sub*-fus' + setting + '.txt')
    data = func.group_var(subject, setting, labels, timestamp, data_matrix)
    print('< Data loaded >\n')

    # CBV SIGNALS IN TIME
    if 1 in sections:
        func.save_time_signals(data, file)
        print('< CBV Signals in Time saved >\n')

    # CORRELATION MATRIX
    corr_matrix = func.correlation_matrix(data)
    if 2 in sections:
        func.save_corr_matrix(corr_matrix, file)
        print('< Correlation Matrix saved >\n')
    
    # K-MEANS CLUSTERING
    #k_means_clusters = k_means_clustering(corr_matrix, labels)
    #save_k_means_clustering(k_means_clusters, file)

    # SPECTRAL COHERENCE ANALYSIS
    if 3 in sections:
        sca = func.spectral_coherence_analysis(data, sampling_freq)
        func.save_spectral_coherence_analysis(sca, file)
        print('< Spectral Coherence Analysis saved >\n')
    
    # NETWORK GRAPH
    network_graph = func.graph(corr_matrix, thr)
    if 4 in sections:
        func.save_graph(network_graph, file)
        print('< Network Graph saved >\n')

    # Adjacency matrix
    if 5 in sections:
        adj_matrix = func.adjacency_matrix(network_graph)
        func.save_heatmap('Adjacency matrix', adj_matrix, file)
        print('< Adjacency Matrix saved >\n')

    # Basic structural parameters
    if 6 in sections:
        file.write('BASIC STRUCTURAL PARAMETERS\n\n')
        func.save_one_liner('Number of nodes', func.graph_nodes(network_graph), file)
        func.save_one_liner('Number of edges', func.graph_edges(network_graph), file)
        func.save_one_liner('Graph density', func.graph_density(network_graph), file)
        print('< Basic Structural Parameters saved >\n')
    
    # Node level metrics
    if 7 in sections:
        file.write('NODE LEVEL METRICS\n\n')
        nodes, degrees = func.node_degree(network_graph)
        func.save_two_dim('Node', 'Degree', nodes, degrees, file)
        degrees, probabilities = func.degree_distribution(network_graph)
        func.save_two_dim('Degree', 'Probability', degrees, probabilities, file)
        node, cc = func.clustering_coeff(network_graph)
        func.save_two_dim('Node', 'Clustering coefficient', node, cc, file)
        node, dc = func.degree_centrality(network_graph)
        node, bc = func.betweenness_centrality(network_graph)
        node, cc = func.closeness_centrality(network_graph)
        node, ec = func.eigenvector_centrality(network_graph)
        func.save_five_dim('Node', 'Degree centrality', 'Betweenness centrality', 'Closeness centrality', 'Eigenvector centrality', node, dc, bc, cc, ec, file)
        print('< Node Level Metrics saved >\n')

    # Path based metrics
    if 8 in sections:
        file.write('PATH BASED METRICS\n\n')
        l_matrix = func.shortest_path_length(network_graph)
        func.save_heatmap('Shortest path lengths', l_matrix, file)
        wl_matrix = func.weighted_shortest_path_length(network_graph)
        func.save_heatmap('Weighted shortest path lengths', wl_matrix, file)
        path = func.shortest_path(network_graph)
        func.save_path(path, file)
        ave_l = func.ave_path_length(network_graph)
        func.save_one_liner('Average shortest path length', ave_l, file)
        ave_wl = func.ave_weighted_path_length(network_graph)
        func.save_one_liner('Average weighted shortest path length', ave_wl, file)
        d = func.diameter(network_graph)
        func.save_one_liner('Diameter', d, file)
        wd = func.weighted_diameter(network_graph)
        func.save_one_liner('Weighted diameter', wd, file)
        print('< Path Based Metrics saved >\n')

    # Global network properties
    if 9 in sections:
        file.write('GLOBAL NETWORK PROPERTIES\n\n')
        n, conn_comp = func.connected_components(network_graph)
        file.write('Number of islands: ' + str(n) + '\n')
        file.write(str(conn_comp) + '\n\n')
        giant = func.giant_component(network_graph)
        func.save_array('Largest island', giant, file)
        mod = func.modularity(network_graph)
        func.save_one_liner('Modularity', mod, file)
        wmod = func.weighted_modularity(network_graph)
        func.save_one_liner('Weighted modularity', wmod, file)
        a = func.assortativity(network_graph)
        func.save_one_liner('Assortativity', a, file)
        print('< Global Network Properties saved >\n')

    # Flow and robustness
    if 10 in sections:
        file.write('FLOW AND ROBUSTNESS\n\n')
        ne = func.network_efficiency(network_graph)
        func.save_one_liner('Network efficiency', ne, file)
        wne = func.weighted_network_efficiency(network_graph)
        func.save_one_liner('Weighted network efficiency', wne, file)
        func.robustness_to_random_failure(network_graph, file)
        func.robustness_to_targeted_attack(network_graph, file)
        print('< Flow and Robustness section saved >\n')

    # INFOMAP
    if 11 in sections:
        imap = func.my_infomap(network_graph, 20, 123)
        func.save_infomap(imap, network_graph, file)
        print('< Infomap saved >\n')

    # CLOSE FILE
    file.close()
    print('< Output File closed >\n')

#runtime()
#analysis('fUS-2383-WT', '3D_vol', 0.4, 'test')