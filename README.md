# **SOTE fUH kutatás**
**A Semmelweis Egyetemen funkcionális ultrahangos kutatómunkához agyterületek kapcsolatát, neurális hálót elemző program, főként gráfelmélet felhasználásával.**

> TO DO LIST
> -
> 
> <ul>
> <li> fix k-means clustering
> <li> add missing graph parametres
> </ul>
> 
> <ul>
> <li> check correctness of computation of graph parametres
> <li> compair data
> </ul>
> 
> <ul>
> <li> (find k for k-means clustering)
> <li> (visualize k-means clustering)
> <li> (visualize connected components)
> <li> (integrate graph parametres into the runtime function)
> <li> (function to decode setting abbreviations)
> </ul>

> NOTES
> -
> 
> <ul>
> <li> (spike) 2581, 3D_vol
> <li> K-means does not accept NaN data
> </ul>

> ABBREVIATIONS
> -
> 
> <ul>
> <li> nblc = no base line correction
> <li> ngs = no global signal
> <li> vol = volumic interpolation
> <li> sbsi-slice = slice-by-slice interpolation
> </ul>
> 
> <ul>
> <li> show_ = function creates a plot or heatmap for visualization
> <li> save_ = function saves to file
> </ul>

THEORY
-

#### **K-MEANS CLUSTERING**

---

K-means clustering is an unsupervised machine learning algorithm used in neuroimaging to categorize data into distinct, non-overlapping sets based on similarity. In the context of functional ultrasound (fUS), it is primarily applied to classify response patterns and perform automatic brain parcellation. This allows for a data-driven definition of brain structures rather than relying solely on anatomical atlases.

---

#### **SPECTRAL COHERENCE ANALYSIS**

---

Spectral coherence analysis is a frequency-domain method used to evaluate the consistency of the relationship between two signals — specifically, how well they correlate at specific frequencies. In the context of functional ultrasound (fUS), it is used to investigate resting-state functional connectivity by determining if different brain regions share synchronized fluctuations in cerebral blood volume (CBV).

---

#### **NETWORK GRAPH PARAMETERS**

---

#### ADJECENCY MATRIX

Contains the weights of edges between graph nodes arranged in matrix form.

---

#### BASIC STRUCTURAL PARAMETERS

**Number of nodes**

Total number of neurons in the network, thus represents the size of the network.

**Number of edges**

Total number of connections between the nodes, indicates how well connected / dense the network is.

**Density**

Ratio of existing edges to all possible edges, shows how close the network is to being fully connected.

---

#### NODE-LEVEL METRICS

**Degree**

Number of connections a node has, reflects local importance / activity.

**Degree distribution**

Probability distribution of node degrees. Helps identify random networks and scale-free networks, which have few highly connected neurons.

**Clustering coefficient**

Measures how connected a node’s neighbors are. A high value means the nodes form clusters and indicates local cohesiveness.

**Degree centrality**

Centralities quantify the importance of nodes. Degree centrality is based on the number of connections of the node.

**Betweenness centrality**

Quantifies how often a node lies on shortest paths, it represents control over information flow.

**Closeness centrality**

Shows the average distance to all other nodes and thus measures how quickly a node can reach others.

**Eigenvector centrality**

Its principle is that a node's importance depends and is based on its neighbors’ importance.

---

#### PATH-BASED METRICS

**Shortest path length**

Minimum number of edges information has to pass through between two nodes. Represents the efficiency of communication between them.

**Average shortest path length**

Average of the shortest paths between all node pairs. Indicates how compact or spread out the network is.

**Diameter**

Longest shortest path length in the network. Shows the maximum distance between any two nodes.

---

#### GLOBAL NETWORK PROPERTIES

**Connected components**

Subgraphs where all nodes are reachable, but none other. Indicates connectivity vs fragmentation.

**Giant component**

The largest connected component / island. It is important in real-world networks (e.g. internet, social networks).

**Modularity**

Measures the strength of division into communities. A high modularity indicates strong community-like structure.

**Assortativity**

Measures the preference for nodes to connect to similar nodes, therefore shows a network mixing patterns.

---

#### FLOW & ROBUSTNESS

**Network efficiency**

How efficiently information is exchanged. Shown with computation based on inverse shortest path lengths.

**Robustness / Resilience**

How the network behaves under node removals or edge failures. Important for biological systems and infrastructure networks.

**Percolation threshold**

The critical point at which a network changes from being mostly connected to being fragmented. Used in studying network stability and phase transitions.

---

#### SPECIAL PROPERTIES

**Small-world property**

Networks with high clustering and short path lengths. Typical in social networks.

**Scale-free property**

When the degree distribution follows a power law. Means the presence of hubs = very highly connected nodes.