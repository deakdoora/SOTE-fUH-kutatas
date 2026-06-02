# SOTE-fUH-kutatas

**SOTE fUH kutatómunkához**

TO DO LIST
--
- file load function
- administration / thr, sampling_freq
- plot time signals
- (find k for k-means clustering)
- (visualize k-means clustering)
- save 2 ROI spectral coherence analysis
- write SCA to file with tabulators so it fits into Excel
- visualize connected components
- add missing graph parametres
- check correctness of computation of graph parametres
- test for 4D data
- compair data

- (integrate graph parametres into the runtime function)

NOTES
--
- test with 0s_to_600.024s_2D_Matrix
- K-means does not accept NaN data

Abbrevitations
--

previous
- wbc = without baseline correction
- f = filtered
- sbs = slice-by-slice interpolation

current
- nblc = no base line correction
- ngs = no global signal
- sbsi = slice-by-slice interpolation

functions
- show_ = function creates a plot or heatmap for visualization
- save_ = function saves to file