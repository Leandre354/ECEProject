# Master's Thesis: 

# Code Structure
* [`ECEProject/`](ECEProject/):
   * [`Analysis/`](ECEProject/Analysis/):
       * [`clinical_plot.py`](ECEProject/Analysis/clinical_plot.py): Contains the plottings for the clinical evaluation grades and the correlation between dice and evaluation.
       * [`correlation_plot.py`](ECEProject/Analysis/correlation_plot.py): Plots of the four correlation heatmaps between properties of segmented area and delta dice induced by perturbations.
       * [`dice_comparator.py`](ECEProject/Analysis/dice_comparator.py): Script computing the difference between baseline and corresponding pertubated case.
       * [`errorHeatMap.py`](ECEProject/Analysis/errorHeatMap.py): Slicer tool for indicating the position of the difference (errors (FP, FN)) between the ground truth and prediction
       * [`metrics_computation.py`](ECEProject/Analysis/metrics_computation.py): Computation of the metrics for the analysis of the effect of the perturbations
       * [`metrics_properties_plot.py`](ECEProject/Analysis/metrics_properties_plot.py): Plottings for the scatter between each properties and each perturbation, and the boxplot of the perturbations effects
       * [`properties_computation.py`](ECEProject/Analysis/properties_computation.py): This script implements the computation of the properties of the segmented area and save them into csv files for the analysis
   * [`Doc/`](ECEProject/Doc/):
   * [`Preprocessing/`](ECEProject/Saliency/):
   * [`Training_and_inference/`](ECEProject/Saliency/):

