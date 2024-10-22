# Master's Thesis: 

# Structure

 * [`Analysis/`](Analysis/):
     * [`clinical_plot.py`](Analysis/clinical_plot.py): Contains the plottings for the clinical evaluation grades and the correlation between dice and evaluation.
     * [`correlation_plot.py`](Analysis/correlation_plot.py): Plots of the four correlation heatmaps between properties of segmented area and delta dice induced by perturbations.
     * [`dice_comparator.py`](Analysis/dice_comparator.py): Script computing the difference between baseline and corresponding pertubated case.
     * [`errorHeatMap.py`](Analysis/errorHeatMap.py): Slicer tool for indicating the position of the difference (errors (FP, FN)) between the ground truth and prediction
     * [`metrics_computation.py`](Analysis/metrics_computation.py): Computation of the metrics for the analysis of the effect of the perturbations
     * [`metrics_properties_plot.py`](Analysis/metrics_properties_plot.py): Plottings for the scatter between each properties and each perturbation, and the boxplot of the perturbations effects
     * [`properties_computation.py`](Analysis/properties_computation.py): This script implements the computation of the properties of the segmented area and save them into csv files for the analysis
     * [`Inference`](Analysis/Inference): Directory containing the csv files with the metrics and properties computed of each cases with each perturbations and the baseline.
 * [`Doc/`](Doc/):
     * [`MscThesis.pdf`](Doc/MscThesis.pdf): Thesis report.
 * [`Preprocessing/`](Preprocessing/):
     * [`preprocessing_images.py`](Preprocessing/preprocessing_images.py): Script containing all the preprocessing steps of the images.
     * [`hecktor2022`](Preprocessing/hecktor2022): Directory containing the csv files from the dataset.
 * [`Training_and_inference/`](Training_and_inference/):
     * [`test_seg_3d_majority_voting.py`](Training_and_inference/test_seg_3d_majority_voting.py): Implementation of the inference with the majority voting and all the pertubations.
     * [`test_seg_3d_majority_voting_perturbed.py`](Training_and_inference/test_seg_3d_majority_voting_perturbed.py): Implementation of inference but specific on the most impactful perturbations and saves input images for the clinical evaluations.
     * [`test_set.py`](Training_and_inference/test_set.py): Script for creating the random test set of 50 cases.
     * [`train_seg_3d_mixed.py`](Training_and_inference/train_seg_3d_mixed.py): Implementation of the training of the model.

