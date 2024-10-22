# Master's Thesis: Development, Robustness Analysis and Clinical Evaluation of a Deep Learning-based Segmentation Model for Head and Neck Cancer Patients

# Description
The goal of this project is to improve the detection of extracapsular extension (ECE) in head and neck cancers by employing a multimodal approach that combines FDG-PET and CT imaging. 
The focus here is solely on the segmentation step, which delineates the tumors for future classification.
A 3D U-Net deep learning model is implemented to segment primary and nodal tumors, using the HECKTOR Challenge 2022 dataset along with additional preprocessing and data augmentation techniques.

The primary objective is to evaluate the model's effectiveness and assess its robustness by analyzing the impact of various image perturbations on the segmentation process. Additionally, we collaborated with physicians to evaluate the model’s clinical relevance.

# Structure

 * [`Analysis/`](Analysis/):
     * [`clinical_plot.py`](Analysis/clinical_plot.py): Contains the plots for clinical evaluation grades and the correlation between Dice scores and evaluations.
     * [`correlation_plot.py`](Analysis/correlation_plot.py): Plots the four correlation heatmaps between properties of segmented areas and the delta Dice induced by perturbations.
     * [`dice_comparator.py`](Analysis/dice_comparator.py): Script computing the difference between the baseline and the corresponding perturbed case.
     * [`errorHeatMap.py`](Analysis/errorHeatMap.py): Slicer tool for indicating the position of differences (errors such as FP, FN) between the ground truth and predictions.
     * [`metrics_computation.py`](Analysis/metrics_computation.py): Computes the metrics for analyzing the effect of perturbations.
     * [`metrics_properties_plot.py`](Analysis/metrics_properties_plot.py): Generates scatter plots between properties and perturbations, and boxplots of the perturbation effects.
     * [`properties_computation.py`](Analysis/properties_computation.py): Computes properties of the segmented area and saves them in CSV files for analysis.
     * [`Inference`](Analysis/Inference): Directory containing the CSV files with the metrics, properties and clinical evaluation for each case, including both perturbations and the baseline.
 * [`Doc/`](Doc/):
     * [`MscThesis.pdf`](Doc/MscThesis.pdf): Master's thesis report.
 * [`Preprocessing/`](Preprocessing/):
     * [`preprocessing_images.py`](Preprocessing/preprocessing_images.py): Script containing all the preprocessing steps for the images.
     * [`hecktor2022`](Preprocessing/hecktor2022): Directory containing the CSV files from the dataset.
 * [`Training_and_Inference/`](Training_and_Inference/):
     * [`test_seg_3d_majority_voting.py`](Training_and_Inference/test_seg_3d_majority_voting.py): Implements inference using majority voting with all perturbations.
     * [`test_seg_3d_majority_voting_perturbed.py`](Training_and_Inference/test_seg_3d_majority_voting_perturbed.py): Implements inference focusing on the most impactful perturbations and saves input images for clinical evaluations.
     * [`test_set.py`](Training_and_Inference/test_set.py): Script for creating a random test set of 50 cases.
     * [`train_seg_3d_mixed.py`](Training_and_Inference/train_seg_3d_mixed.py): Implements the training of the model.

# License
The HECKTOR Challenge 2022 dataset is licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license. This allows for non-commercial use, distribution, and reproduction in any medium, provided proper credit is given to the original creators.

For more details, please refer to the HECKTOR Challenge website (https://hecktor.grand-challenge.org/) and the dataset repository documentation​


