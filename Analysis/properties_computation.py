"""
File: properties_computation.py

Description:
This script implements the computation of the properties of the segmented area and save them into csv files for the analysis
"""
import numpy as np
from pymia.evaluation import metric as pymia_metric
from pymia.evaluation.evaluator import SegmentationEvaluator
from pymia.evaluation import writer as pymia_writer
from pymia.evaluation.writer import StatisticsAggregator as pymia_aggregator
import nibabel as nib
import glob
from scipy import ndimage
from scipy import stats
import pandas as pd
import SimpleITK as sitk

def compute_compactness(labeled_array, num_features):
    surface_areas = []
    compactness_values = []

    # Loop through each feature (labeled object)
    for i in range(1, num_features + 1):
        # Create a binary mask for the current object
        object_mask = (labeled_array == i)

        # Compute volume (number of pixels in the object)
        volume = np.sum(object_mask)

        # Compute surface area by eroding the object and XORing with the original object
        eroded = ndimage.binary_erosion(object_mask)
        surface_area = np.sum(object_mask ^ eroded)

        # Compute compactness
        compactness = ((surface_area / 20) ** 3) / (36 * np.pi * ((volume / 20) ** 2))

        # Store the surface area and compactness
        surface_areas.append(surface_area)
        compactness_values.append(compactness)

    # Compute the mean surface area and mean compactness
    return np.mean(surface_areas), np.mean(compactness_values)

def dice_score(pred_mask, true_mask):

    # Compute intersection and union
    intersection = np.sum(pred_mask * true_mask)
    union = np.sum(pred_mask) + np.sum(true_mask)

    # Avoid division by zero
    if union == 0:
        return 1.0 if np.sum(pred_mask) == 0 and np.sum(true_mask) == 0 else 0.0

    dice = 2. * intersection / union
    return dice

def calculate_additional_metrics(ground_truth, ct_image, pet_image):
    metrics = []
    num_channels = 2 # Assuming channels are labeled as 1, 2, etc.

    for channel in range(num_channels, 0, -1):
        ground_truth_channel = (ground_truth == channel)
        ground_truth_bool = ground_truth_channel.astype(bool)

        # Check if there is any foreground data
        if np.sum(ground_truth_bool) == 0:
            # Skip if no foreground data is present
            metrics.append({
                'File Name': file_name,
                'Volume': np.nan,
                'Surface Area': np.nan,
                'Compactness': np.nan,
                'Distance from Center': np.nan,
                'Boundary Length': np.nan,
                'CT Intensity Variability': np.nan,
                'PET Intensity Variability': np.nan,
                'CT Foreground-to-Background Contrast': np.nan,
                'PET Foreground-to-Background Contrast': np.nan,
                'SUVmax': np.nan,
                'CT Number (HU)': np.nan,
                'Region Connectivity': np.nan,
                'Entropy': np.nan
            })
            continue

        # Compute metrics for the current channel
        labeled_array, num_features = ndimage.label(ground_truth_bool)
        volume = np.sum(ground_truth_bool)
        surface_area, compactness = compute_compactness(labeled_array, num_features)
        centroid = ndimage.center_of_mass(ground_truth_bool)
        image_center = np.array(ground_truth.shape) / 2
        distance_from_center = np.linalg.norm(np.array(centroid) - image_center)
        boundary_length = np.sum(ndimage.sobel(ground_truth_bool))
        ct_intensity_variability = np.std(ct_image[ground_truth_bool])
        pet_intensity_variability = np.std(pet_image[ground_truth_bool])
        ct_foreground_mean = np.mean(ct_image[ground_truth_bool])
        ct_background_mean = np.mean(ct_image[~ground_truth_bool])
        pet_foreground_mean = np.mean(pet_image[ground_truth_bool])
        pet_background_mean = np.mean(pet_image[~ground_truth_bool])
        suvmax = np.max(pet_image[ground_truth_bool]) #PET pixel are SUVs
        ct_number = ct_foreground_mean
        pt_number = pet_foreground_mean

        region_connectivity = num_features
        flattened_mask = ground_truth_bool.flatten()
        histogram, _ = np.histogram(flattened_mask, bins=np.arange(0, 3))
        entropy = stats.entropy(histogram)

        # Store metrics for the current channel
        metrics.append({
            'File Name': file_name,
            'Volume': volume,
            'Surface Area': surface_area,
            'Compactness': compactness,
            'Distance from Center': distance_from_center,
            'Boundary Length': boundary_length,
            'CT Intensity Var': ct_intensity_variability,
            'PET Intensity Var': pet_intensity_variability,
            'CT F/B Contrast': ct_foreground_mean - ct_background_mean,
            'PET F/B Contrast': pet_foreground_mean - pet_background_mean,
            'SUVmax': suvmax,
            'CT Number (HU)': ct_number,
            'PET Number': pt_number,
            'Regions': region_connectivity,
            'Entropy': entropy
        })

    return metrics
# Initialize metrics
dice_metric = pymia_metric.DiceCoefficient()
Hausdorff_distance = pymia_metric.HausdorffDistance()
Jaccard_index = pymia_metric.JaccardCoefficient()
Volume = pymia_metric.ReferenceVolume()
Sensitivity = pymia_metric.Sensitivity()
Specificity = pymia_metric.Specificity()
Accuracy = pymia_metric.Accuracy()
TP = pymia_metric.TruePositive()
TN = pymia_metric.TrueNegative()
FP = pymia_metric.FalsePositive()
FN = pymia_metric.FalseNegative()


labels = {1: 'Primary', 2: 'Nodal'}

# Create evaluator
Modality = 'PET'
Names = ['Bias1', 'Bias2', 'Bias3', 'Blur1', 'Blur2', 'Blur3',
         'Motion1', 'Motion2', 'Motion3', 'Ghost1', 'Ghost2', 'Ghost3',
         'Noise1', 'Noise2', 'Noise3', 'Spike1', 'Spike2', 'Spike3']

for Name in Names:
    evaluator = SegmentationEvaluator(metrics=[dice_metric, Hausdorff_distance, Jaccard_index, Volume, Sensitivity, Specificity, Accuracy, TP, TN, FP, FN],labels=labels)
    print('---' + Name + '---')
    #result_file = f'Inference/Image200/Analysis/{Modality}/{Name}/results_{Name}_pymia.csv'
    #result_file2 = f'Inference/Image200/Analysis/{Modality}/{Name}/results_{Name}_add.csv'
    result_file = 'Inference/Image200/Analysis/Baseline/results_pymia.csv'
    files = sorted(glob.glob(f'Inference/Image200/Inputs/*[!T].nii.gz'))
    #files2 = sorted(glob.glob(f'Inference/Image200/Analysis/{Modality}/{Name}/*.nii.gz'))
    files2 = sorted(glob.glob(f'Inference/Image200/Analysis/Baseline/*.nii.gz'))
    ct_files = sorted(glob.glob(f'Inference/Image200/Inputs/*CT.nii.gz'))
    pt_files = sorted(glob.glob(f'Inference/Image200/Inputs/*PT.nii.gz'))
    all_results = []
    predictions = []
    ground_truths = []
    for i in range(0, len(files), 1):
        print(f'{i}/{len(files)}')
        file_name = files[i].split('.')[0].split('\\')[1]
        prediction = nib.load(files2[i]).get_fdata()
        ground_truth = nib.load(files[i]).get_fdata()
        CT = nib.load(ct_files[i]).get_fdata()
        PT = nib.load(pt_files[i]).get_fdata()
        #evaluator.evaluate(prediction, ground_tr
        # uth, files[i].split('.')[0].split('\\')[1])

        # Calculate additional metrics
        #additional_metrics = calculate_additional_metrics(prediction, CT, PT)

        #for metrics in additional_metrics:
            #all_results.append(metrics)

        pad_width_pred = 310 - prediction.shape[2]
        pad_width_gt = 310 - ground_truth.shape[2]

        padded_pred = np.pad(prediction, ((0, 0), (0, 0), (0, pad_width_pred)), mode='constant', constant_values=0)
        padded_gt = np.pad(ground_truth, ((0, 0), (0, 0), (0, pad_width_gt)), mode='constant', constant_values=0)

        binary_pred = (padded_pred > 0).astype(int)
        binary_gt = (padded_gt > 0).astype(int)

        predictions.append(binary_pred)
        ground_truths.append(binary_gt)

    # Stack and aggregate using logical OR across the stacked dimension
    aggregated_pred = np.stack(predictions, axis=0)
    aggregated_gt = np.stack(ground_truths, axis=0)

    # Compute Dice Aggregate
    score = dice_score(aggregated_pred, aggregated_gt)

    print("dice agg",score)


    #pymia_writer.CSVWriter(result_file).write(evaluator.results)
    #df = pd.DataFrame(all_results)
    #df = df[['File Name'] + [col for col in df.columns if col != 'File Name']]
    #df.to_csv(result_file2, index=False)





