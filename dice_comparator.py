
import numpy as np
import glob
import nibabel as nib

def dice_score(pred_mask, true_mask):

    # Compute intersection and union
    intersection = np.sum(pred_mask * true_mask)
    union = np.sum(pred_mask) + np.sum(true_mask)

    # Avoid division by zero
    if union == 0:
        return 1.0 if np.sum(pred_mask) == 0 and np.sum(true_mask) == 0 else 0.0

    dice = 2. * intersection / union
    return dice

def one_hot(image):
    new_image = (image == 1)
    new_image[image == 2] = 1
    return new_image


name = 'Spike1'

noisy_pred = sorted(glob.glob(f'Analysis/CT_P/{name}/*.nii.gz'))
gt = sorted(glob.glob(f'ImagesTs_pp200/*[!T].nii.gz'))
pred = sorted(glob.glob(f'Analysis/CT/Baseline/*.nii.gz'))

for i in range(0, len(gt), 1):
    prediction = nib.load(pred[i]).get_fdata()
    perturbed_prediction = nib.load(noisy_pred[i]).get_fdata()
    ground_truth = nib.load(gt[i]).get_fdata()

    prediction = one_hot(prediction)
    perturbed_prediction = one_hot(perturbed_prediction)
    ground_truth = one_hot(ground_truth)

    print("Base :", dice_score(prediction,ground_truth))
    print("perturbed :", dice_score(perturbed_prediction, ground_truth))
    print("Delta :", (dice_score(prediction,ground_truth) - dice_score(perturbed_prediction, ground_truth)))


