
import numpy as np
import glob
import nibabel as nib

def dice_score(pred_mask, true_mask):
    """
       Calculate the Dice coefficient between the predicted and ground truth masks.

       Args:
           pred_mask: The predicted segmentation mask (binary or one-hot encoded).
           true_mask: The ground truth segmentation mask (binary or one-hot encoded).

       Returns:
           dice: The Dice score, a value between 0 and 1, where 1 indicates perfect overlap.
    """

    # Compute intersection and union
    intersection = np.sum(pred_mask * true_mask)
    union = np.sum(pred_mask) + np.sum(true_mask)

    # Avoid division by zero
    if union == 0:
        return 1.0 if np.sum(pred_mask) == 0 and np.sum(true_mask) == 0 else 0.0

    dice = 2. * intersection / union
    return dice

def one_hot(image):
    """
       Convert a labeled image to a one-hot encoded binary format.

       Args:
           image: A labeled image, where different integer values represent different classes.

       Returns:
           new_image: A binary one-hot encoded image, where all classes greater than or equal to 1 are set to 1.
    """
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


