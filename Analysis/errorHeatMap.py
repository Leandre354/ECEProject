"""
File: errorHeatMap.py

Description:
This script implements a slicer tool for indicating the position of the difference (errors (FP, FN)) between the ground truth and prediction
"""
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import cv2
from matplotlib import colors

#Select the case
name = 'CHUM-014'

#Paths of ground truth and perturbed predictions
gt_path = f'Inference/Image200/Inputs/{name}.nii.gz'
prediction_paths = [
    f'Inference/Image200/Analysis/Baseline/{name}.nii.gz',  # Baseline
    f'Inference/Image200/Analysis/CT/Noise1/{name}.nii.gz',  # Noise
    f'Inference/Image200/Analysis/CT/Blur1/{name}.nii.gz',   # Blur
    f'Inference/Image200/Analysis/CT/Ghost1/{name}.nii.gz',  # Ghost
    f'Inference/Image200/Analysis/CT/Spike1/{name}.nii.gz',  # Spike
    f'Inference/Image200/Analysis/CT/Bias1/{name}.nii.gz',   # Bias
    f'Inference/Image200/Analysis/CT/Motion1/{name}.nii.gz'  # Motion
]

#Load the Nifti image
gt_img = nib.load(gt_path).get_fdata()
prediction_imgs = [nib.load(pred_path).get_fdata() for pred_path in prediction_paths]

# Compute error maps
error_maps = [gt_img - pred_img for pred_img in prediction_imgs]

# Initial slice index
slice_sel = 140  # Start at the middle slice

# Set up the figure and axis
fig, axes = plt.subplots(3, len(prediction_imgs), figsize=(20, 15))
plt.subplots_adjust(left=0.1, bottom=0.25)

#Titles for plots
titles = ['Base', 'Noise', 'Blur', 'Ghost', 'Spike', 'Bias', 'Motion']
img_gt = []
img_pred = []
img_err = []

for i, (ax_gt, ax_pred, ax_err, pred_img, error_map) in enumerate(
        zip(axes[0], axes[1], axes[2], prediction_imgs, error_maps)):
    cmap = colors.ListedColormap(['black', 'gray', 'white'])  # Map 0 to black, 1 to gray, 2 to white

    # Set the bounds and norm to ensure each value maps to a unique color
    bounds = [0, 0.5, 1.5, 2.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    # Ground truth
    im_gt = ax_gt.imshow(gt_img[:, :, slice_sel], cmap=cmap, norm=norm)
    ax_gt.set_title(f'GT {titles[i]}')
    ax_gt.axis('off')
    img_gt.append(im_gt)

    # Prediction
    im_pred = ax_pred.imshow(pred_img[:, :, slice_sel], cmap=cmap, norm=norm)
    ax_pred.set_title(f'Prediction {titles[i]}')
    ax_pred.axis('off')
    img_pred.append(im_pred)

    # Error map
    color = np.zeros((*error_map[:, :, slice_sel].shape, 3), dtype=np.uint8)
    color[error_map[:, :, slice_sel] == 1] = [255, 0, 0]  # Red for false negative
    color[error_map[:, :, slice_sel] == 2] = [255, 0, 0]
    color[error_map[:, :, slice_sel] == -1] = [255, 0, 255]  # Purple for false positive
    color[error_map[:, :, slice_sel] == -2] = [255, 0, 255]
    im_err = ax_err.imshow(color)
    ax_err.set_title(f'Error {titles[i]}')
    ax_err.axis('off')
    img_err.append(im_err)

# Slider for selecting slices
slice_slider_ax = plt.axes([0.15, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slice_slider = Slider(slice_slider_ax, 'Slice', 0, gt_img.shape[2] - 1, valinit=slice_sel, valstep=1)


# Update function for the slider
def update(val):
    slice_sel = int(slice_slider.val)
    for i, (im_gt, im_pred, im_err, pred_img, error_map) in enumerate(
            zip(img_gt, img_pred, img_err, prediction_imgs, error_maps)):
        # Update ground truth
        im_gt.set_data(gt_img[:, :, slice_sel])

        # Update prediction
        im_pred.set_data(pred_img[:, :, slice_sel])

        # Update error map
        color = np.zeros((*error_map[:, :, slice_sel].shape, 3), dtype=np.uint8)
        color[error_map[:, :, slice_sel] == 1] = [255,0 , 0]  # Red for false positive
        color[error_map[:, :, slice_sel] == 2] = [255, 0, 0]
        color[error_map[:, :, slice_sel] == -1] = [255, 0, 255]  # Purple for false negative
        color[error_map[:, :, slice_sel] == -2] = [255, 0, 255]
        im_err.set_data(color)
    fig.canvas.draw_idle()

# Connect the slider to the update function
slice_slider.on_changed(update)

plt.show()