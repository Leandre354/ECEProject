
import os
import glob
import sys
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import torch
import torch.nn as nn
import torch.nn.functional as F
import monai
from monai.data import ImageDataset, DataLoader, Dataset, ArrayDataset, ShuffleBuffer, CacheDataset
from monai.transforms import Compose, SqueezeDim, Transform, ScaleIntensity, EnsureType, ToTensor, LoadImage, \
    EnsureChannelFirst, RandAdjustContrast, Transpose, ResizeWithPadOrCrop, RandFlip, RandAffine, RandGaussianNoise, RandShiftIntensity, RandGaussianSmooth
from monai.networks.nets import UNet, BasicUNet, DynUNet
from monai.losses import DiceLoss, DiceCELoss
from monai.engines import SupervisedTrainer
from torch.optim import Adam
from torch import optim
import tracemalloc
import h5py
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
import monai.transforms as tf
import SimpleITK as sitk
import torchio.transforms as tiot
import torchio as tio


class Perturbations:
    def __init__(self, perturbation):
        self.perturbation = perturbation

    def __call__(self, img, lbl):
        Modality = 0  # 0:CT, 1:PET

        img_torchio = tio.ScalarImage(tensor=img[Modality][np.newaxis])
        img_torchio = self.perturbation(img_torchio)
        img[Modality] = img_torchio.tensor

        return img, lbl

def zero_mean(pet_image):
    pet_mean = np.mean(pet_image)
    pet_std = np.std(pet_image)
    pet_image_normalized = (pet_image - pet_mean) / pet_std
    return pet_image_normalized

def rescale_ct_image(ct_image):
    min_val = np.min(ct_image)
    max_val = np.max(ct_image)
    ct_image_rescaled = (ct_image - min_val) / (max_val - min_val)
    return ct_image_rescaled

class MedicalSlicesDataset(Dataset):
    def __init__(self, files_path, device, transforms=None, fold_idx=None, is_train=True, n_splits=5):
        self.device = device
        self.transforms = transforms
        self.path = files_path

        # Get list of .npz files
        self.names_ct = sorted(glob.glob(files_path + '*__CT.nii.gz'))
        self.names_pt = sorted(glob.glob(files_path + '*__PT.nii.gz'))
        self.names_seg = sorted(glob.glob(files_path + '*[!T].nii.gz'))

    def __len__(self):
        return len(self.names_ct)

    def __getitem__(self, idx):
        ct = nib.load(self.names_ct[idx]).get_fdata()
        pt = nib.load(self.names_pt[idx]).get_fdata()
        seg = nib.load(self.names_seg[idx]).get_fdata()
        pt = zero_mean(pt)
        ct = rescale_ct_image(ct)
        img = np.stack((ct, pt), axis=0).astype(np.float32)
        seg = np.expand_dims(seg, axis=0).astype(np.float32)

        if self.transforms:
            img, seg = self.transforms(img, seg)

        img = np.transpose(img, (0, 3, 1, 2))
        seg = np.transpose(seg, (0, 3, 1, 2))

        # Prepare segmentation masks
        seg0 = (seg == 0).astype(np.float32)
        seg1 = (seg == 1).astype(np.float32)
        seg2 = (seg == 2).astype(np.float32)
        seg = np.concatenate((seg0, seg1, seg2), axis=0)

        return {'img': img, 'seg': seg, 'name': self.names_seg[idx]}

def patch_based_inference_3d(batch_images, model, device, patch_size=(192, 192, 192), stride=(192, 192, 192)):
    batch_images = batch_images.to(device)

    b, c, d, h, w = batch_images.shape
    patch_depth, patch_height, patch_width = patch_size
    stride_depth, stride_height, stride_width = stride

    # Calculate padding to make depth, height, and width multiples of patch size
    pad_d = (patch_depth - d % patch_depth) if d % patch_depth != 0 else 0
    pad_h = (patch_height - h % patch_height) if h % patch_height != 0 else 0
    pad_w = (patch_width - w % patch_width) if w % patch_width != 0 else 0

    padding = (0, pad_w, 0, pad_h, 0, pad_d)  # Padding (W, H, D)
    batch_images_padded = nn.functional.pad(batch_images, padding)

    # Update dimensions after padding
    d_padded, h_padded, w_padded = batch_images_padded.shape[2:]

    predictions = torch.zeros((b, c + 1, d_padded, h_padded, w_padded), dtype=torch.float32).to(device)
    counts = torch.zeros((b, c + 1, d_padded, h_padded, w_padded), dtype=torch.float32).to(device)

    for z in range(0, d_padded - patch_depth + 1, stride_depth):
        for y in range(0, h_padded - patch_height + 1, stride_height):
            for x in range(0, w_padded - patch_width + 1, stride_width):
                patches = batch_images_padded[:, :, z:z + patch_depth, y:y + patch_height, x:x + patch_width]
                output_patches = model(patches)

                predictions[:, :, z:z + patch_depth, y:y + patch_height, x:x + patch_width] += output_patches
                counts[:, :, z:z + patch_depth, y:y + patch_height, x:x + patch_width] += 1

    # Normalize by counts
    predictions /= counts

    # Remove padding
    predictions = predictions[:, :, :d, :h, :w]

    return predictions

def load_models(model_paths, model_class, device):
    models = []
    for path in model_paths:
        model = model_class.to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        models.append(model)
    return models


def predict_with_majority_voting(models, batch_images, device):
    # Store all predictions from different models
    all_predictions = []

    for model in models:
        outputs = patch_based_inference_3d(batch_images, model, device)
        outputs = nn.functional.softmax(outputs, dim=1)
        all_predictions.append(outputs.cpu().numpy())

    # Convert list of predictions to numpy array
    all_predictions = np.stack(all_predictions, axis=0)  # Shape: (num_models, batch_size, num_classes, D, H, W)

    # Majority voting
    majority_voted_predictions = np.argmax(np.sum(all_predictions, axis=0), axis=1)  # Shape: (batch_size, D, H, W)

    one_hot = np.zeros((majority_voted_predictions.shape[0], 3, majority_voted_predictions.shape[1], majority_voted_predictions.shape[2], majority_voted_predictions.shape[3]), dtype=np.float32)
    for i in range(3):
        one_hot[:, i, :, :, :] = (majority_voted_predictions == i).astype(np.float32)

    return one_hot

def perform_inference_and_save(models, perturbation, output_dir, device):
    FILES = 'imagesTs_pp200/'

    test_dataset = MedicalSlicesDataset(FILES, device, transforms=perturbation)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred = AsDiscrete(threshold=0.5)

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with torch.no_grad():
        for batch_data in test_loader:
            inputs, labels, name = batch_data["img"].to(device), batch_data["seg"].to(device), batch_data["name"]

            # Apply perturbation
            #inputs, labels = perturbation(inputs, labels)

            # Predict using majority voting
            majority_voted_predictions = predict_with_majority_voting(models, inputs, device)

            # Convert to torch tensor for Dice metric calculation
            majority_voted_predictions = torch.tensor(majority_voted_predictions, dtype=torch.float32).to(device)

            # Post-processing: apply threshold to get discrete predictions
            majority_voted_predictions = post_pred(majority_voted_predictions)

            # Combine the segmentation outputs
            result_image = majority_voted_predictions[0, 1]
            result_image[majority_voted_predictions[0, 2] == 1] = 2

            result_image = np.transpose(result_image.cpu().numpy(), (0, 2, 1))
            result_image = sitk.GetImageFromArray(result_image)

            # Save the resulting image
            output_path = os.path.join(output_dir, ("".join(name)).split('/')[1])
            sitk.WriteImage(result_image, output_path)
             
            print("maj",majority_voted_predictions.shape)
            print("lbl",labels.shape)
            # Calculate Dice score
            dice_metric(y_pred=majority_voted_predictions, y=labels)
            temp = dice_metric.aggregate().item()
            print(f'Dice score for {name}: {temp}')

        # Aggregate Dice scores
        average_dice_score = dice_metric.aggregate().item()
        dice_metric.reset()

    print(f"Average Dice Score for perturbation {perturbation}: {average_dice_score}")
    return average_dice_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to the saved models for the 5 folds
model_paths = [
    "Models_inference/model_fold0.pth",
    "Models_inference/model_fold1.pth",
    "Models_inference/model_fold2.pth",
    "Models_inference/model_fold3.pth",
    "Models_inference/model_fold4.pth"
]
# Load the 5 models
models = load_models(model_paths, DynUNet(
    spatial_dims=3,
    in_channels=2,
    out_channels=3,
    kernel_size=[(3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)],
    strides=[(1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)],
    upsample_kernel_size=[(2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)],
    filters=[32, 64, 128, 256, 256, 256],
    norm_name="batch",
    dropout=0,
    deep_supervision=False,
    deep_supr_num=2,
    res_block=True
), device)

perturbations = [
    ('Baseline', None),
    ('Motion1', Perturbations(tiot.RandomMotion(degrees=10, translation=10, num_transforms=2))),
    ('Motion2', Perturbations(tiot.RandomMotion(degrees=20, translation=20, num_transforms=3))),
    ('Motion3', Perturbations(tiot.RandomMotion(degrees=30, translation=30, num_transforms=4))),
    ('Ghost1', Perturbations(tiot.RandomGhosting(num_ghosts=(4, 10), axes=(0, 1, 2), intensity=(0.5, 1), restore=0.02))),
    ('Ghost2', Perturbations(tiot.RandomGhosting(num_ghosts=(6, 15), axes=(0, 1, 2), intensity=(0.7, 1.5), restore=0.04))),
    ('Ghost3', Perturbations(tiot.RandomGhosting(num_ghosts=(8, 20), axes=(0, 1, 2), intensity=(1, 2), restore=0.06))),
    ('Spike1', Perturbations(tiot.RandomSpike(num_spikes=1, intensity=(1, 3)))),
    ('Spike2', Perturbations(tiot.RandomSpike(num_spikes=2, intensity=(2, 5)))),
    ('Spike3', Perturbations(tiot.RandomSpike(num_spikes=3, intensity=(3, 7)))),
    ('Bias1', Perturbations(tiot.RandomBiasField(coefficients=0.5, order=3))),
    ('Bias2', Perturbations(tiot.RandomBiasField(coefficients=0.75, order=3))),
    ('Bias3', Perturbations(tiot.RandomBiasField(coefficients=1.0, order=4))),
    ('Noise1', Perturbations(tiot.RandomNoise(mean=0, std=(0, 0.25)))),
    ('Noise2', Perturbations(tiot.RandomNoise(mean=0, std=(0.1, 0.5)))),
    ('Noise3', Perturbations(tiot.RandomNoise(mean=0, std=(0.2, 0.75)))),
    ('Blur1', Perturbations(tiot.RandomBlur(std=(0, 2)))),
    ('Blur2', Perturbations(tiot.RandomBlur(std=(1, 3)))),
    ('Blur3', Perturbations(tiot.RandomBlur(std=(2, 4))))
]

for name, perturb in perturbations:
    print(f'--{name}--')
    output_dir = f'Analysis/CT/{name}/'
    perform_inference_and_save(models, perturb, output_dir, device)


