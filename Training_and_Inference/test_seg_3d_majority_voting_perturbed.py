
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
    """
        Class to apply a specified perturbation to the input image.

        Args:
            perturbation: A function or transformation to apply to the image.

        Methods:
            __call__(img, lbl):
                Applies the perturbation to the CT modality of the image.

        Attributes:
            perturbation: The transformation function to be applied to the image.
    """
    def __init__(self, perturbation):
        """
                   Initialize the Perturbations class with a given perturbation function.

                   Args:
                       perturbation: A transformation function that modifies the image.
                """
        self.perturbation = perturbation

    def __call__(self, img, lbl):
        """
                    Apply perturbation to the CT modality of the image.

                    Args:
                        img: The input image array containing both CT and PET modalities.
                             The 0th index corresponds to the CT modality.
                        lbl: The corresponding label array.

                    Returns:
                        img, lbl: The modified image and label, where the perturbation has been applied to the CT modality.
                """
        Modality = 1  # 0:CT, 1:PET

        img_torchio = tio.ScalarImage(tensor=img[Modality][np.newaxis])
        img_torchio = self.perturbation(img_torchio)
        img[Modality] = img_torchio.tensor

        return img, lbl

def zero_mean(pet_image):
    """
            Normalize the PET image to have zero mean and unit variance.

            Args:
                pet_image: The PET image array to be normalized.

            Returns:
                pet_image_normalized: The normalized PET image where the mean is zero and standard deviation is one.
        """
    pet_mean = np.mean(pet_image)
    pet_std = np.std(pet_image)
    pet_image_normalized = (pet_image - pet_mean) / pet_std
    return pet_image_normalized

def rescale_ct_image(ct_image):
    """
           Rescale the CT image to the range [0, 1].

           Args:
               ct_image: The CT image array to be rescaled.

           Returns:
               ct_image_rescaled: The CT image array with values rescaled to the [0, 1] range based on the min and max values.
       """
    min_val = np.min(ct_image)
    max_val = np.max(ct_image)
    ct_image_rescaled = (ct_image - min_val) / (max_val - min_val)
    return ct_image_rescaled

class MedicalSlicesDataset(Dataset):
    """
           Custom PyTorch dataset class for loading medical image slices.

           Args:
               files_path: Path to the folder containing the medical image files (.nii.gz).
               device: The device (CPU/GPU) where the data will be loaded.
               transforms: Optional transformations to apply to the images and labels.
               fold_idx: Index for splitting the data (useful for cross-validation).
               is_train: Boolean flag indicating whether the dataset is for training or testing.
               n_splits: Number of splits for cross-validation (default is 5).

           Attributes:
               names_ct: Sorted list of file paths for CT images.
               names_pt: Sorted list of file paths for PET images.
               names_seg: Sorted list of file paths for segmentation masks.
       """
    def __init__(self, files_path, device, transforms=None, fold_idx=None, is_train=True, n_splits=5):
        """
                    Initialize the dataset by loading the file paths and setting the device and transforms.

                    Args:
                        files_path: Path to the folder containing the .nii.gz files for CT, PET, and segmentation images.
                        device: The device (CPU or GPU) to which the images will be loaded.
                        transforms: Optional transformations to apply to the images and segmentation masks.
                        fold_idx: Optional index for cross-validation fold.
                        is_train: Boolean indicating whether this dataset is for training or testing purposes.
                        n_splits: Number of folds for cross-validation.
                """
        self.device = device
        self.transforms = transforms
        self.path = files_path

        # Get list of .npz files
        self.names_ct = sorted(glob.glob(files_path + '*__CT.nii.gz'))
        self.names_pt = sorted(glob.glob(files_path + '*__PT.nii.gz'))
        self.names_seg = sorted(glob.glob(files_path + '*[!T].nii.gz'))

    def __len__(self):
        """
                    Return the total number of images in the dataset.

                    Returns:
                        int: The number of CT image files, which is the same for PET and segmentation files.
                """
        return len(self.names_ct)

    def __getitem__(self, idx):
        """
                    Retrieve and preprocess an image and its corresponding segmentation mask.

                    Args:
                        idx: Index of the image to be loaded.

                    Returns:
                        dict: A dictionary containing:
                            - 'img': The preprocessed CT and PET images stacked as a 4D tensor.
                            - 'seg': The segmentation mask split into 3 channels (one-hot encoded format).
                            - 'name': The file name of the segmentation mask for reference.
                """
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
    """
            Perform patch-based 3D inference using a sliding window approach.

            Args:
                batch_images: The input 5D tensor of shape (B, C, D, H, W) where B is the batch size,
                              C is the number of channels, and D, H, W are the depth, height, and width of the images.
                model: The trained model used for inference.
                device: The device (CPU/GPU) where the computation will take place.
                patch_size: The size of the 3D patches (depth, height, width) used for inference (default is (192, 192, 192)).
                stride: The step size for moving the sliding window (default is (192, 192, 192)).

            Returns:
                predictions: The output predictions from the model for the entire volume, aggregated from patches.
        """
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
    """
            Load multiple models from the provided file paths.

            Args:
                model_paths: A list of file paths to the saved model checkpoints.
                model_class: The class of the model to be instantiated for each checkpoint.
                device: The device (CPU or GPU) on which the models will be loaded.

            Returns:
                models: A list of loaded model instances, each set to evaluation mode.
        """
    models = []
    for path in model_paths:
        model = model_class.to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        models.append(model)
    return models


def predict_with_majority_voting(models, batch_images, device):
    """
            Perform model ensemble prediction using majority voting.

            Args:
                models: A list of trained models to be used for predictions.
                batch_images: A batch of input images to be predicted on, with shape (B, C, D, H, W).
                device: The device (CPU or GPU) on which the computation will take place.

            Returns:
                one_hot: The final majority-voted predictions in one-hot encoded format,
                         with shape (batch_size, num_classes, D, H, W).
        """
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
    """
            Perform inference using multiple models with the 3 most degenerative perturbation and save the input perturbed images for the clinical evaluation

            Args:
                models: A list of trained models to be used for prediction.
                perturbation: the set of perturbation applied for the analysis of robustness
                output_dir: Directory where the output segmentation results will be saved.
                device: The device (CPU or GPU) on which the computation will take place.

            Returns:
                average_dice_score: The average Dice score across all test samples.
        """
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

            inputs_ct = inputs[0, 0]
            inputs_pet = inputs[0, 1]

            inputs_ct = np.transpose(inputs_ct.cpu().numpy(), (0, 2, 1))
            inputs_pet = np.transpose(inputs_pet.cpu().numpy(), (0, 2, 1))
            
            inputs_ct = sitk.GetImageFromArray(inputs_ct)
            inputs_pet = sitk.GetImageFromArray(inputs_pet)

            base_name = ("".join(name)).split('/')[1]
            base_name = base_name.split('.')[0]

            output_path = os.path.join(output_dir, base_name + "__CT.nii.gz")
            sitk.WriteImage(inputs_ct, output_path)

            output_path = os.path.join(output_dir, base_name + "__PT.nii.gz")
            sitk.WriteImage(inputs_pet, output_path)

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
    ('Spike1', Perturbations(tiot.RandomSpike(num_spikes=1, intensity=(1, 3)))),
    ('Spike2', Perturbations(tiot.RandomSpike(num_spikes=2, intensity=(2, 5)))),
    ('Spike3', Perturbations(tiot.RandomSpike(num_spikes=3, intensity=(3, 7)))),
    ('Bias1', Perturbations(tiot.RandomBiasField(coefficients=0.5, order=3))),
    ('Bias2', Perturbations(tiot.RandomBiasField(coefficients=0.75, order=3))),
    ('Bias3', Perturbations(tiot.RandomBiasField(coefficients=1.0, order=4))),
    ('Noise1', Perturbations(tiot.RandomNoise(mean=0, std=(0, 0.25)))),
    ('Noise2', Perturbations(tiot.RandomNoise(mean=0, std=(0.1, 0.5)))),
    ('Noise3', Perturbations(tiot.RandomNoise(mean=0, std=(0.2, 0.75))))
]

for name, perturb in perturbations:
    print(f'--{name}--')
    output_dir = f'Analysis/PET_P/{name}/'
    perform_inference_and_save(models, perturb, output_dir, device)


