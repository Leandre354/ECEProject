# -*- coding: utf-8 -*-


import os
import glob
import sys
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import torch
import torch.nn as nn
import monai
from monai.data import ImageDataset, DataLoader, Dataset, ArrayDataset, ShuffleBuffer, CacheDataset
import monai.transforms as tf
from monai.networks.nets import UNet, BasicUNet, SegResNet, DynUNet
from monai.losses import DiceLoss, DiceCELoss, GeneralizedDiceLoss, DiceFocalLoss, GeneralizedDiceFocalLoss
from monai.engines import SupervisedTrainer
from torch.optim import Adam, AdamW
from torch import optim
import tracemalloc
from saveslicedimages import read_hdf5
import h5py
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts, LambdaLR
from torch.utils.data import WeightedRandomSampler
import wandb
import random
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, EnsureType
from monai.data import decollate_batch
import torch.multiprocessing as mp
from sklearn.model_selection import KFold
import time
from torch.cuda.amp import autocast, GradScaler


wandb.login(key='9d5812a471d8d3122966d9d3bb7b41e4d2ff2e3d')

wandb.init(project='ece-thesis',
config={
    'learning_rate': 1e-4,
    'weight_decay': 3e-5,
    'batch_size': 2,
    'num_epochs': 100,
    'dropout': 0,
    'model': 'DynUNet',
    'fold': 0,
    'augmentation_proba': 0.2,
    'lambda_dice': 0.5,
    'lambda_ce': 0.5,
    'deep_supervision': False,
    'deep_supervision_layers': 2,
    'name': 'model_f0_new'
})
config = wandb.config

class JointTransform:
    def __init__(self, augmentation_proba, device):
        self.device = device
        self.rand_flip = tf.RandFlip(spatial_axis=1, prob=augmentation_proba)
        self.rand_affine = tf.RandAffine(prob=augmentation_proba, rotate_range=(-np.pi/8, np.pi/8), scale_range=(-0.1, 0.1), translate_range=(-10, 10))
        self.rand_adjust_contrast = tf.RandAdjustContrast(prob=augmentation_proba, gamma=(0.7, 1.3))
        self.rand_shift_intensity = tf.RandShiftIntensity(offsets=(-0.1,0.1), prob=augmentation_proba)
        self.rand_gaussian_noise = tf.RandGaussianNoise(prob=augmentation_proba, mean=0.0, std=0.1)
        self.rand_gaussian_smooth = tf.RandGaussianSmooth(prob=augmentation_proba, sigma_x=[0.5, 1], sigma_y=[0.5, 1], sigma_z=[0.1, 1])
        self.rand_crop = tf.RandCropByLabelClasses(spatial_size=(192, 192, 192), num_classes=3, ratios=[0.1, 0.45, 0.45], warn=True)
        self.rand_crop2 = tf.RandCropByLabelClasses(spatial_size=(192, 192, 192), num_classes=3, ratios=[0.1, 0, 0.9], warn=True)
        self.rand_crop3 = tf.RandCropByLabelClasses(spatial_size=(192, 192, 192), num_classes=3, ratios=[0.1, 0.9, 0], warn=True)

    def __call__(self, img, lbl):
        fusion = np.concatenate((img, lbl), axis=0)

        if np.any((lbl == 1)) == True and np.any((lbl == 2)) == True:
            fusion = self.rand_crop(fusion, label=lbl)[0]
        elif np.any((lbl == 1)) == True and np.any((lbl == 2)) == False:
            fusion = self.rand_crop3(fusion, label=lbl)[0]
        elif np.any((lbl == 1)) == False and np.any((lbl == 2)) == True:
            fusion = self.rand_crop2(fusion, label=lbl)[0]
        
        fusion = self.rand_flip(fusion)
        fusion = self.rand_affine(fusion)

        img = fusion[0:2]
        lbl = fusion[2:3]

        # Apply remaining CT-specific transformations sequentially
        img[0] = self.rand_adjust_contrast(img[0][np.newaxis])
        img[0] = self.rand_shift_intensity(img[0][np.newaxis])
        img[0] = self.rand_gaussian_noise(img[0][np.newaxis])
        img[0] = self.rand_gaussian_smooth(img[0][np.newaxis])

        return img, lbl

class JointTransformVal:
    def __init__(self, device):
        self.device = device
        self.rand_crop = tf.RandCropByLabelClasses(spatial_size=(192, 192, 192), num_classes=3, ratios=[0.33, 0.33, 0.33], warn=False)
        self.rand_crop2 = tf.RandCropByLabelClasses(spatial_size=(192, 192, 192), num_classes=3, ratios=[0.33, 0, 0.66], warn=False)
        self.rand_crop3 = tf.RandCropByLabelClasses(spatial_size=(192, 192, 192), num_classes=3, ratios=[0.33, 0.66, 0], warn=False)

    def __call__(self, img, lbl):
        fusion = np.concatenate((img, lbl), axis=0)

        if np.any((lbl == 1)) == True and np.any((lbl == 2)) == True:
            fusion = self.rand_crop(fusion, label=lbl)[0]
        elif np.any((lbl == 1)) == True and np.any((lbl == 2)) == False:
            fusion = self.rand_crop3(fusion, label=lbl)[0]
        elif np.any((lbl == 1)) == False and np.any((lbl == 2)) == True:
            fusion = self.rand_crop2(fusion, label=lbl)[0]

        img = fusion[0:2]
        lbl = fusion[2:3]

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

        # Perform 5-fold split
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(kf.split(self.names_ct))

        # Choose the fold to use
        train_index, val_index = splits[fold_idx]

        # Select training or validation files based on is_train flag
        if is_train:
            self.names_ct = [self.names_ct[i] for i in train_index]
            self.names_pt = [self.names_pt[i] for i in train_index]
            self.names_seg = [self.names_seg[i] for i in train_index]
        else:
            self.names_ct = [self.names_ct[i] for i in val_index]
            self.names_pt = [self.names_pt[i] for i in val_index]
            self.names_seg = [self.names_seg[i] for i in val_index]

    def __len__(self):
        return len(self.names_ct)

    def __getitem__(self, idx):
        ct = nib.load(self.names_ct[idx]).get_fdata()
        pt = nib.load(self.names_pt[idx]).get_fdata()
        seg = nib.load(self.names_seg[idx]).get_fdata()
        print(self.names_seg[idx])
        #pt = zero_mean(pt)
        #ct = rescale_ct_image(ct)
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

class CombinedLossWithDeepSupervision(nn.Module):
    def __init__(self, dice_loss, ce_loss, deep_supervision_weights, lambda_dice=0.5, lambda_ce=0.5):
        super(CombinedLossWithDeepSupervision, self).__init__()
        self.dice_loss = dice_loss
        self.ce_loss = ce_loss
        self.deep_supervision_weights = deep_supervision_weights
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce

    def forward(self, outputs, target):
        if config['deep_supervision'] == True:
            outputs = outputs.permute(1, 0, 2, 3, 4, 5)
            final_output = outputs[0]
            deep_supervision_outputs = outputs[1:]
        else:
            final_output = outputs
            
        final_dice_loss = self.dice_loss(final_output, target)
        final_ce_loss = self.ce_loss(final_output, torch.argmax(target, dim=1))

        final_loss = self.lambda_dice * final_dice_loss + self.lambda_ce * final_ce_loss

        if config['deep_supervision'] == True:
            deep_supervision_losses = [
                w * (self.lambda_dice * self.dice_loss(output, target) + self.lambda_ce * self.ce_loss(output, target))
                for output, w in zip(deep_supervision_outputs, self.deep_supervision_weights)
            ]
            total_loss = final_loss + sum(deep_supervision_losses)
        else:
            total_loss = final_loss

        return total_loss

class MonitoringLoss(nn.Module):
    def __init__(self, dice_loss, ce_loss):
        super(MonitoringLoss, self).__init__()
        self.dice_loss = dice_loss
        self.ce_loss = ce_loss

    def forward(self, outputs, target):

        final_dice_loss = self.dice_loss(outputs, target)
        ce_loss = self.ce_loss(outputs, torch.argmax(target, dim=1))

        return final_dice_loss, ce_loss

def normalize_image(img):
    img_min, img_max = img.min(), img.max()
    return (img - img_min) / (img_max - img_min)

def patch_based_inference_3d(batch_images, model, device, patch_size=(64, 192, 192), stride=(64, 192, 192)):
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

scaler = torch.cuda.amp.GradScaler()

FILES = 'imagesTr_pp200/'

joint_transform = JointTransform(augmentation_proba=config['augmentation_proba'], device=device)
joint_transform2 = JointTransformVal(device=device)

dataset = MedicalSlicesDataset(FILES, device, transforms=joint_transform, fold_idx=config['fold'], is_train=True)
val_dataset = MedicalSlicesDataset(FILES, device, transforms=joint_transform2, fold_idx=config['fold'], is_train=False)

loader = DataLoader(dataset, batch_size=config['batch_size'], sampler=None, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=1, num_workers=8, pin_memory=True)

model = DynUNet(
    spatial_dims=3,
    in_channels=2,  # CT and PET channels
    out_channels=3,  # Segmentation classes
    kernel_size=[(3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)],  # Number of layers
    strides=[(1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)],
    upsample_kernel_size=[(2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)],
    filters=[32, 64, 128, 256, 256, 256],  # Number of filters in each layer
    norm_name="batch",
    dropout=config['dropout'],
    deep_supervision=config['deep_supervision'],
    deep_supr_num=config['deep_supervision_layers'],
    res_block=True
).to(device)

wandb.watch(model, log='all', log_freq=100)

optimizer2 = torch.optim.SGD(
    model.parameters(),
    lr=config['learning_rate'],
    momentum=0.99,
    weight_decay=config['weight_decay'],
    nesterov=True,
)

optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: (1 - epoch / config['num_epochs']) ** 0.9)

#dice_loss_one_class = GeneralizedDiceFocalLoss(include_background=True, softmax=True, weight=torch.tensor([0.1,1,0], dtype=torch.float32))
#dice_loss_two_class = GeneralizedDiceFocalLoss(include_background=True, softmax=True, weight=torch.tensor([0.1,0,1], dtype=torch.float32))
#dice_loss = GeneralizedDiceFocalLoss(include_background=True, softmax=True, weight=torch.tensor([0.1,1,1], dtype=torch.float32))
dice_loss = DiceLoss(sigmoid=False, softmax=True, include_background=True)
#dice_loss = GeneralizedDiceLoss(include_background=True, softmax=True)
ce_loss = nn.CrossEntropyLoss()

#dice_loss2_one_class = GeneralizedDiceFocalLoss(include_background=False, softmax=True, weight=torch.tensor([1,0], dtype=torch.float32))
#dice_loss2_two_class = GeneralizedDiceFocalLoss(include_background=False, softmax=True, weight=torch.tensor([0,1], dtype=torch.float32))
#dice_loss2 = GeneralizedDiceFocalLoss(include_background=False, softmax=True, weight=torch.tensor([1,1], dtype=torch.float32))
#dice_loss2 = GeneralizedDiceLoss(include_background=False, softmax=True)
dice_loss2 = DiceLoss(sigmoid=False, softmax=True, include_background=False) #Exclude background for the monitoring dice

loss_function = CombinedLossWithDeepSupervision(dice_loss, ce_loss, deep_supervision_weights=[0.5, 0.25], lambda_dice=config['lambda_dice'], lambda_ce=config['lambda_ce'])
ce_loss2 = nn.CrossEntropyLoss()
loss_function2 = MonitoringLoss(dice_loss2, ce_loss2)

dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

post_pred = AsDiscrete(threshold=0.5)  # Assuming 2 classes

num_epochs = config['num_epochs']

for epoch in range(num_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{num_epochs}")
    model.train()
    epoch_loss, step, step2 = 0, 0, 0
    epoch_loss2 = 0
    epoch_ce_loss2 = 0
    temp = 1
    temp2 = 1
    for batch_data in loader:
        step += 1
        inputs, labels, name = batch_data["img"].to(device), batch_data["seg"].to(device), batch_data["name"]
        print(f"{step*config['batch_size']} images")
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            if config['deep_supervision'] == True:
                loss2, ce_loss2 = loss_function2(outputs.permute(1, 0, 2, 3, 4, 5)[0], labels)
            else:
                loss2, ce_loss2 = loss_function2(outputs, labels)

        if config['deep_supervision'] == True:
            outputs_show = nn.functional.softmax(outputs, dim=2)
            outputs_show = outputs_show.permute(1, 0, 2, 3, 4, 5)[0]
        else:
            outputs_show = nn.functional.softmax(outputs, dim=1)

        if epoch % 4 == 0 and temp == 1:
            # Initialize a list to hold the 2D slices
            train_sample_slices = []
    
            # Assuming outputs_show, labels, and inputs are 4D tensors with shape [batch, channels, depth, height, width]
            found_valid_slice = False
            ind = 0
            while not found_valid_slice:
                ind += 1
                # Select a random slice index
                random_slice_idx = random.randint(0, outputs_show.shape[2] - 1)
                label_slice = labels[0, 1:3, random_slice_idx, :, :].cpu().detach().numpy()
                # Check if the selected slice contains any label
                if np.any(label_slice) or ind > 50:
                    found_valid_slice = True

            # Loop through the channels
            for i in range(3):  # Assuming 3 channels for inputs, labels, and outputs
                # Extract the 2D slice for the input, label, and output
                if i < 2:
                    if i == 1:
                        input_slice = normalize_image(inputs[0, i, random_slice_idx, :, :].cpu().detach().numpy())
                    else:
                    	input_slice = inputs[0, i, random_slice_idx, :, :].cpu().detach().numpy()
                label_slice = labels[0, i, random_slice_idx, :, :].cpu().detach().numpy()
                output_slice = outputs_show[0, i, random_slice_idx, :, :].cpu().detach().numpy()

                # Append the slices as images
                if i < 2:
                    train_sample_slices.append(wandb.Image(input_slice, caption=f"Input Channel {i} - Slice {random_slice_idx}"))
                train_sample_slices.append(wandb.Image(label_slice, caption=f"Label Channel {i} - Slice {random_slice_idx}"))
                train_sample_slices.append(wandb.Image(output_slice, caption=f"Output Channel {i} - Slice {random_slice_idx}"))

                # Log the 2D slices
            wandb.log({f"train_sample_slices_{epoch}": train_sample_slices})
    
            # Reset temp variable
            temp = 0

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        epoch_loss2 += loss2.item()
        epoch_ce_loss2 += ce_loss2.item()
        
    epoch_loss /= step
    epoch_loss2 /= step
    epoch_ce_loss2 /= step
    
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": epoch_loss,
        "train_dice": epoch_loss2,
        "train_ce": epoch_ce_loss2
    })

    model.eval()
    val_loss = 0.0
    val_ce_loss = 0.0
    val_loss2 = 0.0
    index = 0
    with torch.no_grad():
        for val_data in val_loader:
            index += 1
            print(f'{index} image')
            val_inputs, val_labels = val_data["img"].to(device), val_data["seg"].to(device)
            
            with autocast():
                val_outputs = model(val_inputs)
                #val_outputs = patch_based_inference_3d(val_inputs, model, device)
                loss, loss_ce = loss_function2(val_outputs, val_labels)

            val_outputs_show = nn.functional.softmax(val_outputs, dim=1)

            if epoch % 4 == 0 and temp2 == 1:
                # Initialize a list to hold the 2D slices
                val_sample_slices = []

                # Assuming outputs_show, labels, and inputs are 4D tensors with shape [batch, channels, depth, height, width]
                val_found_valid_slice = False
                ind = 0
                while not val_found_valid_slice:
                    ind += 1
                    # Select a random slice index
                    random_slice_idx = random.randint(0, val_outputs_show.shape[2] - 1)
                    label_slice = val_labels[0, 1:3, random_slice_idx, :, :].cpu().detach().numpy()
                    # Check if the selected slice contains any label
                    if np.any(label_slice) or ind > 50:
                        val_found_valid_slice = True

                # Loop through the channels
                for i in range(3):  # Assuming 3 channels for inputs, labels, and outputs
                    # Extract the 2D slice for the input, label, and output
                    if i < 2:
                        if i == 1:
                            input_slice = normalize_image(val_inputs[0, i, random_slice_idx, :, :].cpu().detach().numpy())
                        else:
                            input_slice = val_inputs[0, i, random_slice_idx, :, :].cpu().detach().numpy()
                    label_slice = val_labels[0, i, random_slice_idx, :, :].cpu().detach().numpy()
                    output_slice = val_outputs_show[0, i, random_slice_idx, :, :].cpu().detach().numpy()

                    # Append the slices as images
                    if i < 2:
                        val_sample_slices.append(
                            wandb.Image(input_slice, caption=f"Validation Input Channel {i} - Slice {random_slice_idx}"))
                    val_sample_slices.append(
                        wandb.Image(label_slice, caption=f"Validation Label Channel {i} - Slice {random_slice_idx}"))
                    val_sample_slices.append(
                        wandb.Image(output_slice, caption=f"Validation Output Channel {i} - Slice {random_slice_idx}"))

                    # Log the 2D slices
                wandb.log({f"val_sample_slices_{epoch}": val_sample_slices})

                # Reset temp variable
                temp2 = 0

            if loss.item() > 0:
                step2 += 1
            val_loss += loss.item()
            val_ce_loss += loss_ce.item()

            val_outputs2 = nn.functional.softmax(val_outputs, dim=1)
            val_outputs2 = post_pred(val_outputs2)
            
            # Compute Dice score
            dice_metric(y_pred=val_outputs2, y=val_labels)

    val_loss /= step2
    val_ce_loss /= step2
    val_loss2 /= len(val_loader)

    dice_score = dice_metric.aggregate(reduction="mean").item()
    dice_metric.reset()

    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']

    wandb.log({
        "epoch": epoch + 1,
        "val_dice": val_loss,
        "val_ce": val_ce_loss,
        "val_penalty": val_loss2,
        "dice score": dice_score,
        "learning_rate": current_lr
    })

    if epoch % 50 == 0 and epoch != 0:
        version = epoch // 50
        model_save_path = config['name'] + f'_v{version}' + ".pth"
        torch.save(model.state_dict(), model_save_path)
        wandb.save(model_save_path)
        print(f"Model saved to {model_save_path}")

model_save_path = config['name'] + '_final' + ".pth"
torch.save(model.state_dict(), model_save_path)
wandb.save(model_save_path)
print(f"Model saved to {model_save_path}")
