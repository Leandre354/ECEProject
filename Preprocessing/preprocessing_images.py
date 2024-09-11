import SimpleITK as sitk
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def normalize_to_255(image):
    # Convert SimpleITK image to NumPy array
    image_array = sitk.GetArrayFromImage(image).astype(np.float32)

    # Normalize to [0, 255]
    min_val = np.min(image_array)
    max_val = np.max(image_array)
    normalized_array = (image_array - min_val) / (max_val - min_val) * 255
    normalized_array = np.clip(normalized_array, 5, 255).astype(np.uint8)

    # Convert back to SimpleITK image
    normalized_image = sitk.GetImageFromArray(normalized_array)
    normalized_image.CopyInformation(image)
    return normalized_image


def compute_center(image_array):
    # Convert to a numpy array for contour detection
    img_np = image_array.astype(np.uint8)
    for i in reversed(range(img_np.shape[0])):
        slice_2d = img_np[i]
        _, bin = cv2.threshold(slice_2d, 5, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        center_y = 0
        center_x = 0
        center_z = 0
        if contours:
            # Taking the biggest area from detected contours
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > max_area:
                    max_area = area
            # Threshold area
            if (max_area > 2000):
                # Computing the center of the object (brain)
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    center_y = int(M["m10"] / M["m00"])
                    center_x = int(M["m01"] / M["m00"])
                    center_z = i

                    # Making sure that the center is not in the border
                    if (center_x > 140) and (center_x < (512 - 140)) and (center_y > 140) and (center_y < (512 - 140)):
                        break
    return center_x, center_y, center_z

def resample_pet(source_image, reference_image):
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(reference_image.GetSize())
    resampler.SetOutputSpacing(reference_image.GetSpacing())
    resampler.SetOutputDirection(reference_image.GetDirection())
    resampler.SetOutputOrigin(reference_image.GetOrigin())
    resampler.SetTransform(sitk.Transform())  # Identity transform
    resampler.SetInterpolator(sitk.sitkLinear)
    return resampler.Execute(source_image)

def resize_label(ct_image, lbl_image):

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(ct_image.GetSize())
    resampler.SetOutputSpacing(ct_image.GetSpacing())
    resampler.SetOutputDirection(ct_image.GetDirection())
    resampler.SetOutputOrigin(ct_image.GetOrigin())
    resampler.SetTransform(sitk.Transform())  # Identity transform
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    return resampler.Execute(lbl_image)

def resample_spacing(image, label, target_spacing=[1.0, 1.0, 1.0]):
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()

    new_size = [int(round(original_size[i] * original_spacing[i] / target_spacing[i])) for i in range(3)]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(target_spacing)
    resample.SetSize(new_size)
    if label == True:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkLinear)

    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetOutputPixelType(image.GetPixelID())
    image = resample.Execute(image)
    new_image = sitk.Cast(image, sitk.sitkFloat32)
    return new_image

def z_score_clip_image(image, clip_range=(-3, 3)):
    # Convert SimpleITK image to numpy array
    image_array = sitk.GetArrayFromImage(image)

    # Compute mean and standard deviation
    mean = np.mean(image_array)
    std = np.std(image_array)

    # Normalize to Z-scores
    z_score_array = (image_array - mean) / std

    # Clip Z-scores to the specified range
    min_clip, max_clip = clip_range
    clipped_array = np.clip(z_score_array, min_clip, max_clip)

    # Rescale back to original range
    rescaled_array = clipped_array * std + mean

    # Convert numpy array back to SimpleITK image
    clipped_image = sitk.GetImageFromArray(rescaled_array)
    clipped_image.CopyInformation(image)  # Preserve metadata from the original image

    return clipped_image

def cropping(resampled_ct_img, resampled_pet_img, resampled_lbl_img, plotting=False):
    #Normalize to 0-255 range for the plotting
    ct_norm = sitk.GetArrayFromImage(normalize_to_255(resampled_ct_img))
    pt_norm = sitk.GetArrayFromImage(normalize_to_255(resampled_pet_img))

    # Convert images to numpy arrays
    ct_array = sitk.GetArrayFromImage(resampled_ct_img)
    pt_array = sitk.GetArrayFromImage(resampled_pet_img)
    lbl_array = sitk.GetArrayFromImage(resampled_lbl_img)
    # Compute center of the brain
    center_x, center_y, center_z = compute_center(pt_norm)

    #Margin for the cropping region (center +- margin)
    margin = 100

    if plotting == True:
        # Plotting the cropped zone on original image
        slice_sel = center_z
        color = cv2.cvtColor(pt_norm[center_z], cv2.COLOR_GRAY2BGR)
        color[center_x - 1:center_x + 1, center_y - 1:center_y + 1] = (255, 0, 0)
        color = cv2.rectangle(color, (center_y - margin, center_x - margin), (center_y + margin, center_x + margin),
                              (255, 0, 0), 1)

        # Display image with matplotlib
        img_ct = plt.imshow(color)
        slice_slider_ax = plt.axes([0.15, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        slice_slider = Slider(slice_slider_ax, 'Slice', 0, pt_array.shape[0] - 1, valinit=slice_sel)

        def update(val):
            slice_sel = int(slice_slider.val)
            color = cv2.cvtColor(pt_norm[slice_sel], cv2.COLOR_GRAY2BGR)
            color[center_x - 1:center_x + 1, center_y - 1:center_y + 1] = (255, 0, 0)
            color = cv2.rectangle(color, (center_y - margin, center_x - margin), (center_y + margin, center_x + margin),
                                  (255, 0, 0), 1)
            img_ct.set_data(color)

        slice_slider.on_changed(update)
        plt.show()

    crop = 0

    for slice in range(ct_array.shape[0] - 1, 0, -1):  # Iterate from the last slice to the first slice
        mean = np.mean(ct_array[slice, :, :])  # Extract the slice at index z
        if mean == 0:
            crop += 1
        else:
            break

    high_margin = ct_array.shape[0] - center_z - crop

    if ct_array.shape[0] > 310:
        low_margin = 310 - high_margin
    else:
        low_margin = ct_array.shape[0] - high_margin - crop

    if ct_array.shape[2] > 256:
        # Cropping images
        ct_crop = ct_array[center_z - low_margin:center_z + high_margin, center_x - margin:center_x + margin,
                  center_y - margin:center_y + margin]
        pt_crop = pt_array[center_z - low_margin:center_z + high_margin, center_x - margin:center_x + margin,
                  center_y - margin:center_y + margin]
        lbl_crop = lbl_array[center_z - low_margin:center_z + high_margin,
                   center_x - margin:center_x + margin, center_y - margin:center_y + margin]
    else:
        # Cropping images
        ct_crop = ct_array[center_z - low_margin:center_z + high_margin, :, :]
        pt_crop = pt_array[center_z - low_margin:center_z + high_margin, :, :]
        lbl_crop = lbl_array[center_z - low_margin:center_z + high_margin, :, :]

    # Update SimpleITK image with cropped data
    cropped_ct_img = sitk.GetImageFromArray(ct_crop)
    cropped_pet_img = sitk.GetImageFromArray(pt_crop)
    cropped_lbl_img = sitk.GetImageFromArray(lbl_crop)

    before = np.count_nonzero(lbl_array)
    after = np.count_nonzero(lbl_crop)

    if (before != after):
        # Show if a loss of ground truth occurs when cropping
        print("error :", pt_files[file], " before :", before, " after :", after, "lost : ", (1 - after / before) * 100,"%")
    else:
        print(pt_files[file] + " crop succeed")

    return cropped_ct_img, cropped_pet_img, cropped_lbl_img

name = 'CHUM*'
show = False

#Paths to images and labels
ct_name = 'hecktor2022/imagesTr/'+name+'__CT.nii.gz'
pt_name = 'hecktor2022/imagesTr/'+name+'__PT.nii.gz'
lbl_name = 'hecktor2022/labelsTr/'+name+'.nii.gz'

#Create a sorted list of the files from the paths
ct_files = sorted(glob.glob(ct_name))
pt_files = sorted(glob.glob(pt_name))
lbl_files = sorted(glob.glob(lbl_name))

for file in range(len(ct_files)):
    #Load of images and labels
    ct_img = sitk.ReadImage(ct_files[file])
    pet_img = sitk.ReadImage(pt_files[file])
    lbl_img = sitk.ReadImage(lbl_files[file])

    print("Initial")
    #Print original size
    print(ct_img.GetSize())
    print(pet_img.GetSize())
    print(lbl_img.GetSize())

    pet_img = resample_pet(pet_img, ct_img)

    print("PET resample")
    # Print new size
    print(ct_img.GetSize())
    print(pet_img.GetSize())
    print(lbl_img.GetSize())

    #Correction of little change in dimension of the labels
    if lbl_img.GetSize() != ct_img.GetSize():
        print(lbl_img.GetSpacing())
        print(lbl_img.GetDirection())
        print(lbl_img.GetOrigin())
        print(lbl_img.GetSize())
        lbl_img = resize_label(ct_img, lbl_img)
        print(lbl_img.GetSpacing())
        print(lbl_img.GetDirection())
        print(lbl_img.GetOrigin())
        print(lbl_img.GetSize())

    #Resample images to an isovoxel of [1,1,1]
    resampled_ct_img = resample_spacing(ct_img, False)
    resampled_pet_img = resample_spacing(pet_img, False)
    resampled_lbl_img = resample_spacing(lbl_img, True)

    print("Spacing")
    #Print the new sizes
    print(resampled_ct_img.GetSize())
    print(resampled_pet_img.GetSize())
    print(resampled_lbl_img.GetSize())

    #Cropping of the ROI (Head and neck)
    resampled_ct_img, resampled_pet_img, resampled_lbl_img = cropping(resampled_ct_img, resampled_pet_img, resampled_lbl_img, show)

    print("Cropping")
    #Print new cropped size
    print(resampled_ct_img.GetSize())
    print(resampled_pet_img.GetSize())
    print(resampled_lbl_img.GetSize())

    #z-score clipping to avoid outliers
    resampled_ct_img = z_score_clip_image(resampled_ct_img, clip_range=(-3, 3))
    resampled_pet_img = z_score_clip_image(resampled_pet_img, clip_range=(-3, 3))

    print("Normalize")
    print(resampled_ct_img.GetSize())
    print(resampled_pet_img.GetSize())
    print(resampled_lbl_img.GetSize())


    # Save resampled PET image
    sitk.WriteImage(resampled_ct_img, 'hecktor2022/imagesTr_pp/' + ct_files[file].split('\\')[1])
    sitk.WriteImage(resampled_pet_img,'hecktor2022/imagesTr_pp/' + pt_files[file].split('\\')[1])
    sitk.WriteImage(resampled_lbl_img,'hecktor2022/labelsTr_pp/' + lbl_files[file].split('\\')[1])