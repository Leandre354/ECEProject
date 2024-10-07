import os
import glob
import shutil
import numpy as np

# Define paths
base_path = 'imagesTr_pp200/'         # Path where your CT, PET, and seg files are located
test_folder = 'imagesTs_pp200/'       # Path where you want to move test files

# Get lists of files
ct_files = sorted(glob.glob(os.path.join(base_path, '*__CT.nii.gz')))
pet_files = sorted(glob.glob(os.path.join(base_path, '*__PT.nii.gz')))
seg_files = sorted(glob.glob(os.path.join(base_path, '*[!T].nii.gz')))

# Check that all lists have the same length
assert len(ct_files) == len(pet_files) == len(seg_files), "Mismatched file lengths"

# Create a dictionary to map filenames (excluding extensions) to full paths
file_dict = {}
for ct_file, pet_file, seg_file in zip(ct_files, pet_files, seg_files):
    file_key = os.path.basename(ct_file).split('__')[0]  # Using the basename and removing the suffix to match
    file_dict[file_key] = {
        'ct': ct_file,
        'pet': pet_file,
        'seg': seg_file
    }

# Randomly select 50 cases for the test set
np.random.seed(42)  # For reproducibility
selected_keys = np.random.choice(list(file_dict.keys()), size=50, replace=False)

# Move selected files to the test folder
for key in selected_keys:
    files_to_move = file_dict[key]
    for file_type, file_path in files_to_move.items():
        dest_path = os.path.join(test_folder, os.path.basename(file_path))
        shutil.move(file_path, dest_path)


