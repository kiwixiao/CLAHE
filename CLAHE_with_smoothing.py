import nibabel as nib
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.filters import frangi

def load_nii_image(file_path):
    """
    Load a NIfTI image and return the 3D image data.
    """
    nii_img = nib.load(file_path)
    img_data = nii_img.get_fdata()
    voxel_spacing = nii_img.header.get_zooms()
    return img_data, voxel_spacing, nii_img.affine, nii_img.header

def extract_slice(img_data, view='axial'):
    """
    Extract the middle slice from the 3D image data based on the view (axial, coronal, sagittal).
    """
    if view == 'axial':
        middle_slice = img_data[:, :, img_data.shape[2] // 2]
    elif view == 'coronal':
        middle_slice = img_data[:, img_data.shape[1] // 2, :]
    elif view == 'sagittal':
        middle_slice = img_data[img_data.shape[0] // 2, :, :]
    else:
        raise ValueError("View must be 'axial', 'coronal', or 'sagittal'")
    
    # Convert to uint8
    middle_slice = cv2.normalize(middle_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return middle_slice

def windowing(image, window_center, window_width):
    """
    Apply windowing to normalize the intensity values.
    """
    min_intensity = window_center - (window_width // 2)
    max_intensity = window_center + (window_width // 2)
    windowed_image = np.clip(image, min_intensity, max_intensity)
    windowed_image = cv2.normalize(windowed_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return windowed_image

def clahe(image, clip_limit=2.0, tile_grid_size=(16, 16)):
    """
    Apply Contrast-Limited Adaptive Histogram Equalization (CLAHE).
    """
    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    clahe_image = clahe_obj.apply(image)
    return clahe_image

def enhanced_clahe(image, clip_limit=2.0, tile_grid_size=(16, 16)):
    """
    Apply enhanced CLAHE with windowing and Frangi vesselness filtering.
    """
    # Apply windowing to normalize intensity values
    windowed = windowing(image, window_center=40, window_width=400)
    
    # Apply Frangi vesselness filtering
    vesselness = frangi(windowed)
    
    # Normalize the vesselness image to uint8
    vesselness = cv2.normalize(vesselness, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Apply CLAHE to the vesselness-enhanced image
    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_image = clahe_obj.apply(vesselness)
    
    return enhanced_image

def save_nifti(image_data, affine, header, file_path):
    """
    Save the 3D image data as a NIfTI file.
    """
    new_img = nib.Nifti1Image(image_data, affine, header)
    nib.save(new_img, file_path)

# Load the chest CT image (NIfTI format)
image_path = './NL001_non_contrast.nii.gz'  # Replace with your image path
img_data, voxel_spacing, affine, header = load_nii_image(image_path)

# Choose the view: 'axial', 'coronal', or 'sagittal'
view = 'sagittal'  # Replace with 'axial' or 'coronal' as needed

# Extract the middle slice based on the chosen view
middle_slice = extract_slice(img_data, view=view)

# Apply different enhancement techniques to the middle slice
clahe_slice = clahe(middle_slice, clip_limit=3.0, tile_grid_size=(16, 16))
enhanced_clahe_slice = enhanced_clahe(middle_slice, clip_limit=3.0, tile_grid_size=(16, 16))

# Determine the aspect ratio for the selected view
if view == 'axial':
    aspect_ratio = voxel_spacing[0] / voxel_spacing[1]
elif view == 'coronal':
    aspect_ratio = voxel_spacing[0] / voxel_spacing[2]
elif view == 'sagittal':
    aspect_ratio = voxel_spacing[1] / voxel_spacing[2]

# Display the original and enhanced images with correct aspect ratio
fig, axs = plt.subplots(1, 3, figsize=(20, 10))

axs[0].imshow(middle_slice, cmap='gray', aspect=aspect_ratio)
axs[0].set_title(f"Original {view.capitalize()} Slice")
axs[0].axis('off')

axs[1].imshow(clahe_slice, cmap='gray', aspect=aspect_ratio)
axs[1].set_title("CLAHE with Larger Tiles")
axs[1].axis('off')

axs[2].imshow(enhanced_clahe_slice, cmap='gray', aspect=aspect_ratio)
axs[2].set_title("Enhanced CLAHE with Vesselness")
axs[2].axis('off')

plt.tight_layout()
plt.show()

# Apply enhancement techniques to the entire 3D volume
clahe_volume = np.zeros_like(img_data)
enhanced_clahe_volume = np.zeros_like(img_data)

for i in range(img_data.shape[0]):
    clahe_volume[i, :, :] = clahe(cv2.normalize(img_data[i, :, :], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), clip_limit=3.0, tile_grid_size=(16, 16))
    #enhanced_clahe_volume[i, :, :] = enhanced_clahe(cv2.normalize(img_data[i, :, :], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), clip_limit=3.0, tile_grid_size=(16, 16))

# Save the enhanced volumes as NIfTI files
save_nifti(clahe_volume, affine, header, './NL001_clahe_larger_tiles.nii.gz')
#save_nifti(enhanced_clahe_volume, affine, header, '/mnt/data/chest_ct_enhanced_clahe_vesselness_larger_tiles.nii.gz')