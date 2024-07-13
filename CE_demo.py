import nibabel as nib
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure

def load_nii_image(file_path):
    """
    Load a NIfTI image and return the 3D image data.
    """
    nii_img = nib.load(file_path)
    img_data = nii_img.get_fdata()
    return img_data, nii_img.affine, nii_img.header

def extract_sagittal_slice(img_data):
    """
    Extract the middle sagittal slice from the 3D image data.
    """
    middle_slice = img_data[:, :, img_data.shape[2] // 2]
    # Convert to uint8
    middle_slice = cv2.normalize(middle_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return middle_slice

def histogram_equalization(image):
    """
    Apply Histogram Equalization to enhance contrast.
    """
    return cv2.equalizeHist(image)

def clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply Contrast-Limited Adaptive Histogram Equalization (CLAHE).
    """
    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe_obj.apply(image)

def gamma_correction(image, gamma=1.0):
    """
    Apply Gamma Correction to enhance contrast.
    """
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def adaptive_histogram_equalization(image, clip_limit=0.01):
    """
    Apply Adaptive Histogram Equalization (AHE) using skimage.
    """
    return exposure.equalize_adapthist(image, clip_limit=clip_limit)

def creah(image, clip_limit=0.01, tile_grid_size=(8, 8)):
    """
    Apply Contextual and Region-based Adaptive Histogram Equalization (CREAH) with a damped grid.
    """
    # Create CLAHE object with the given clip limit and tile grid size
    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    # Apply CLAHE to the image
    enhanced_image = clahe_obj.apply(image)
    
    # Use a damped grid to smooth transitions between contextual regions
    smoothed_image = exposure.equalize_adapthist(image, clip_limit=clip_limit)

    return smoothed_image

def save_nifti(image_data, affine, header, file_path):
    """
    Save the 3D image data as a NIfTI file.
    """
    new_img = nib.Nifti1Image(image_data, affine, header)
    nib.save(new_img, file_path)

# Load the chest CT image (NIfTI format)
image_path = 'NL001_non_contrast.nii.gz'  # Replace with your image path
img_data, affine, header = load_nii_image(image_path)

# Extract the middle sagittal slice
sagittal_slice = extract_sagittal_slice(img_data)

# Apply different enhancement techniques to the middle sagittal slice
he_slice = histogram_equalization(sagittal_slice)
clahe_slice = clahe(sagittal_slice)
gc_slice = gamma_correction(sagittal_slice, gamma=1.5)
ahe_slice = adaptive_histogram_equalization(sagittal_slice)
creah_slice = creah(sagittal_slice)

# Display the original and enhanced images
plt.figure(figsize=(20, 10))

plt.subplot(2, 3, 1)
plt.title("Original Sagittal Slice")
plt.imshow(sagittal_slice, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title("Histogram Equalization")
plt.imshow(he_slice, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title("CLAHE")
plt.imshow(clahe_slice, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title("Gamma Correction")
plt.imshow(gc_slice, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title("Adaptive Histogram Equalization (AHE)")
plt.imshow(ahe_slice, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.title("CREAH")
plt.imshow(creah_slice, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

# Apply enhancement techniques to the entire 3D volume
he_volume = np.zeros_like(img_data)
clahe_volume = np.zeros_like(img_data)
gc_volume = np.zeros_like(img_data)
ahe_volume = np.zeros_like(img_data)
creah_volume = np.zeros_like(img_data)

for i in range(img_data.shape[0]):
    he_volume[i, :, :] = histogram_equalization(cv2.normalize(img_data[i, :, :], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    clahe_volume[i, :, :] = clahe(cv2.normalize(img_data[i, :, :], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    gc_volume[i, :, :] = gamma_correction(cv2.normalize(img_data[i, :, :], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), gamma=1.5)
    ahe_volume[i, :, :] = adaptive_histogram_equalization(cv2.normalize(img_data[i, :, :], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    creah_volume[i, :, :] = creah(cv2.normalize(img_data[i, :, :], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))

# Save the enhanced volumes as NIfTI files
save_nifti(he_volume, affine, header, './NL001_he.nii.gz')
save_nifti(clahe_volume, affine, header, './NL001_clahe.nii.gz')
save_nifti(gc_volume, affine, header, './NL001_gc.nii.gz')
save_nifti(ahe_volume, affine, header, './NL001_ahe.nii.gz')
save_nifti(creah_volume, affine, header, './NL001_creah.nii.gz')