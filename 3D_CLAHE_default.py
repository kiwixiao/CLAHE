import SimpleITK as sitk
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os
import cv2

def load_nii_image(file_path):
    """
    Load a NIfTI image and return the 3D image data along with the spacing.
    """
    nii_img = sitk.ReadImage(file_path)
    img_data = sitk.GetArrayFromImage(nii_img)
    spacing = nii_img.GetSpacing()
    return img_data, spacing, nii_img

def save_nifti(image_data, original_image, file_path):
    """
    Save the 3D image data as a NIfTI file.
    """
    new_img = sitk.GetImageFromArray(image_data)
    new_img.CopyInformation(original_image)
    sitk.WriteImage(new_img, file_path)

def analyze_histogram(img_data, bins=512):
    """
    Analyze the histogram of the image data to find high-signal intensity ranges.
    """
    hist, bin_edges = np.histogram(img_data.flatten(), bins=bins)
    return hist, bin_edges

def detect_signal_windows(hist, bin_edges, threshold=0.01):
    """
    Detect high-signal windows based on histogram analysis.
    """
    high_signal_bins = np.where(hist > threshold * np.max(hist))[0]
    signal_windows = []
    for bin_index in high_signal_bins:
        center = (bin_edges[bin_index] + bin_edges[bin_index + 1]) / 2
        width = bin_edges[1] - bin_edges[0]
        signal_windows.append((center, width))
    return signal_windows

def window_image(img_data, min_intensity, max_intensity):
    """
    Apply windowing to the image data to focus on a specific HU range.
    """
    windowed_img_data = np.clip(img_data, min_intensity, max_intensity)
    windowed_img_data = cv2.normalize(windowed_img_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return windowed_img_data

def apply_3d_clahe(img_data, radius=None, alpha=None):
    """
    Apply 3D CLAHE using SimpleITK.
    """
    image = sitk.GetImageFromArray(img_data)
    # Apply CLAHE
    clahe_filter = sitk.AdaptiveHistogramEqualizationImageFilter()
    if radius is not None:
        clahe_filter.SetRadius(radius)
    if alpha is not None:
        clahe_filter.SetAlpha(alpha)
    enhanced_image = clahe_filter.Execute(image)
    enhanced_img_data = sitk.GetArrayFromImage(enhanced_image)
    return enhanced_img_data

def enhance_signal_window(img_data, window, radius=None, alpha=None):
    """
    Enhance the image only within the specified signal window.
    """
    min_intensity, max_intensity = window
    windowed_img_data = window_image(img_data, min_intensity, max_intensity)
    enhanced_window = apply_3d_clahe(windowed_img_data, radius, alpha)
    mask = (img_data >= min_intensity) & (img_data <= max_intensity)
    enhanced_img_data = img_data.copy()
    enhanced_img_data[mask] = enhanced_window[mask]
    return enhanced_img_data

def display_slices(img_data, title="Image"):
    """
    Display example slices from the image volume.
    """
    middle_slice_axial = img_data[img_data.shape[0] // 2, :, :]
    middle_slice_coronal = img_data[:, img_data.shape[1] // 2, :]
    middle_slice_sagittal = img_data[:, :, img_data.shape[2] // 2]

    fig, axs = plt.subplots(1, 3, figsize=(20, 10))

    axs[0].imshow(middle_slice_axial, cmap='gray')
    axs[0].set_title(f"{title} - Axial Slice")
    axs[0].axis('off')

    axs[1].imshow(middle_slice_coronal, cmap='gray')
    axs[1].set_title(f"{title} - Coronal Slice")
    axs[1].axis('off')

    axs[2].imshow(middle_slice_sagittal, cmap='gray')
    axs[2].set_title(f"{title} - Sagittal Slice")
    axs[2].axis('off')

    plt.tight_layout()
    plt.savefig("demo_output.png")
    #plt.show()

# Load the chest CT image (NIfTI format)
image_path = 'NL001_non_contrast.nii.gz'  # Replace with your image path
output_dir = './clahe_results/'  # Directory to save results
os.makedirs(output_dir, exist_ok=True)

img_data, spacing, original_image = load_nii_image(image_path)

# Option to manually specify intensity ranges
manual_signal_windows = [
    # Example of manually specified intensity ranges (min_intensity, max_intensity)
    (10,60),
    (20,70)
]

use_manual_windows = len(manual_signal_windows) > 0

if not use_manual_windows:
    # Analyze histogram to detect high-signal windows if manual ranges are not specified
    hist, bin_edges = analyze_histogram(img_data, bins=512)
    signal_windows = detect_signal_windows(hist, bin_edges, threshold=0.01)
    signal_windows = [(center - width / 2, center + width / 2) for center, width in signal_windows]
else:
    signal_windows = manual_signal_windows

# Arrays of alpha, beta, radii, and windows to explore
alphas = [0.1, 0.3, 0.5, 0.7]
radii = [(4, 4, 4), (8, 8, 8), (20,20,20)]

# Iterate over combinations of windows, alpha, beta, and radii
for window in signal_windows:
    for radius in radii:
        for alpha in alphas:
                print(f"Processing with window={window}, radius={radius}, alpha={alpha}")
                enhanced_img_data = enhance_signal_window(img_data, window, radius=radius, alpha=alpha)

                # Save the enhanced volume
                output_file_path = os.path.join(output_dir, f'NL001_3d_clahe_window_{window[0]}_{window[1]}_radius_{radius[0]}_alpha_{alpha}.nii.gz')
                save_nifti(enhanced_img_data, original_image, output_file_path)

                # Optionally display example slices
                display_slices(enhanced_img_data, title=f"Enhanced Image (window={window}, radius={radius}, alpha={alpha})")
