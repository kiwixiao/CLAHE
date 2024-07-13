import SimpleITK as sitk
import numpy as np
import nibabel as nib
import pydicom
import os
import matplotlib.pyplot as plt

def read_nifti(file_path):
    img = nib.load(file_path)
    data = img.get_fdata()
    return data, img.affine

def read_dicom(folder_path):
    slices = [pydicom.dcmread(os.path.join(folder_path, f)) for f in os.listdir(folder_path) if f.endswith('.dcm')]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    data = np.stack([s.pixel_array for s in slices], axis=-1)
    return data

def read_nrrd(file_path):
    img = sitk.ReadImage(file_path)
    data = sitk.GetArrayFromImage(img)
    return data

def reverse_clahe_3d(image, window_min, window_max, radius=(8, 8, 8), alpha=0.3):
    """
    Apply a reverse CLAHE method using SimpleITK to reduce the contrast of a 3D image
    within a specified intensity window.
    """
    sitk_image = sitk.GetImageFromArray(image)
    
    # Apply CLAHE
    clahe_filter = sitk.AdaptiveHistogramEqualizationImageFilter()
    clahe_filter.SetRadius(radius)
    clahe_filter.SetAlpha(alpha)
    equalized_image = clahe_filter.Execute(sitk_image)
    
    equalized_array = sitk.GetArrayFromImage(equalized_image)
    
    # Apply windowing and contrast reduction within the specified intensity range
    mask = (image >= window_min) & (image <= window_max)
    reduced_contrast_array = np.copy(image)
    reduced_contrast_array[mask] = 255 - equalized_array[mask]  # Invert intensities within window
    
    return reduced_contrast_array

def apply_reverse_clahe_3d(data, window_min, window_max, radius=(8, 8, 8), alpha=0.3):
    """
    Apply the reverse CLAHE to the entire 3D volume within the specified intensity window.
    """
    return reverse_clahe_3d(data, window_min, window_max, radius, alpha)

def plot_histogram(data, title="Histogram"):
    plt.figure()
    plt.hist(data.flatten(), bins=100, color='c', alpha=0.75)
    plt.title(title)
    plt.xlabel('Intensity Value')
    plt.ylabel('Frequency')
    plt.show()

def main():
    file_path = input("Enter the path to the CT image file (NIfTI or NRRD) or directory (DICOM): ")
    window_min = float(input("Enter the minimum intensity of the window: "))
    window_max = float(input("Enter the maximum intensity of the window: "))
    radius = tuple(map(int, input("Enter the tile grid size for CLAHE (e.g., 8,8,8): ").split(',')))
    alpha = float(input("Enter the alpha value for CLAHE (e.g., 0.3): "))

    if file_path.endswith('.nii') or file_path.endswith('.nii.gz'):
        data, affine = read_nifti(file_path)
        plot_histogram(data, title="Original NIfTI Image Histogram")
        reduced_contrast_data = apply_reverse_clahe_3d(data, window_min, window_max, radius, alpha)
        plot_histogram(reduced_contrast_data, title="Reduced Contrast NIfTI Image Histogram")
        reduced_contrast_image = nib.Nifti1Image(reduced_contrast_data, affine)
        nib.save(reduced_contrast_image, 'reduced_contrast_image.nii.gz')
    elif file_path.endswith('.nrrd'):
        data = read_nrrd(file_path)
        plot_histogram(data, title="Original NRRD Image Histogram")
        reduced_contrast_data = apply_reverse_clahe_3d(data, window_min, window_max, radius, alpha)
        plot_histogram(reduced_contrast_data, title="Reduced Contrast NRRD Image Histogram")
        reduced_contrast_image = sitk.GetImageFromArray(reduced_contrast_data)
        sitk.WriteImage(reduced_contrast_image, 'reduced_contrast_image.nrrd')
    else:
        print("Unsupported file format or directory structure")

if __name__ == "__main__":
    main()
    