import os
import nibabel as nib
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

def read_nifti(file_path):
    img = nib.load(file_path)
    data = img.get_fdata()
    return data

def read_dicom(folder_path):
    slices = [pydicom.dcmread(os.path.join(folder_path, f)) for f in os.listdir(folder_path)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    data = np.stack([s.pixel_array for s in slices], axis=-1)
    return data

def read_nrrd(file_path):
    img = sitk.ReadImage(file_path)
    data = sitk.GetArrayFromImage(img)
    return data

def plot_histogram(data, title="Histogram"):
    plt.figure()
    plt.hist(data.flatten(), bins=100, color='c', alpha=0.75)
    plt.title(title)
    plt.xlabel('Intensity Value')
    plt.ylabel('Frequency')
    plt.show()

def main():
    file_path = input("Enter the path to the CT image file (NIfTI or NRRD) or directory (DICOM): ")
    if file_path.endswith('.nii') or file_path.endswith('.nii.gz'):
        data = read_nifti(file_path)
        plot_histogram(data, title="NIfTI Image Histogram")
    elif os.path.isdir(file_path):
        data = read_dicom(file_path)
        plot_histogram(data, title="DICOM Image Histogram")
    elif file_path.endswith('.nrrd'):
        data = read_nrrd(file_path)
        plot_histogram(data, title="NRRD Image Histogram")
    else:
        print("Unsupported file format or directory structure")

if __name__ == "__main__":
    main()