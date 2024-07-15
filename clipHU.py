import nibabel as nib
import nrrd
import numpy as np
import sys
import os

def read_and_clip_ct_image(file_path):
    # Determine the file format from the extension
    if file_path.endswith('.nii') or file_path.endswith('.nii.gz'):
        img = nib.load(file_path)
        image_data = img.get_fdata()
        affine = img.affine
    elif file_path.endswith('.nrrd'):
        image_data, header = nrrd.read(file_path)
        affine = np.eye(4)  # Default affine for NRRD, you might want to adjust it based on your data
    else:
        raise ValueError("Unsupported file format. Please provide a NIfTI (.nii, .nii.gz) or NRRD (.nrrd) file.")

    # Clip the Hounsfield Units to be within [-1000, 1000]
    clipped_image_data = np.clip(image_data, -1000, 1000)

    # Save the clipped image as NIfTI
    clipped_img = nib.Nifti1Image(clipped_image_data, affine)
    if file_path.endswith('.nii') or file_path.endswith('nrrd'):
        output_file_path = os.path.splitext(file_path)[0] + '_clipped.nii.gz'
    elif:
        output_file_path = file_path[:-7] + '_nii.gz'
    nib.save(clipped_img, output_file_path)

    return output_file_path

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    output_file_path = read_and_clip_ct_image(file_path)
    print(f"Clipped image saved as: {output_file_path}")
