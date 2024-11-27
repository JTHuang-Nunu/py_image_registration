import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from utils import *


# Load the uploaded TIF files
blue = cv2.imread('spectral_images/Blue.tif', cv2.IMREAD_GRAYSCALE)
green = cv2.imread('spectral_images/Green.tif', cv2.IMREAD_GRAYSCALE)
nir = cv2.imread('spectral_images/NIR.tif', cv2.IMREAD_GRAYSCALE)
red = cv2.imread('spectral_images/Red.tif', cv2.IMREAD_GRAYSCALE)
rededge = cv2.imread('spectral_images/RedEdge.tif', cv2.IMREAD_GRAYSCALE)

# Paramter Setting
spectral_band = {
    'blue': blue,
    'green': green,
    'nir': nir,
    'red': red,
    'rededge': rededge
}
fixed_spec_band = 'green'
mode='surf'
fixed_image = spectral_band[fixed_spec_band]

# Align all images to the fixed image
aligned_red = preprocess_and_align(fixed_image, red, red, color='red', mode=mode, fix_spec_band=fixed_spec_band)
aligned_blue = preprocess_and_align(fixed_image, blue, blue, color='blue', mode=mode, fix_spec_band=fixed_spec_band)
aligned_green = preprocess_and_align(fixed_image, green, green, color='green', mode=mode, fix_spec_band=fixed_spec_band)
aligned_nir = preprocess_and_align(fixed_image, nir, nir, color='nir', mode=mode, fix_spec_band=fixed_spec_band)
aligned_rededge = preprocess_and_align(fixed_image, rededge, rededge, color='rededge', mode=mode, fix_spec_band=fixed_spec_band)

# Stack images into a 3-band(bgr) array
# visualize_rgb_stack(aligned_blue, aligned_green, aligned_red, output_path='aligned_rgb.png')

# Stack images into a 5-band array
aligned_images = []
aligned_images.append(aligned_blue)
aligned_images.append(aligned_green)
aligned_images.append(fixed_image)
aligned_images.append(aligned_nir)
aligned_images.append(aligned_rededge)
aligned_pil = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in aligned_images]


# Save as gif
aligned_pil[0].save(
    "aligned_output.gif",
    save_all=True,
    append_images=aligned_pil[1:],
    duration=500,  # 每幀的延遲時間(ms)
    loop=0         # 無限循環
)