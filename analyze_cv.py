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

blue_hist = blue
green_hist = green
nir_hist = nir
red_hist = red
rededge_hist = rededge
# blue_hist = cv2.equalizeHist(blue)
# green_hist = cv2.equalizeHist(green)
# nir_hist = cv2.equalizeHist(nir)
# red_hist = cv2.equalizeHist(red)
# rededge_hist = cv2.equalizeHist(rededge)

fixed_image = green
# cv2.imshow('red',red)


# Align all images to the fixed image
aligned_red = preprocess_and_align(fixed_image, red_hist, red, color='red')
aligned_blue = preprocess_and_align(fixed_image, blue_hist, blue, color='blue')
aligned_green = preprocess_and_align(fixed_image, green_hist, green, color='green')
aligned_nir = preprocess_and_align(fixed_image, nir_hist, nir, color='nir')
aligned_rededge = preprocess_and_align(fixed_image, rededge_hist, rededge, color='rededge')

# aligned_red = align_image_orb(fixed_image, red)
# aligned_blue = align_image_orb(fixed_image, blue)
# aligned_green = align_image_orb(fixed_image, green)
# # aligned_nir = align_image_surf(fixed_image, nir)
# aligned_rededge = align_image_orb(fixed_image, rededge)

# aligned_red = align_image_surf(fixed_image, red)
# aligned_blue = align_image_surf(fixed_image, blue)
# aligned_green = align_image_surf(fixed_image, green)
# # aligned_nir = align_image_surf(fixed_image, nir)
# aligned_rededge = align_image_surf(fixed_image, rededge)


# ==============================================================================
# # Stack images into a 5-band arraymatch_image
# spectral_stack = np.stack((aligned_blue, aligned_green, fixed_image, aligned_nir, aligned_rededge), axis=-1)

# # Normalize for visualization
# spectral_stack_norm = (spectral_stack - spectral_stack.min()) / (spectral_stack.max() - spectral_stack.min())
# spectral_stack_uint8 = (spectral_stack_norm * 255).astype(np.uint8)

# # Display an RGB Composite (Blue, Green, Red)
# rgb_image = spectral_stack_uint8[:, :, :3]  # Blue, Green, Red

rgb_stack = np.stack((aligned_blue, green, aligned_red), axis=-1)
rgb_norm = (rgb_stack - rgb_stack.min()) / (rgb_stack.max() - rgb_stack.min())
rgb_array_uint8 = (rgb_norm * 255).astype(np.uint8)

# 加权计算
# r_weight, g_weight, b_weight = 0.299, 0.587, 0.114
# rgb_weighted = np.zeros_like(rgb_stack, dtype=np.float32)
# rgb_weighted[:, :, 0] = rgb_stack[:, :, 0] * b_weight  # Blue
# rgb_weighted[:, :, 1] = rgb_stack[:, :, 1] * g_weight  # Green
# rgb_weighted[:, :, 2] = rgb_stack[:, :, 2] * r_weight  # Red

# 归一化并转换为 uint8
# rgb_weighted_norm = (rgb_weighted - rgb_weighted.min()) / (rgb_weighted.max() - rgb_weighted.min())
# rgb_array_uint8 = (rgb_weighted_norm * 255).astype(np.uint8)

# rgb_array_uint8 = rgb_stack.astype(np.uint8)
# Example comparison for the blue band
# display_comparison(
#     fixed=fixed_image, 
#     moving=green, 
#     aligned=aligned_green, 
#     title_fixed="Fixed Image (NIR)", 
#     title_moving="Moving Image (g)", 
#     title_aligned="Aligned Image ()"
# )
# display_comparison(
#     fixed=fixed_image, 
#     moving=blue, 
#     aligned=aligned_blue, 
#     title_fixed="Fixed Image (NIR)", 
#     title_moving="Moving Image (b)", 
#     title_aligned="Aligned Image ()"
# # )
# display_comparison(
#     fixed=aligned_blue, 
#     moving=aligned_green, 
#     aligned=aligned_red, 
#     title_fixed="aligned_blue", 
#     title_moving="aligned_green", 
#     title_aligned="aligned_red"
# )

display_two_image(aligned_blue,aligned_red)


plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(rgb_array_uint8, cv2.COLOR_BGR2RGB))
plt.title('RGB Image (Aligned with ORB)')
plt.axis('off')
plt.show()