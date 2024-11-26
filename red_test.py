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

fixed_image = green

aligned_red = preprocess_and_align(fixed_image, red, red, color='red')

