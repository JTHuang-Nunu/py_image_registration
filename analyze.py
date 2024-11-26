import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# Load the uploaded TIF files
blue = sitk.ReadImage('spectral_images/Blue.tif', sitk.sitkFloat32)
green = sitk.ReadImage('spectral_images/Green.tif', sitk.sitkFloat32)
nir = sitk.ReadImage('spectral_images/NIR.tif', sitk.sitkFloat32)
red = sitk.ReadImage('spectral_images/Red.tif', sitk.sitkFloat32)
rededge = sitk.ReadImage('spectral_images/RedEdge.tif', sitk.sitkFloat32)

# Use the Red image as the fixed reference
fixed_image = red

# Function to register images
def register_images(fixed, moving):
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0, numberOfIterations=100,
        convergenceMinimumValue=1e-6, convergenceWindowSize=10
    )
    initial_transform = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler2DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    registration_method.SetInterpolator(sitk.sitkLinear)
    final_transform = registration_method.Execute(fixed, moving)
    moving_resampled = sitk.Resample(
        moving, fixed, final_transform, sitk.sitkLinear, 0.0, moving.GetPixelID()
    )
    return moving_resampled

# Register the images to the fixed image
registered_blue = register_images(fixed_image, blue)
registered_green = register_images(fixed_image, green)
registered_nir = register_images(fixed_image, nir)
registered_rededge = register_images(fixed_image, rededge)

# Convert to NumPy arrays for stacking
red_array = sitk.GetArrayFromImage(red)
blue_array = sitk.GetArrayFromImage(registered_blue)
green_array = sitk.GetArrayFromImage(registered_green)
nir_array = sitk.GetArrayFromImage(registered_nir)
rededge_array = sitk.GetArrayFromImage(registered_rededge)

# Stack the images to create a 5-band array
spectral_stack = np.stack((blue_array, green_array, red_array, nir_array, rededge_array), axis=-1)
rgb_stack = np.stack((red_array, green_array, blue_array), axis=-1)
# Normalize for visualization
spectral_stack_norm = (spectral_stack - spectral_stack.min()) / (spectral_stack.max() - spectral_stack.min())
spectral_stack_uint8 = (spectral_stack_norm * 255).astype(np.uint8)

rgb_norm = (rgb_stack - rgb_stack.min()) / (rgb_stack.max() - rgb_stack.min())
rgb_array_uint8 = (rgb_norm * 255).astype(np.uint8)

plt.imshow(rgb_array_uint8)
plt.title('RGB 彩色影像')
plt.axis('off')
plt.show()

rgb_image = Image.fromarray(rgb_array_uint8)
rgb_image.save('rgb_image.png')
