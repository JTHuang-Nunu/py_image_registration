'''
有點問題
需要嘗試三次內插反算法可能會比較好
'''
import cv2
import os
import numpy as np
from PIL import Image

# sensor_size = (36, 24)  # 36mm x 24mm
image_size = (5568, 3712)
pixel_size = 0.005 # mm/pixel
# 1. 已知的內參數矩陣 (Camera Matrix) 和 畸變係數 (Distortion Coefficients)
f = 24.0838
fx = f / pixel_size
fy = f / pixel_size

cx_ori = -0.0593
cx = cx_ori / pixel_size + image_size[0] / 2
cy_ori = -0.1157
cy = cy_ori / pixel_size + image_size[1] / 2

camera_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]])

k1 = 7.0766e-06
k2 = -8.8571e-08
k3 = 3.0652e-10
p1 = 7.1548e-05
p2 = -9.6609e-07
dist_coeffs = np.array([k1, k2, p1, p2, k3])

# 2. 讀取影像的資料夾路徑
input_folder = 'images3/'  # 放置待校正影像的資料夾
output_folder = 'output3-2/'  # 放置校正後影像的資料夾

# 如果 output 資料夾不存在，則創建該資料夾
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
# def cubic_interpolate(p0, p1, p2, p3, x):
#     '''
#     cubic spline interpolation
#     '''
#     return (
#         -0.5 * p0 * (x - 1) * (x - 2) * (x - 3) +
#         1.5 * p1 * x * (x - 2) * (x - 3) -
#         1.5 * p2 * x * (x - 1) * (x - 3) +
#         0.5 * p3 * x * (x - 1) * (x - 2)
#     )

def cubic_interpolate(p0, p1, p2, p3, x):
    a = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3
    b = p0 - 2.5 * p1 + 2 * p2 - 0.5 * p3
    c = -0.5 * p0 + 0.5 * p2
    d = p1
    return a * x**3 + b * x**2 + c * x + d
def undistort_image(image, camera_matrix, dist_coeffs):
    h, w = image.shape[:2]
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    k1, k2, p1, p2, k3 = dist_coeffs

    undistorted_img = np.zeros_like(image)

    for y in range(h):
        for x in range(w):
            xn = (x - cx) / fx
            yn = (y - cy) / fy
            r2 = xn * xn + yn * yn

            # 計算徑向和切向畸變
            radial_distortion = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
            x_distorted = xn * radial_distortion + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn**2)
            y_distorted = yn * radial_distortion + p1 * (r2 + 2 * yn**2) + 2 * p2 * xn * yn

            # 映射回原始影像座標
            x_new = fx * x_distorted + cx
            y_new = fy * y_distorted + cy

            # 四捨五入並檢查是否在影像範圍內
            x0 = int(np.clip(np.floor(x_new), 1, w - 3))
            y0 = int(np.clip(np.floor(y_new), 1, h - 3))

            # 取得 4x4 區域，進行三次內插
            patch = image[y0:y0 + 4, x0:x0 + 4]

            if patch.shape == (4, 4):
                col_values = [cubic_interpolate(patch[i, 0], patch[i, 1], patch[i, 2], patch[i, 3], x_new - (x0 + 1)) for i in range(4)]
                pixel_value = cubic_interpolate(col_values[0], col_values[1], col_values[2], col_values[3], y_new - (y0 + 1))
                undistorted_img[y, x] = np.clip(pixel_value, 0, 255)

    return undistorted_img

# 3. 依序讀取資料夾內所有的影像檔案
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.JPG'):
        # 讀取影像
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        image_pil = Image.open(img_path)
        exif_data = image_pil.info.get('exif')
        
        if img is not None:
            h, w = img.shape[:2]

            # 4. 使用內參數和畸變參數進行影像校正
            # new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
            # undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)
            # undistorted_rgb_img = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB) # OpenCV uses BGR, PIL uses RGB
            
            # map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, None, (w, h), cv2.CV_32FC1)
            # undistorted_img = cv2.remap(img, map1, map2, cv2.INTER_AREA)
            # undistorted_rgb_img = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB)
            undistorted_image = undistort_image(img, camera_matrix, dist_coeffs)

            # 儲存校正後的影像
            output_path = os.path.join(output_folder, "undistorted_" + filename)
            Image.fromarray(undistorted_image).save(output_path)
            print(f"Saved undistorted image: {output_path}")

            # 5. 剪裁結果圖片（如果有需要）
            # x, y, w, h = roi
            # if w > 0 and h > 0:
            #     # 剪裁結果圖片
            #     undistorted_rgb_img = undistorted_rgb_img[y:y+h, x:x+w]
            # else:
            #     print(f'Invalid ROI for image: {filename}, skipping crop.')

            # 6. 將校正後的影像存儲到 output 資料夾
        #     output_path = os.path.join(output_folder, filename)
        #     if undistorted_rgb_img is not None and undistorted_rgb_img.size > 0:
        #         result_img = Image.fromarray(undistorted_rgb_img)
        #         result_img.save(output_path, exif=exif_data, quality=100)
        #         # cv2.imwrite(output_path, undistorted_img)
        #         print(f'Processed and saved: {output_path}')
        #     else:
        #         print(f'Error: undistorted image is empty for {filename}')
        # else:
        #     print(f'Failed to load image: {img_path}')


