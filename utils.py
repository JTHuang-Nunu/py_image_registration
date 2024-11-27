import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

def align_image_orb(fixed, moving, moving_ori=None):
    if fixed is None or moving is None:
        raise ValueError("One of the input images is empty. Check the file paths or formats.")
    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=50000)

    # Detect keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(fixed, None)
    kp2, des2 = orb.detectAndCompute(moving, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good_matches.append(m)
    
    # Extract matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Compute homography matrix
    matrix, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    # Warp the moving image
    aligned = cv2.warpPerspective(moving_ori, matrix, (fixed.shape[1], fixed.shape[0]))
        
    # Extract inlier matches based on the mask
    inlier_matches = [good_matches[i] for i in range(len(mask)) if mask[i]]

    # Visualize the inlier matches
    match_img =  cv2.drawMatches(fixed, kp1, moving, kp2, inlier_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


    return aligned, match_img, src_pts, dst_pts, mask, matrix

def align_image_surf(fixed, moving, moving_ori=None):
    if fixed is None or moving is None:
        raise ValueError("One of the input images is empty. Check the file paths or formats.")
    # Initialize SURF detector
    surf = cv2.xfeatures2d.SURF_create(0)  # Hessian Threshold
    
    # Detect keypoints and descriptors
    kp1, des1 = surf.detectAndCompute(fixed, None)
    kp2, des2 = surf.detectAndCompute(moving, None)

    # Match descriptors using FLANN-based matcher
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good_matches.append(m)
    
    # Extract matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Compute homography matrix
    matrix, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    
    # Warp the moving image
    aligned = cv2.warpPerspective(moving_ori, matrix, (fixed.shape[1], fixed.shape[0]))
    
    inlier_matches = [good_matches[i] for i in range(len(mask)) if mask[i]]

    # Visualize the inlier matches
    match_img =  cv2.drawMatches(fixed, kp1, moving, kp2, inlier_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return aligned, match_img, src_pts, dst_pts, mask, matrix

def display_image(image, title="", save=False, save_path=""):
    # Display
    plt.figure(figsize=(15, 10))
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')

    if save:
        cv2.imwrite(save_path, image)
        print(f"Image saved at {save_path}")
    

def visualize_error_vectors(fixed, src_pts, dst_pts, mask, output_path):
    """
    Visualize error vectors between matched points and log the process.

    :param fixed: Fixed image.
    :param moving: Moving image.
    :param src_pts: Source points (in fixed image).
    :param dst_pts: Destination points (in moving image).
    :param mask: Inlier mask from RANSAC.
    :param log_path: Path to save the log.
    """
    # Change the fixed image to color if it is grayscale
    if len(fixed.shape) == 2: 
        error_image = cv2.cvtColor(fixed, cv2.COLOR_GRAY2BGR)
    else:
        error_image = fixed.copy()


    inliers = mask.ravel().astype(bool)
    log_data = []

    for src, dst in zip(src_pts[inliers], dst_pts[inliers]):
        pt1 = (int(src[0][0]), int(src[0][1]))
        pt2 = (int(dst[0][0]), int(dst[0][1]))
        cv2.arrowedLine(error_image, pt1, pt2, (0, 255, 0), 2, tipLength=1)

    # cv2.imshow("Error Vectors", error_image)
    cv2.imwrite(output_path, error_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def visualize_misalignment(fixed, aligned, output_path):
    """
    Visualize misalignment as the absolute difference between fixed and aligned images.

    :param fixed: Fixed image.
    :param aligned: Aligned image.
    :param output_path: Path to save the output visualization.
    :param log_path: Path to save the log.
    """
    # Compute absolute difference
    difference = cv2.absdiff(fixed, aligned)

    # Normalize difference
    normalized_diff = cv2.normalize(difference, None, 0, 255, cv2.NORM_MINMAX)

    # Enhance contrast
    alpha = 2.0  # Contrast factor
    beta = 0     # Brightness offset
    enhanced_diff = cv2.convertScaleAbs(normalized_diff, alpha=alpha, beta=beta)

    # Apply color map
    colored_diff = cv2.applyColorMap(enhanced_diff, cv2.COLORMAP_JET)

    # Ensure both images are the same size and channel count
    if fixed.shape != colored_diff.shape:
        colored_diff = cv2.resize(colored_diff, (fixed.shape[1], fixed.shape[0]))

    if len(fixed.shape) == 2:  # If fixed image is grayscale, convert to 3 channels
        fixed = cv2.cvtColor(fixed, cv2.COLOR_GRAY2BGR)

    # Overlay with the fixed image
    overlay = cv2.addWeighted(fixed, 0.1, colored_diff, 0.8, 0)

    # Save and display the results
    # cv2.imshow("Misalignment", overlay)
    cv2.imwrite(output_path, overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def calculate_alignment_error(src_pts, dst_pts, mask, log_path):
    """
    Calculate alignment error between matched points and log the results.

    :param src_pts: Source points (in fixed image).
    :param dst_pts: Destination points (in moving image).
    :param mask: Inlier mask from RANSAC.
    :param log_path: Path to save the log.
    :return: Mean error and RMSE.
    """
    inliers = mask.ravel().astype(bool)
    errors = np.linalg.norm(src_pts[inliers] - dst_pts[inliers], axis=2)
    mean_error = np.mean(errors)
    rmse = np.sqrt(np.mean(errors**2))

    # Log the error results
    with open(log_path, "a") as log_file:
        log_file.write(f"Mean Error: {mean_error:.2f}, RMSE: {rmse:.2f}\n\n")

    return mean_error, rmse

def calculate_inlier_ratio(mask, log_path):
    """
    Calculate the inlier ratio based on RANSAC mask.

    :param mask: Inlier mask from RANSAC.
    :param log_path: Path to save the log.
    :return: Inlier ratio.
    """
    inliers = mask.ravel().astype(bool)
    inlier_count = np.sum(inliers)
    total_count = len(mask)
    inlier_ratio = inlier_count / total_count

    # Log the inlier ratio
    with open(log_path, "a") as log_file:
        log_file.write(f"Inlier Ratio: {inlier_ratio:.2%} ({inlier_count}/{total_count})\n\n")

    return inlier_ratio


def preprocess_and_align(fixed, moving, moving_ori=None, color="", mode="orb", fix_spec_band="green"):
    # 預處理
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # fixed = clahe.apply(fixed)
    # moving = clahe.apply(moving)

    
    # 對齊影像
    if mode == "surf":
        aligned, match_img, src_pts, dst_pts, mask, matrix = align_image_surf(fixed, moving, moving_ori)
    elif mode == "orb":
        aligned, match_img, src_pts, dst_pts, mask, matrix = align_image_orb(fixed, moving, moving_ori)

    if aligned is not None:
        image_path = f'process_image_{mode}_{fix_spec_band}'
        align_folder_path = f'{image_path}/aligned'
        log_folder_path = f'{image_path}/report'
        log_path = f'{log_folder_path}/log_{color}.txt'

        if not os.path.exists(image_path):
            os.makedirs(image_path)
        if not os.path.exists(align_folder_path):
            os.makedirs(align_folder_path)
        if not os.path.exists(log_folder_path):
            os.makedirs(log_folder_path)
        if not os.path.exists(log_path):
            with open(log_path, "w") as f:
                f.write("")  # 寫入空內容，創建文件
        else: 
            open(log_path, "w").close()

        # Save the aligned image
        # cv2.imwrite(f"{align_folder_path}/aligned_{color}.png", aligned)

        # # 可視化錯誤向量
        # visualize_error_vectors(fixed, src_pts, dst_pts, mask, f"{image_path}/error_vectors_{color}.png")

        # # 可視化對齊錯誤
        # visualize_misalignment(fixed, aligned, f"{image_path}/misalignment_{color}.png")

        # # 計算並記錄對齊誤差
        # calculate_alignment_error(src_pts, dst_pts, mask, log_path)

        # # 計算並記錄內點比例
        # calculate_inlier_ratio(mask, log_path)

        return aligned
    
    return None


def visualize_rgb_stack(aligned_blue, align_green, aligned_red, output_path, title="RGB Visualization", figsize=(10, 5), ):
    """
    Visualize the RGB stack of aligned color channels.

    :param aligned_blue: Aligned blue channel (2D array).
    :param green: Green channel (2D array).
    :param aligned_red: Aligned red channel (2D array).
    :param title: Title of the visualization plot (default: "RGB Visualization").
    :param figsize: Tuple specifying the figure size for the plot (default: (10, 5)).
    :return: RGB image as uint8 array.
    """
    # Stack the channels into an RGB array
    rgb_stack = np.stack((aligned_blue, align_green, aligned_red), axis=-1)
    #==============================================================================
    # Method 1
    # # Normalize the RGB stack to [0, 1]
    # rgb_norm = (rgb_stack - rgb_stack.min()) / (rgb_stack.max() - rgb_stack.min())
    
    # # Convert normalized array to uint8 format
    # rgb_array_uint8 = (rgb_norm * 255).astype(np.uint8)
    #==============================================================================
    
    #==============================================================================
    # Method 2 
    # Convert to uint8 format
    rgb_array_uint8 = rgb_stack.astype(np.uint8)
    #==============================================================================


    cv2.imwrite(output_path, rgb_array_uint8)
    # Visualize the RGB image
    plt.figure(figsize=figsize)
    plt.imshow(cv2.cvtColor(rgb_array_uint8, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(title)
    plt.show()

    return rgb_array_uint8