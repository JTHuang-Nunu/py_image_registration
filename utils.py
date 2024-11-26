import numpy as np
import matplotlib.pyplot as plt
import cv2

def align_image_orb(fixed, moving, moving_ori=None):
    if fixed is None or moving is None:
        raise ValueError("One of the input images is empty. Check the file paths or formats.")
    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=5000)

    # Detect keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(fixed, None)
    kp2, des2 = orb.detectAndCompute(moving, None)


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
        
    # Extract inlier matches based on the mask
    inlier_matches = [matches[i] for i in range(len(mask)) if mask[i]]

    # Visualize the inlier matches
    match_img =  cv2.drawMatches(fixed, kp1, moving, kp2, inlier_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


    return aligned, match_img

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


def display_comparison(fixed, moving, aligned, title_fixed, title_moving, title_aligned):
    # Convert images to RGB for consistent display in Matplotlib
    fixed_rgb = cv2.cvtColor(fixed, cv2.COLOR_GRAY2RGB)
    moving_rgb = cv2.cvtColor(moving, cv2.COLOR_GRAY2RGB)
    aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_GRAY2RGB)
    
    # Create a figure to display the images
    plt.figure(figsize=(15, 5))
    
    # Display fixed image
    plt.subplot(1, 3, 1)
    plt.imshow(fixed_rgb)
    plt.title(title_fixed)
    plt.axis('off')
    
    # Display moving image
    plt.subplot(1, 3, 2)
    plt.imshow(moving_rgb)
    plt.title(title_moving)
    plt.axis('off')
    
    # Display aligned image
    plt.subplot(1, 3, 3)
    plt.imshow(aligned_rgb)
    plt.title(title_aligned)
    plt.axis('off')
    
    # Show the plots
    plt.tight_layout()
    plt.show()

def display_two_image(image1, image2):
    # 初始化 ORB 特徵檢測器
    orb = cv2.ORB_create(nfeatures=1000)

    # 偵測特徵點和描述子
    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)

    # 使用暴力匹配器 (BFMatcher) 匹配描述子
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # 根據匹配的距離排序，選取最佳匹配
    matches = sorted(matches, key=lambda x: x.distance)

    # 繪製前 50 個匹配結果
    result_image = cv2.drawMatches(
        image1, kp1, image2, kp2, matches[:50], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # 顯示結果
    plt.figure(figsize=(15, 10))
    plt.imshow(result_image)
    plt.title("Feature Matching with Lines")
    plt.axis('off')

    
def preprocess_and_align(fixed, moving, moving_ori=None, color=""):
    # 預處理
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    fixed = clahe.apply(fixed)
    moving = clahe.apply(moving)
    
    # # 提取梯度
    # fixed_gradient = cv2.Laplacian(fixed, cv2.CV_64F, ksize=3)
    # moving_gradient = cv2.Laplacian(moving, cv2.CV_64F, ksize=3)
    
    # 對齊影像
    aligned, match_img, src_pts, dst_pts, mask, matrix = align_image_surf(fixed, moving, moving_ori)
    # vis = visualize_error_vectors(fixed, moving, src_pts, dst_pts, mask)
    visualize_misalignment(fixed, aligned)
    calculate_alignment_error(src_pts, dst_pts, mask)
    # match_path = f'processed_image/match_image_{color}.png'
    # cv2.imwrite(match_path, match_image)
    # aligned_path = f'processed_image/aligned_image_{color}.png'
    # cv2.imwrite(aligned_path, aligned)
    
    return aligned


def visualize_error_vectors(fixed, moving, src_pts, dst_pts, mask):
    """
    Visualize error vectors between matched points.

    :param fixed: Fixed image.
    :param moving: Moving image.
    :param src_pts: Source points (in fixed image).
    :param dst_pts: Destination points (in moving image).
    :param mask: Inlier mask from RANSAC.
    """
    inliers = mask.ravel().astype(bool)
    for src, dst in zip(src_pts[inliers], dst_pts[inliers]):
        pt1 = (int(src[0][0]), int(src[0][1]))
        pt2 = (int(dst[0][0]), int(dst[0][1]))
        cv2.arrowedLine(fixed, pt1, pt2, (0, 255, 0), 1, tipLength=0.3)

    cv2.imshow("Error Vectors", fixed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def visualize_misalignment(fixed, aligned, output_path='misalignment.png'):
    """
    Visualize misalignment as the absolute difference between fixed and aligned images.

    :param fixed: Fixed image.
    :param aligned: Aligned image.
    :param output_path: Path to save the output visualization.
    """
    difference = cv2.absdiff(fixed, aligned)
    normalized_diff = cv2.normalize(difference, None, 0, 255, cv2.NORM_MINMAX)

    cv2.imshow("Misalignment", normalized_diff)
    cv2.imwrite(output_path, normalized_diff)
    print(f"Misalignment visualization saved at {output_path}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def calculate_alignment_error(src_pts, dst_pts, mask):
    """
    Calculate alignment error between matched points.

    :param src_pts: Source points (in fixed image).
    :param dst_pts: Destination points (in moving image).
    :param mask: Inlier mask from RANSAC.
    :return: Mean error and RMSE.
    """
    inliers = mask.ravel().astype(bool)
    errors = np.linalg.norm(src_pts[inliers] - dst_pts[inliers], axis=2)
    mean_error = np.mean(errors)
    rmse = np.sqrt(np.mean(errors**2))
    print(f"Mean Error: {mean_error:.2f}, RMSE: {rmse:.2f}")
    return mean_error, rmse

