import cv2
import numpy as np

def align_images_optimized(image1_path, image2_path, output_path):
    # 讀取兩張.tif圖形
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print("無法讀取圖像，請檢查路徑！")
        return

    # 初始化 SIFT 檢測器
    sift = cv2.SIFT_create()

    # 檢測和計算特徵點與描述符
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # 使用 BFMatcher 進行描述符匹配 (使用 KNN)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # 應用 Lowe's Ratio Test 過濾匹配
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    print(f"Filtered Matches: {len(good_matches)}")

    # 提取匹配的特徵點位置
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

    # 如果匹配點過少，終止過程
    if len(points1) < 4:
        print("特徵點不足以進行透視變換。")
        return

    # 計算透視變換矩陣
    matrix, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 10.0)
    if matrix is None:
        print("Homography calculation failed.")
        return

    print("Homography Matrix:\n", matrix)
    inliers = mask.sum()
    print(f"Inliers: {inliers} / {len(good_matches)} (Inlier Ratio: {inliers / len(good_matches):.2f})")

    # 可視化匹配 (僅顯示內點匹配)
    inlier_matches = [good_matches[i] for i in range(len(mask)) if mask[i]]
    match_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, inlier_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite("inlier_matches.tif", match_img)

    # 進行影像套合
    height, width = img1.shape
    aligned_img = cv2.warpPerspective(img2, matrix, (width, height))

    # 儲存結果
    cv2.imwrite(output_path, aligned_img)

    # 顯示結果
    cv2.imshow("Image 1", img1)
    cv2.imshow("Aligned Image 2", aligned_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 使用範例
image1_path = 'Green.tif'  # 請替換為第一張圖像的路徑
image2_path = 'Red.tif'  # 請替換為第二張圖像的路徑
output_path = 'aligned_image.tif'  # 結果圖像儲存路徑

align_images_optimized(image1_path, image2_path, output_path)
