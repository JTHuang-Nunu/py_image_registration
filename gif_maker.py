import os
from PIL import Image
import imageio

def list_images_in_folder(folder_path, image_extensions=None):
    """
    列出資料夾內的所有圖片文件。
    
    :param folder_path: 要尋覽的資料夾路徑
    :param image_extensions: 支援的圖片副檔名（列表），如 ['jpg', 'png', 'gif']
    :return: 圖片檔案的完整路徑清單
    """
    if image_extensions is None:
        image_extensions = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff']
    
    # 確保副檔名大小寫不敏感
    image_extensions = set(ext.lower() for ext in image_extensions)
    
    # 儲存圖片文件的清單
    image_files = []

    # 遍歷資料夾內的文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(tuple(image_extensions)):
                image_files.append(os.path.join(root, file))
    
    return image_files


folder_path = r"F:\Code\image_match\process_image_surf_green\aligned"
output_gif_path = f"{folder_path}/aligned_output.gif"
images = list_images_in_folder(folder_path)

# GIF maker
images_gif = []
for image_path in images:
    img = Image.open(image_path)
    
    images_gif.append(img)

images_gif[0].save(
    output_gif_path,
    save_all=True,
    append_images=images_gif[1:],
    duration=500,  # delay time per frame (ms)
    loop=0         # infinite loop
)
