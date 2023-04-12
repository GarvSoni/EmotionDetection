import os
import  config
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}

def count_images_in_dir(dir_path):
    count = 0
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if os.path.splitext(file)[-1].lower() in IMAGE_EXTENSIONS:
                count += 1
    return count

dir_path = config.train_dir
total_image_count = count_images_in_dir(dir_path)
print(f"Total number of images in {dir_path}: {total_image_count}")
