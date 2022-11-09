import os
import cv2
import scipy
import numpy as np


def create_image_data(image_dir, video_file):
    try:
        os.mkdir(image_dir)
    except:
        pass

    print("ffmpeg -i {0} {1}/%05d.png".format(video_file, image_dir))
    os.system("ffmpeg -i {0} {1}%05d.png".format(video_file, image_dir))

def generate_laserpoint_images_from_mat(matfile, save_path, image_height, image_width):
    f = scipy.io.loadmat(matfile)
    points_per_frame = list(np.moveaxis(f['PP'].reshape(-1, 2, 200).transpose(), -1, -2))
    freezed_points = f['freeze'].reshape(-1, 200).transpose()

    for count, points in enumerate(points_per_frame):
        image = np.zeros((image_height, image_width), dtype=np.uint8)
        cleaned_points = points[~np.isnan(points).any(axis=1)]
        
        for i in range(cleaned_points.shape[0]):
            if freezed_points[count, i] == 1:
                continue

            cv2.circle(image, cleaned_points[i].astype(np.int), radius=6, thickness=-1, color=255)
    
        write_laserdot_mask(save_path, count, image)

def write_laserdot_mask(path, index, mask_image):
    try:
        os.mkdir(path)
    except:
        pass

    cv2.imwrite("{}{:05d}.png".format(path, index), mask_image)


if __name__ == "__main__":
    #create_image_data("data/png/", "data/Human_P181133_top_Broc5_4001-4200.avi")
    generate_laserpoint_images_from_mat("data/VideoClick_P181133_E010_A010_F3.mat", "data/laserpoints/", 1200, 800)