import os
import sys
from pathlib import Path
import cv2

class VideoGenerator:
    def __init__(self, base_path):
        self.base_path = base_path


    def create_ffmpeg_call(self, video_path, save_path, start_number, crop_x, crop_y):
        file_path = os.path.join(self.base_path, video_path)
        save = os.path.join(self.base_path, save_path)
        ffmpeg_string = "ffmpeg -framerate 15 -start_number {0} -i '{1}/%05d.png' -filter:v \"crop={2}:{3}\" -c:v libx264 -crf 17 -pix_fmt yuv420p {4}".format(start_number, file_path, crop_x, crop_y, save)
        return ffmpeg_string


    def get_image_offsets(self, video_path):
        base_path = os.path.join(self.base_path, video_path)
        file_list = os.listdir(base_path)
        sorted(file_list)
        file_list = [os.path.join(self.base_path, video_path, file) for file in file_list]
        width, height, channel = cv2.imread(file_list[0]).shape

        if width % 2 == 1:
            width = width - 1
        
        if height % 2 == 1:
            height = height - 1

        return height, width


    def find_start_number(self, video_path):
        base_path = os.path.join(self.base_path, video_path)
        file_list = os.listdir(base_path)
        sorted(file_list)
        
        index = Path(file_list[0]).stem
        return str(int(index))


    def generate_video(self, video_path, filename):
        start_number = self.find_start_number(video_path)
        crop_x, crop_y = self.get_image_offsets(video_path)

        ffmpeg_string = self.create_ffmpeg_call(video_path, filename, start_number, crop_x, crop_y)
        os.system(ffmpeg_string)


if __name__ == "__main__":
    video_list = ["VIDEOS/CM_VIDEO", "VIDEOS/2D3D_CM", "VIDEOS/2D3D_MK", "VIDEOS/2D3D_MS", "VIDEOS/2D3D_SS"]

    for video in video_list:
        video_gen = VideoGenerator(video)
        video_gen.generate_video("0_Error", "error.mp4")
        video_gen.generate_video("0_GT", "video.mp4")
        video_gen.generate_video("0_GTPoints", "gtpoints.mp4")
        video_gen.generate_video("0_GTSeg", "gtseg.mp4")
        video_gen.generate_video("0_Points", "points.mp4")
        video_gen.generate_video("0_Seg", "seg.mp4")

    print("Done.")