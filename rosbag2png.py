import pyrealsense2 as rs
import numpy as np
import cv2
import os
import shutil
import argparse

# reset data folder
if os.path.exists('data'):
    shutil.rmtree('data')
os.mkdir('data')
os.mkdir('data/color')
os.mkdir('data/depth')

# input
parser = argparse.ArgumentParser(description="Read recorded bag file and display depth stream in jet colormap.\
                                Remember to change the stream resolution, fps and format to match the recorded.")
parser.add_argument("-i", "--input", type=str, help="Path to the bag file")
args = parser.parse_args()
if not args.input:
    print("No input paramater have been given.")
    print("For help type --help")
    exit()
if os.path.splitext(args.input)[1] != ".bag":
    print("The given file is not of correct file format.")
    print("Only .bag files are accepted")
    exit()

pipeline = rs.pipeline()
config = rs.config()
rs.config.enable_device_from_file(config, args.input)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

align_to = rs.stream.color
align = rs.align(align_to)

pipeline.start(config)

i = 0
try:
    while True:
        frames = pipeline.wait_for_frames()
        timestamp = int(frames.get_timestamp() * 10000)
        print(timestamp)

        frames = align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        image_color = np.array(color_frame.get_data())
        image_color = cv2.cvtColor(image_color, cv2.COLOR_RGB2BGR)
        image_depth = np.array(depth_frame.get_data())

        cv2.imwrite('data/color/color{:012}.png'.format(timestamp), image_color)
        cv2.imwrite('data/depth/depth{:012}.png'.format(timestamp), image_depth)
        i += 1

finally:
    pipeline.stop()
