import glob
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from tqdm import tqdm
import copy


def segmentation(image_color):
    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(image_gray, (7, 7))
    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,
        9)
    kernel = np.ones((9, 9), np.uint8)
    sure_bg = cv2.dilate(thresh, kernel, iterations=2)

    # # noise removal
    thresh = cv2.medianBlur(thresh, ksize=5)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # # sure background area
    kernel = np.ones((9, 9), np.uint8)
    sure_bg = cv2.dilate(opening, kernel, iterations=2)

    # make distance image
    dist_transform = cv2.distanceTransform(sure_bg, cv2.DIST_L2, 5)

    # Finding sure foreground area
    ret, sure_fg = cv2.threshold(
        dist_transform, 0.2 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    sure_bg = np.uint8(sure_bg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv2.watershed(image_color, markers)

    return markers


def select_marker(markers):
    markers_selected = copy.deepcopy(markers)
    markers[markers == -1] = 0
    label_num = np.max(markers)
    for i in range(2, label_num + 1):
        mask = np.uint8(markers)
        mask[markers != i] = 0
        mask[markers == i] = 1

        point_num = np.sum(mask)
        if point_num < 500 or 10000 < point_num:
            markers_selected[mask == 1] = 1

    return markers_selected


def make_gif(folder, filename, extention='png', fps=30):
    files = sorted(glob.glob('{}/./*.{}'.format(folder, extention)))
    images = list(map(lambda file: Image.open(file), files))
    duration = int(1000 / fps)
    images[0].save('{}/{}'.format(folder, filename),
                   save_all=True,
                   append_images=images[1:],
                   duration=duration,
                   loop=0)


if __name__ == '__main__':
    import pathlib
    import shutil
    import os

    if os.path.exists('results'):
        shutil.rmtree('results')

    os.mkdir('results')

    folder_color = 'images/ground_arrow/color'
    folder_depth = 'images/ground_arrow/depth'

    filename_list_color = list(pathlib.Path(folder_color).glob('*.png'))
    filename_list_depth = list(pathlib.Path(folder_depth).glob('*.png'))

    def marker2colormap(markers):
        markers_color = cv2.convertScaleAbs(
            markers, alpha=(255 / np.max(markers)))
        markers_color = cv2.applyColorMap(markers_color, cv2.COLORMAP_RAINBOW)
        markers_color[markers == -1] = [0, 0, 0]
        markers_color[markers == 0] = [0, 0, 0]
        markers_color[markers == 1] = [0, 0, 0]
        return markers_color

    for i in tqdm(range(len(filename_list_depth))):
        image_color = cv2.imread(
            '{}/{}'.format(folder_color, filename_list_color[i].name))
        image_depth = cv2.imread(
            '{}/{}'.format(folder_depth, filename_list_depth[i].name), cv2.IMREAD_ANYDEPTH)
        markers = segmentation(image_color)
        markers = select_marker(markers)

        # markers_color = cv2.convertScaleAbs(
        #     markers, alpha=(255 / np.max(markers)))
        # markers_color = cv2.applyColorMap(markers_color, cv2.COLORMAP_RAINBOW)
        # markers_color[markers == 1] = [0, 0, 0]
        markers_color = marker2colormap(markers)
        blended = cv2.addWeighted(
            src1=image_color,
            alpha=0.7,
            src2=markers_color,
            beta=0.3,
            gamma=0)
        cv2.imwrite('results/result{:08}.png'.format(i), blended)

    print('Creating gif file ...')
    make_gif('results', 'result.gif')
