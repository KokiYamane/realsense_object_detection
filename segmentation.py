import glob
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from tqdm import tqdm


def segmentation(image):
    ret, thresh = cv2.threshold(
        image, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # thresh[image == 0] = 0

    # add outline
    laplacian = cv2.Laplacian(image, cv2.CV_8U, ksize=5)
    ret, line = cv2.threshold(
        laplacian, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    line = cv2.medianBlur(line, ksize=3)
    thresh = cv2.bitwise_or(thresh, thresh, mask=line)

    # noise removal
    thresh = cv2.medianBlur(thresh, ksize=5)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # make distance image
    tmp = cv2.convertScaleAbs(opening, alpha=(255))
    dist_transform = cv2.distanceTransform(tmp, cv2.DIST_L2, 5)

    # Finding sure foreground area
    ret, sure_fg = cv2.threshold(
        dist_transform, 0.05 * dist_transform.max(), 1, 0)

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

    tmp = cv2.convertScaleAbs(image, alpha=(255 / 65535))
    color = cv2.cvtColor(tmp, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(color, markers)

    return markers


class AnimationMaker():
    def __init__(self, image_list, markers_list, fps=60):
        self.image_list = image_list
        self.markers_list = markers_list
        self.interval = 1000 / fps
        self._init_figure()

    def _init_figure(self):
        self.fig, self.ax = plt.subplots(1, 2, figsize=(12, 6))
        self.ax[0].set_title('image')
        self.ax[1].set_title('segmentation')

    def _update(self, i):
        self.ax[0].imshow(self.image_list[i], cmap='jet')
        self.ax[1].imshow(self.markers_list[i], cmap='tab20')
        print('{}/{}'.format(i, len(self.image_list)))

    def makeAnimation(self):
        return animation.FuncAnimation(self.fig, self._update,
                                       interval=self.interval, frames=len(self.image_list))


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

    folder_color = 'images/flying_arrow/color'
    folder_depth = 'images/flying_arrow/depth'

    filename_list_color = list(pathlib.Path(folder_color).glob('*.png'))
    filename_list_depth = list(pathlib.Path(folder_depth).glob('*.png'))

    # fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    # ax[0].set_title('image')
    # ax[1].set_title('segmentation')

    # image_list = []
    # markers_list = []
    # i = 0
    for i in tqdm(range(len(filename_list_depth))):
        image_color = cv2.imread(
            '{}/{}'.format(folder_color, filename_list_color[i].name))
        image_depth = cv2.imread(
            '{}/{}'.format(folder_depth, filename_list_depth[i].name), cv2.IMREAD_ANYDEPTH)
        markers = segmentation(image_depth)

        markers_color = cv2.convertScaleAbs(
            markers, alpha=(255 / np.max(markers)))
        markers_color = cv2.applyColorMap(markers_color, cv2.COLORMAP_RAINBOW)
        markers_color[markers == 1] = [0, 0, 0]
        blended = cv2.addWeighted(
            src1=image_color,
            alpha=0.7,
            src2=markers_color,
            beta=0.3,
            gamma=0)
        cv2.imwrite('results/result{:08}.png'.format(i), blended)
        # i += 1

        # image_list.append(image)
        # markers_list.append(markers)

        # ax[0].imshow(image)
        # ax[1].imshow(markers, cmap='tab20')
        # # plt.pause(0.02)
        # plt.savefig('tmp/{}'.format(filename.name))

    # ani = AnimationMaker(image_list, markers_list).makeAnimation()
    # ani.save('animation.gif', writer='pillow')

    print('Creating gif file ...')
    make_gif('results', 'result.gif')
