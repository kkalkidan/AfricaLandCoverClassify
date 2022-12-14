import os
import numpy as np
from skimage import io
import matplotlib
from pathlib import Path

def mosaic(predictions, destination):
    os.system( f'gdal_merge.py -init 255 -o {destination} {predictions}/*')
    visualize(destination)

COLOR_DICT = {
    0:[0, 0, 0],
    1:[0, 0, 255],
    2:[53, 53, 53],
    3:[82, 64, 43],
    4:[96, 96, 100],
    5:[84, 30, 17],
    6:[9, 41, 9],
    7:[0, 100, 0]   
}

def visualize(destination):
    print(destination)
    img = io.imread(destination)
    
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    # print('img_out', img_out.shape)
    for i in range(len(COLOR_DICT)):
        img_out[img == i,:] = COLOR_DICT[i]
    # img_out = img_out / 255    
    matplotlib.image.imsave(f"{destination.split('.')[0]}.png", img_out/255)

if __name__ == "__main__":
    visualize("/home/kalkidan/AfricaLandCoverClassify/sentinel2_level2A_median_2022-10-24.tiff")