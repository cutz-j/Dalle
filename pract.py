
import random
import numpy as np
import os
import pickle
from PIL import Image
import click
#from utils.config import Config
#from utils.visualization.plot_images_grid import plot_images_grid
#from datasets.main import load_dataset
import torch
import imageio

import matplotlib
import sys
import glob


from pathlib import Path
################################################################################
# Settings
# ###############################################################################
@click.command()
@click.argument('dataset_name',  type=int, default=0)
def main(dataset_name):
    
    
    aa = glob.glob("/vision/7016118/ori_codes/2D-Shape-Generator/output2/*.png")
    #print(aa)
    #print(aa[1])
    
    for i in range(2):
        if i%100==0:
            print(i)
            
        imname = aa[i]
        bb = imname[52:]
        im = imageio.imread(imname)
        #im = im[:,:,:-1]
        print(np.max(np.max(im)))
        im = im/255
        imageio.imwrite('/vision/7016118/ori_codes/2D-Shape-Generator/output3/'+ bb, im)
        im = imageio.imread('/vision/7016118/ori_codes/2D-Shape-Generator/output3/'+ bb)
        print(np.max(np.max(im)))
    
    
    
    
    
    
    
if __name__ == '__main__':
    main()
