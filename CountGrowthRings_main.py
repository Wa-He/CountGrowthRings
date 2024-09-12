# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 15:26:14 2023

@author: Sarina Söhl & Henning Wache
"""

import matplotlib.pyplot as plt
import time
import click

#import functions from seperate file
from CGR_functions import GetGray, GetImgsGreyRGB, RotateImage, LaplaceFilter, FindCenter, PrintImage, SliceAtCenter, FindPeaks, CountGrowthRings


print('Welcome to Count Growth Rings: A python script to count tree growth rings from an image.\n')
if click.confirm('Do you want to print all color images?', default=True):
    print_imgs = True
else:
    print_imgs = False

path = 'Input/wood_1.bmp'
filename = path.split('.')[0].split('/')[1]



plt.figure(dpi=300)
img = plt.imread(f'{path}')
plt.show()

img = RotateImage(img)


images = GetImgsGreyRGB(img)
img_orig, img_gray, img_R, img_G, img_B = images


images_laplace = LaplaceFilter(images)

center = FindCenter(img, filename)
time.sleep(1)


cmap = ['None', 'gray', 'Reds', 'Greens', 'Blues']
appendix = ['orig', 'gray', 'red', 'green', 'blue']

for ii, img in enumerate(images):
    if ii == 0:
        if print_imgs: PrintImage(img, cmap='None', appendix=appendix[ii], filename=filename)
        continue #skip laplace filter for original image
    #if ii >= 2:
    #    img = img[::,::, ii-2] #select only color values as 2d array to apply laplace filter
    
    img_lap = images_laplace[ii]
    
    if print_imgs: PrintImage(img, cmap[ii], appendix[ii], filename=filename)
    if print_imgs: PrintImage(img_lap, cmap[ii], appendix[ii] + '_laplace', filename=filename)



ringcount = CountGrowthRings(img_orig, images_laplace[1], center, filename)

print(f'The total number of growth rings is {ringcount}')












