# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 15:26:14 2023

@author: Sarina Söhl & Henning Wache
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import cv2
import click
import time

def GetGray(images):
    '''
    Calculates gray scale version of an image or a list/array of images.
    '''
    #calculate grayscale image by using the following luminosity coefficients for R G and B
    luminosity_constant = [0.2126, 0.7152, 0.0722]#[1/3, 1/3, 1/3]

    if isinstance(images, list):
        images_gray = []
        for img in images:
            
            #calculate via matrix multiplication of img array and luminosity vector
            img_gray = np.dot(img, luminosity_constant).astype('uint8')
            images_gray.append(img_gray)
        return images_gray
    else:
        img_gray = np.dot(images, luminosity_constant).astype('uint8')
        return img_gray


def GetImgsGreyRGB(img):
    '''
    This function returns the original, grayscaled and seperated R G B versions of an image.
    
    Parameters
    ----------
    img : numpy.ndarray
        original image
        
    Returns
    ----------
    images : list of numpy.ndarray
        list with original, grayscaled and R G B versions of img
    '''

    #create copies of the image and set specific color channels to zero (0 Red, 1 Green, 2 Blue)
    #return images with R G and B values
    
    '''
    img_R = img.copy()
    img_R[:, :, (1, 2)] = 0
    
    img_G = img.copy()
    img_G[:, :, (0, 2)] = 0
    
    img_B = img.copy()
    img_B[:, :, (0, 1)] = 0
    '''
    img_R = img[:, :, 0]
    img_G = img[:, :, 1]
    img_B = img[:, :, 2]

    img_gray = GetGray(img)
    
    images = [img, img_gray, img_R, img_G, img_B]
    return images

    
def LaplaceFilter(images):
    '''
    This function uses a kernel and convolution to apply a laplace like filter to an image or list of images.
    The returned value is an image or list of images with high edge contrast.
    
    If list of images is used, this function will also automatically extract color values as 2d arrays for laplace filtering.
    '''
    kernel = np.array([[1, 1, 1], [1, -15, 1], [1, 1, 1]]) #laplass filter based kernel for high contrast edge detection
    if isinstance(images, list):
        images_laplace = []
        for img in images:
            #print('New image to filter:' + f' {img.shape}')
            #check if image is grayscale, if not extract color channel as 2d array
            if len(img.shape) != 2:
                count = 0
                #check for all zero channels and slice if nonzero values exist
                for ii in [0, 1, 2]:
                    if not np.any(img[::, ::, ii]):
                        #print('All zero')
                        continue
                    else:
                        #print('Color found')
                        img_color = img[::, ::, ii]
                        count += 1
                if count > 1 :
                    #print('LaplaceFilter(): Too many color values. Image will be automatically grayscaled.')
                    img = GetGray(img)
                else:
                    img = img_color
            images_laplace.append(sc.signal.convolve2d(img, kernel, mode='same'))
        return images_laplace
    #if single image: return laplace filtered result directly
    else:
        img = images
        img_laplace = sc.signal.convolve2d(img, kernel, mode='same')
    return img_laplace


def DrawTargetLines(x_lim, y_lim, center_pos):
    '''
    This function draws a horizontal and vertical line across an image.
    '''
    row, col = center_pos
    x_l, x_u = 0, x_lim
    y_l, y_u = 0, y_lim
    global c_hori
    global c_vert
    c_hori = 'blue'
    c_vert = 'orange'
    plt.hlines(row, x_l, x_u, color=c_hori) #y, min, max
    plt.vlines(col, y_l, y_u, color=c_vert) #x, min, max


def RotateImage(img):
    '''
    This function rotates an image of type numpy.ndarray and plots the result. The rotation can be controlled with an input angle of type int.
    '''
    save = img
    plt.figure(dpi=300)
    center_pos = img.shape[0]/2, img.shape[1]/2
    #DrawTargetLines(img.shape[1]-1, img.shape[0]-1, center_pos)
    plt.imshow(img, cmap='gray')
    
    plt.show()
    while True:
        angle = input('Input Angle to rotate image clockwise, reset image with r. If finished, input c:\n')
        try: 
            int(angle)
        except:
            is_angle = False
        else:
            is_angle = True
        if angle in ['C', 'c']:
            break
        if angle in ['R', 'r']:
            img = save
        if is_angle:
            angle = -int(angle)
            img = sc.ndimage.rotate(img, angle)
        else:
            print('Wrong input (only integer values)')
            continue
        plt.figure(dpi=300)
        #DrawTargetLines(img.shape[1]-1, img.shape[0]-1, center_pos)
        plt.imshow(img, cmap='gray')
        plt.show()
    return img


def ShiftCenter(img, rowcol):
    '''
    This function shifts an image (np.ndarray) in x or y direction.
    rowcol : tuple of int
        value of shift dx, dy
    '''
    row, col = rowcol
    height, width = img.shape[:2]
    dy = row
    dx = col
    
    DrawTargetLines(img.shape[1]-1, img.shape[0]-1, (dx,dy))
    '''
    img = np.roll(img, dy, axis=0)
    img = np.roll(img, dx, axis=1)
    if dy>0:
        img[:dy, :] = 0
    elif dy<0:
        img[dy:, :] = 0
    if dx>0:
        img[:, :dx] = 0
    elif dx<0:
        img[:, dx:] = 0
    
    return img
    '''

def FindCenterManually(img):
    '''
    This function displays an image and lets you shift the center to a new location by user input (wasd or dx dy)
    Input c to set new centerpoint and return image.

    Parameters
    ----------
    img : numpy.ndarray
        image that is to be altered.

    Returns
    -------
    img : numpy.ndarray.
        image with new center
    '''
    center_pos = img.shape[0]/2, img.shape[1]/2
    plt.figure(dpi=300)
    DrawTargetLines(img.shape[1]-1, img.shape[0]-1, center_pos)
    plt.imshow(img, cmap='gray')
    plt.show()

    dx = dx_save = img.shape[1]//2
    dy = dy_save = img.shape[0]//2
    while True:
        direction = input('Input WASD or x y position to move target, reset with r. If centered, input c. \n')
        try:
            int(direction.split()[1])
        except:
            is_pos = False
            pass
        else:
            is_pos = True
        if direction in ['W', 'w']: 
            dy += -10
        elif direction in ['S', 's']: 
            dy += 10
        elif direction in ['A', 'a']: 
            dx += -10
        elif direction in ['D', 'd']: 
            dx += 10
        elif direction in ['C', 'c']:
            #plt.axis('off')
            #plt.savefig('target.jpg', bbox_inches='tight', pad_inches=0.0, dpi=720)
            break
        elif direction in ['R', 'r']:
            dx, dy = dx_save, dy_save
            plt.close('all')
        elif is_pos:
            dx = int(direction.split(' ')[0])
            dy = int(direction.split(' ')[1])
        else:    
            print('Invalid input (W A S D  or  dx dy)')
            continue
        plt.figure(dpi=300)
        ShiftCenter(img, (dx, dy))
        print(f'Currently at {dx, dy}')
        plt.imshow(img, cmap='gray')
        plt.show()
    #DrawTargetLines(img.shape[1]-1, img.shape[0]-1, center_pos)
    return (dx,dy)


def FindCenter(img, filename):
    '''
    Find Center of Wood Growth Rings by matching with given template.
    
    img : grey scaled image of type numpy array
    center : tuple of image coordiantes of growth rings center point 
    '''
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    template = cv2.imread('Input/center_template.bmp')
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    w, h = template_gray.shape[::-1]

    res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)

    #Select all values that are near the template with a given threshold
    threshold = 0.45
    loc = np.where(res >= threshold)
    #if matching is not successfull, get center manually
    if loc[0].size == 0:
        print('Center not found, please select manually:')
        #plt.figure(dpi=300)
        #plt.imshow(img)
        #DrawTargetLines(img.shape[1], img.shape[0], (img.shape[0]//2, img.shape[1]//2))
        #plt.xlim(0, img.shape[1])
        #plt.ylim(img.shape[0], 0)
        #plt.show()
        center = FindCenterManually(img)
        return center
    #if matching is successfull, select center coordinate as middle of loc array
    else:
        #for pt in zip(*loc[::-1]):
        #    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 1)
        loc = loc[::-1]
        center = (loc[0][loc[0].size//2] + w//2, loc[1][loc[1].size//2] + h//2)    
        
        plt.figure(dpi=300)
        #plt.scatter(x=center[0], y=center[1], marker='+', s=20, color='cyan', zorder=10)
        DrawTargetLines(img.shape[1]-1, img.shape[0]-1, (center[1], center[0]))
        plt.title(f'Center found at {center}')
        plt.imshow(img)
        plt.savefig(f'Output/{filename}/CenterFound.png', dpi=720)
        plt.show()
        
    if click.confirm('Correct center found?', default=True):
        return center
    else:
        return FindCenterManually(img)


def PrintImage(img, cmap='gray', appendix='', filename='NewWood'):
    plt.figure(dpi=300)
    if cmap=='None':
        plt.imshow(img)
    else:
        plt.imshow(img, cmap=cmap)
    plt.title(f'Growth Rings {appendix}')
    plt.savefig(f'Output/{filename}/{filename}_{appendix}')


def SliceAtCenter(img, center, intvl):
    rows = img[center[1]-intvl : center[1]+intvl, ::]
    cols = img[::, center[0]-intvl : center[0]+intvl]
    
    row = np.average(rows, axis=0)
    col = np.average(cols, axis=1)
    
    row = row/np.median(row)
    col = col/np.median(col)
    return row, col


def FindPeaks(array):
    height = np.array([.4, 1.5])
    distance = 5
    prominence = .075
    #threshold = 1
    width = 3.5

    minima = sc.signal.find_peaks(-array, height=tuple(-1*height[::-1]), distance=distance, width=width, prominence=prominence)
    minima_x = minima[0]
    minima_y = -minima[1]['peak_heights']
    
    peaks = sc.signal.find_peaks(array, height=height, distance=distance, width=width, prominence=prominence)
    peaks_x = peaks[0]
    peaks_y = peaks[1]['peak_heights']
    
    return minima_x, minima_y, peaks_x, peaks_y


def CountGrowthRings(img_orig, img_lap, center, filename):
    '''
    This function uses the SliceAtCenter() function to get a single horizontal row and vertical col of the laplace filtered image.
    These are evaluated by the FindPeaks() function, which returns the positions of local maximums (peaks) and minimums (troughs). 
    The number of these is the double growth ring count (rings go around image).
    So to get an average, the sum must be divided by four to get the total number of growth rings in the given direction.
    
    The laplace filtered values (standardized by median) are plotted with the peaks and troughs and the original image in the background.
    
    img_lap : laplace filtered image as numpy.ndarray (uint8)
    img_orig : the original image used for background as numpy.ndarray (uint8)
    
    returns:
    ringcount : total number of growth rings as int 
    '''
    intvl = 10
    row, col = SliceAtCenter(img_lap, center, intvl)
    
    total_ringcount = 0
    for ii, arr in enumerate([row, col]):
    
        minima_x, minima_y, peaks_x, peaks_y = FindPeaks(arr)
        ringcount = (minima_x.size + peaks_x.size) // 4 
        total_ringcount += ringcount
        
        x_min, x_max = 0, arr.size
        ymin, ymax = center[ii]-intvl, center[ii]+intvl
        
        plt.figure(dpi=720)

        fig, ax = plt.subplots()
        ax1 = ax.twinx()
        slice_type = ['horizontal', 'vertical']
        ax.set_title(f'{slice_type[ii].capitalize()} slice of {filename}')
        ax.imshow(img_orig, aspect='auto')
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(x_min, x_max)
        
        ax1.plot(arr, color='powderblue', label='Laplace Filtered', linewidth=.5)
        ax1.scatter(minima_x, minima_y, marker='x', s=20, color='midnightblue', label='Troughs')
        ax1.scatter(peaks_x, peaks_y, marker='x', s=20, color='darkred', label='Peaks')
        
        ax.annotate(f'{ringcount} growth rings found!', xy= (.05, .05), xycoords='axes fraction', color='white')
        plt.legend(loc='lower right', facecolor='white', framealpha=.5)
        
        plt.savefig(f'Output/{filename}/GrowthRingsCount_{slice_type[ii]}.png', dpi=720)
        
        
    return total_ringcount//2
