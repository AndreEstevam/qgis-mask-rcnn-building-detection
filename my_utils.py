"""
Utility functions.

Most were used to train the model.
"""

import os
import numpy as np
from osgeo import gdal
from PIL import Image


def load_raster_as_np_array(raster_path):
    '''
    Takes a raster file and returns it as a numpy array.
    Args:
        raster_path[str] = path to raster file; may be a RGB raster(image) or 
        a single channel raster(mask);
    Return:
        img_np[numpy.array] = np array representing the raster input;
    '''
    raster = gdal.Open(raster_path)
    num_channels = raster.RasterCount
    
    if num_channels>=3:
        band1 = raster.GetRasterBand(1) # Red channel
        band2 = raster.GetRasterBand(2) # Green channel
        band3 = raster.GetRasterBand(3) # Blue channel
        
        b1 = band1.ReadAsArray()
        b2 = band2.ReadAsArray()
        b3 = band3.ReadAsArray()
        
        img_np = np.dstack((b1, b2, b3))
        
    elif num_channels==1:
        band1 = raster.GetRasterBand(1) # Instances channel
        img_np = band1.ReadAsArray()
        
    return img_np


def load_image_as_np_array(path):
    '''
    Takes path to a local image file and returns the image as a numpy array. 
    Does not work with raster data (load_raster_as_np_array alternative function).
    Args:
        - path[str] = path to image file;
    Return:
        - [numpy.array] = np array of shape (img_height, img_width, 3) OR  
        (img_height, img_width, 1);
    '''
    with open(path, 'rb') as file:
        PIL_img = Image.open(file)
        img_width, img_height = PIL_img.size
        PIL_img_data = PIL_img.getdata()
        img_np = np.array(PIL_img_data)
        
        if PIL_img.mode == 'RGB':
            return np.reshape(img_np, (img_height, img_width, 3))
        elif PIL_img.mode == "L":
            return np.reshape(img_np, (img_height, img_width))
        else:
            raise RuntimeError("load_image_as_np_array failed to load image as numpy array.")


def concat_through_first_axis(arr):
    '''
    Concatenates an array of shape (n, x, y) through the first axis, returning (x, y).
    Args:
        - arr[numpy.array] = a numpy array of shape (n, x, y);
    Return:
        - out_arr[numpy.array] = a numpy array of shape (x, y)
    
    obs.: numpy.concat() not working as expected.
    '''
    out_arr = arr[0]
    for arr_ in arr[1:]:
        out_arr = out_arr+arr_
    return out_arr


def from_prob_to_id(arr):
    '''
    Assign a single value for each detection stored in a numpy array of shape 
    (n° of detections, width, height).
    Args:
        - arr[numpy.array] = array of shape (n, x, y);
    Return:
        - out_arr[numpy.array] = array of shape (x, y), with np.unique(out_arr)
        returning "n" integer values;
    '''
    n = np.shape(arr)[0]
    count = 1
    for i in range(n):
        arr[i] = np.where(arr[i]>0, count, 0)
        count=count+1
    return arr


def single_dimension_mask(arr, threshold):
    '''
    This function prepares the mask to be saved as a raster and later be converted
    into a shape file (vector file). Here three process are chained: filter the mask 
    in respect to the threshold, for each detection assign an int value and, finally,
    considering an array of shape (n, x, y), concat through first dimension returning
    in a single band array of shape (x, y).
    Args:
        - arr[numpy.array] = an array containing values between 0 and 1;
        - threshold[float] = pixels below this value will be zeroed;
    Return:
        - out_array[numpy.array] = array of shape (x, y) containing only int values
        representing each detection;
    '''
    # filtering mask
    arr = np.where(arr>=threshold, arr, 0)
    
    # Rearranging detections values
    arr = from_prob_to_id(arr)
    
    # Concatenating array through first axis
    out_array = concat_through_first_axis(arr)
    
    return out_array


def save_array_as_raster(raster_path, arr, output_path, threshold):
    '''
    Writes a tensor to disk as a geotiff file;
    Args:
        - raster_path[string] = path of raster which the inference was performed upon;
        - arr[numpy.array] = array of shape (depth, image height, image width);
        - output_path[string] = full output file path excluding extension (default is
        .tif);
        - threshold[float] = threshold from which to filter the mask's instances;
    '''
    # instantiating raster as a gdal object
    raster_ref = gdal.Open(raster_path)
    
    # writing raster to disk
    height = arr.shape[1]
    width = arr.shape[2]
    driver = gdal.GetDriverByName('GTiff')
    raster_out = driver.Create(output_path+'.tif', width, height, 1, gdal.GDT_Int32)
    
    # setting coordinates and projection
    raster_out.SetGeoTransform(raster_ref.GetGeoTransform())
    raster_out.SetProjection(raster_ref.GetProjection())
    
    # writing array into raster 
    band = raster_out.GetRasterBand(1)
    arr = single_dimension_mask(arr, threshold)
    band.WriteArray(arr)
    band.FlushCache()


