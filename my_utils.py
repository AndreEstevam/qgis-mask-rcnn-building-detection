"""
Utility functions.

Most were used to train the model.

NEEDS PADRONIZATION:
ARRAYS WILL BE img_np OR arr?
RASTERS WIL BE raster_lyr OR raster?
EVERYTHING WILL HAVE ITS TYPE (path, np, lyr) BEFORE OR AFTER THE OBJECTS'S NAME?
THE DESCRIPTION OF EACH FUNCTION!!!
"""

import os
import numpy as np
from osgeo import gdal
from PIL import Image


def load_raster_as_np_array(raster_path):
    '''
    Takes a raster file(TIFF etc.) and returns it as a numpy array.
    Args:
        raster_path = path to raster file; may be a RGB raster(image) or a single channel raster(mask);
    Return:
        img_np = np array representing the raster input;
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
    """
    Takes path to a local rgb image and returns the image as a numpy array of shape (img_height, img_width, 3) 
    OR a single channel image returning a numpy array with shape (img_height, img_width, 1).
    Does not work with raster data (load_raster_as_np_array alternative function).
    Args:
        - path to image;
    Return:
        - numpy array;
    """
    with open(path, 'rb') as file:
        PIL_image = Image.open(file)
        PIL_image_width, PIL_image_height = PIL_image.size
        
        if PIL_image.mode == 'RGB':
            return np.array(np.reshape(PIL_image.getdata(), (PIL_image_height, PIL_image_width, 3)))
        elif PIL_image.mode == "L":
            return np.array(np.reshape(PIL_image.getdata(), (PIL_image_height, PIL_image_width)))
        else:
            raise RuntimeError("Failed to load image as numpy array.")


def concat_through_first_axis(arr):
    """
    Concatenates an array of shape (n, x, y) through the first axis, returning (x, y).
    Args:
        - arr[numpy.array] = a numpy array of shape (n, x, y);
    Return:
        - out_arr[numpy.array] = a numpy array of shape (x, y)
    
    obs.: numpy.concat() not working as expected.
    """
    out_arr = arr[0]
    for arr_ in arr[1:]:
        out_arr = out_arr+arr_
    return out_arr


def from_prob_to_id(arr):
    """
    Assign a single value for each detection stored in a numpy array of shape 
    (n° of detections, width, height).
    Args:
        - arr[numpy.array] = a numpy array of shape (n, x, y);
    Return:
        - out_arr[numpy.array] = a numpy array with same shape of arr, but with 
        np.unique(out_arr) returning "n" integer values;
        
    IS THIS RIGHT? (IS THERE A CLEANER WAY?)
    """
    n = np.shape(arr)[0]
    count = 1
    for i in range(n):
        arr[i] = np.where(arr[i]>0, count, 0)
        count=count+1
    return arr


def single_dimension_mask(arr, threshold):
    """
    This function prepares the mask to be saved as a raster and later be converted
    into a shape file (vector file). Here three process are chained: filter the mask 
    in respect to the threshold, for each detection assign an int value and, finally,
    considering an array of shape (n, x, y), concat through first dimension returning
    a single band array of shape (x, y).
    Args:
        - arr[numpy.array] = an array containing values between 0 and 1;
        - threshold[float] = pixels below this value will be zeroed;
    Return:
        - final_arr[numpy.array] = array of shape (x, y) containing only int values
        representing each detection;
    """
    # filtering mask
    arr = np.where(arr>=threshold, arr, 0)
    
    # Rearranging detections values
    arr = from_prob_to_id(arr)
    
    # Concatenating array through first axis
    final_arr = concat_through_first_axis(arr)
    
    return final_arr



def save_tensor_as_raster(raster_path, tensor, output_path, threshold):
    '''
    Writes a tensor to disk as a geotiff file;
    Args:
        - raster_path[string] = path of raster which the inference was performed upon;
        - tensor[torch.Tensor] = a tensor of shape (image height, image width, depth);
        - output_path[string] = full output file path excluding extension (default is
        .tif);
    '''
    # instantiating raster as a gdal object and it's detections mask as numpy array
    raster_ref = gdal.Open(raster_path)
    arr = tensor.detach().numpy()
    
    # writing raster to disk
    width = arr.shape[1]
    height = arr.shape[2]
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


def save_raster_as_polygon():
    pass