import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from glob import glob
from sklearn.model_selection import train_test_split


def p_join ( path1 , path2 ) :
    return (os.path.join ( path1 , path2 ))

# fonction load_data tres importante ,ou on manipule data et les variables train predict on va les utilser

def load_data ( split = 0.1 ) :
    parent_dir = r"D:\PCD\curated"
    df = pd.read_excel ( r'D:\PCD\metadata_good2.0.xlsx' , engine = 'openpyxl' )
    ROI_mask_paths = df [ 'ROI mask file path' ]
    im_paths = df [ 'image full mammo path' ]
    sample = (list ( im_paths ) [ :5 ])
    ROI_mask_paths_sample = (list ( ROI_mask_paths ) [ :5 ])
    images = [ p_join ( parent_dir , path ) for path in sample ]
    masks = [ p_join ( parent_dir , path ) for path in ROI_mask_paths_sample ]

    total_size = len ( images )
    valid_size = float ( split * total_size )
    test_size = float ( split * total_size )

    train_x , valid_x = train_test_split ( images , test_size = valid_size , random_state = 42 )
    train_y , valid_y = train_test_split ( masks , test_size = valid_size , random_state = 42 )

    train_x , test_x = train_test_split ( train_x , test_size = test_size , random_state = 42 )
    train_y , test_y = train_test_split ( train_y , test_size = test_size , random_state = 42 )

    return (train_x , train_y) , (valid_x , valid_y) , (test_x , test_y)


def read_image ( path ) :
    path = path.decode ( )
    x = cv2.imread ( path , cv2.IMREAD_COLOR )
    x = cv2.resize ( x , (256 , 256) )
    x = x / 255.0
    return x


def read_mask ( path ) :
    path = path.decode ( )
    x = cv2.imread ( path , cv2.IMREAD_GRAYSCALE )
    x = cv2.resize ( x , (256 , 256) )
    x = x / 255.0
    x = np.expand_dims ( x , axis = -1 )
    return x


def tf_parse ( x , y ) :
    def _parse ( x , y ) :
        x = read_image ( x )
        y = read_mask ( y )
        return x , y

    x , y = tf.numpy_function ( _parse , [ x , y ] , [ tf.float64 , tf.float64 ] )
    x.set_shape ( [ 256 , 256 , 3 ] )
    y.set_shape ( [ 256 , 256 , 1 ] )
    return x , y


def tf_dataset ( x , y , batch = 8 ) :
    dataset = tf.data.Dataset.from_tensor_slices ( (x , y) )
    dataset = dataset.map ( tf_parse )
    dataset = dataset.batch ( batch )
    dataset = dataset.repeat ( )
    return dataset
