import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from tqdm import tqdm

from data import load_data , tf_dataset
from train import iou


def read_image ( path ) :
    x = cv2.imread ( path , cv2.IMREAD_COLOR )
    x = cv2.resize ( x , (256 , 256) )
    x = x / 255.0
    return x


def read_mask ( path ) :
    x = cv2.imread ( path , cv2.IMREAD_GRAYSCALE )
    x = cv2.resize ( x , (256 , 256) )
    x = np.expand_dims ( x , axis = -1 )
    return x


def mask_parse ( mask ) :
    mask = np.squeeze ( mask )
    mask = [ mask , mask , mask ]
    mask = np.transpose ( mask , (1 , 2 , 0) )
    return mask


if __name__ == "__main__" :
    # Dataset
    path = pd.read_excel ( r'D:\PCD\metadata_good2.0.xlsx' , engine = 'openpyxl' )

    batch_size = 8
    (train_x , train_y) , (valid_x , valid_y) , (test_x , test_y) = load_data ( )

    test_dataset = tf_dataset ( test_x , test_y , batch = batch_size )

    test_steps = (len ( test_x ) // batch_size)
    if len ( test_x ) % batch_size != 0 :
        test_steps += 1

    with CustomObjectScope ( {'iou' : iou} ) :
        new_model = tf.keras.models.load_model ( 'model.h5' )


    new_model.evaluate ( test_dataset , steps = test_steps )

    for i , (x , y) in tqdm ( enumerate ( zip ( test_x , test_y ) ) , total = len ( test_x ) ) :
        x = read_image ( x )
        y = read_mask ( y )
        y_pred = new_model.predict ( np.expand_dims ( x , axis = 0 ) ) [ 0 ] > 0.5
        h , w , _ = x.shape
        white_line = np.ones ( (h , 10 , 3) ) * 255.0

        all_images = [
            x * 255.0 , white_line ,
            mask_parse ( y ) , white_line ,
            mask_parse ( y_pred ) * 255.0
        ]
        image = np.concatenate ( all_images , axis = 1 )
        cv2.imwrite ( f"D:\\PCD\\results/{i}.png" , image )

# on enregistre resultats dans un dossier results ou on trouve une concatination entre l'image-masque-l'image segmentee
