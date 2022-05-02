import os
import numpy as np
import pandas as pd
import cv2
from glob import glob
import tensorflow as tf
import data
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from data import load_data, tf_dataset
from model import build_model

'''
 cette fonction iOU (Intersection over Union) IOU est principalement utilise dans les applications liees a la detection d'objets,
 ou nous entrainons un modele pour produire une boite qui s'adapte parfaitement autour d'un objet '''

def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)
if __name__ == "__main__":
    ## Dataset
    path = r"D:\PCD\curated"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = data.load_data( )

    ## Hyperparameters
    batch = 8
    lr = 1e-4
    epochs = 20

    train_dataset = tf_dataset(train_x, train_y, batch=batch)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch)

    model = build_model()

    opt = tf.keras.optimizers.Adam(lr)
    metrics = ["acc", tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), iou]
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=metrics)
    df= pd.read_excel (r'D:\PCD\metadata_good2.0.xlsx',engine='openpyxl')
    callbacks = [
        ModelCheckpoint(r"D:\PCD\train"),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4),
        CSVLogger("df"),
        TensorBoard(),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
    ]

    train_steps = len(train_x)//batch
    valid_steps = len(valid_x)//batch

    if len(train_x) % batch != 0:
        train_steps += 1
    if len(valid_x) % batch != 0:
        valid_steps += 1

    model.fit(train_dataset,
        validation_data=valid_dataset,
        epochs=epochs,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
        callbacks=callbacks)
    model.save ( 'model.h5' )

#a la fin de training on va trouver les var loss/acc/recall/precision
