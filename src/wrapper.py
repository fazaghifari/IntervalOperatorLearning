import numpy as np
import joblib
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from keras import datasets

from timeit import default_timer

import warnings
warnings.filterwarnings("ignore")

def interval_midpoint(dat):
    mid = (dat[...,0] + dat[...,1])/2
    return mid


def dataset_generator(x,y,batch_size=100):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.shuffle(len(x)).batch(batch_size)
    return ds


def neg_distance_pen(dist):
     """Negative distance penalty"""
     return tf.where(dist < 0, (5*dist)**2, 0)


def linex_loss(d,a):
    """Linear-exponential loss"""
    b = 2
    term1 = (tf.exp(-a*d)+(a*d)-1)
    return b*term1


def trainer(model, x_train, y_train, epochs=200, linex_a=5):
    """Trainer wrapper for Interval MLP model

    Args:
        model (_type_): _description_
        data (_type_): _description_
        config_dict (_type_): _description_
    """
    loss_tracker = keras.metrics.Mean(name="Loss")
    result_history = []

    train_dataset = dataset_generator(x_train, y_train, batch_size=32)

    ### INNER FUNCTION FOR TRAIN STEP ###
    def train_step(data):
        """single train inner function

        Args:
            model (_type_): _description_
            data (_type_): _description_
            loss_tracker (_type_): _description_
        """
        inputs, target = data
        input_lo = inputs[...,0]
        input_hi = inputs[...,1]

        with tf.GradientTape() as tape:
            preds = model([input_lo, input_hi], training=True)
        
            # Extract upper and lower probabilities
            preds_lo, preds_up = preds
            
             # linex loss
            err_lo = (np.array(target[...,0], dtype=float) - preds_lo) #preds better be lower than target
            err_hi = (preds_up - np.array(target[...,1], dtype=float)) #target better be lower than preds
            term1 = linex_loss(err_lo, a=linex_a)
            term2 = linex_loss(err_hi, a=linex_a)
            dist = preds_up-preds_lo
            loss = tf.reduce_mean(term1) + tf.reduce_mean(term2) + neg_distance_pen(dist)

            # # Classical loss
            # mid_target = 0.5*(target_lo + target_hi)
            # mid_pred = 0.5*(preds_lo + preds_up)
            # term1 = tf.square(target_lo - preds_lo)
            # term2 = tf.square(target_hi - preds_up)
            # # term3 = tf.square(mid_target - mid_pred)
            # loss = tf.reduce_mean(term1 + term2 )
        
        grads = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        loss_tracker.update_state(loss)

        return loss, loss_tracker
    ### END INNER FUNCTION BLOCK ###

    for epoch in range(epochs):
        t1 = default_timer()
        for data in train_dataset:
            loss, _ = train_step(data)
        result_history.append(loss_tracker.result().numpy())

        t2 = default_timer()
        template = ("Epoch {}, Loss: {:.4f}, Time: {} s")
        print(template.format(epoch+1, loss_tracker.result(), t2-t1))
    
    return model

def model_evaluate(model, x_test, y_test):
    """_summary_

    Args:
        model (_type_): _description_
        x_test (_type_): _description_
        y_test (_type_): _description_
    """

    pred = model.predict([x_test, x_test])

    mid_target = interval_midpoint(y_test)
    mid_pred = interval_midpoint(pred)
    midpoint_loss = keras.losses.MSE(mid_target, mid_pred)

    return pred, midpoint_loss