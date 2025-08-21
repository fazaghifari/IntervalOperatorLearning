import numpy as np
import joblib
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from keras import datasets
from src import helper

from timeit import default_timer

import warnings
warnings.filterwarnings("ignore")

def dataset_generator(y,u,g,batch_size=100):
    ds = tf.data.Dataset.from_tensor_slices((y,u,g))
    ds = ds.shuffle(len(y)).batch(batch_size)
    return ds

def linex_loss(d,a):
    """Linear-exponential loss"""
    b = 2
    term1 = (tf.exp(-a*d)+(a*d)-1)
    return b*term1

def trainer(model, y_train, u_train, g_train, u_val=None, y_val=None, g_val=None, epochs=200, 
            scheduler=None, loss_type="linex", batch=512, verbose=True, filename=None):
    """Trainer wrapper for Interval MLP model

    Args:
        model (_type_): _description_
        data (_type_): _description_
        config_dict (_type_): _description_
    """
    loss_tracker = keras.metrics.Mean(name="Loss")
    result_history = []
    val_history = []

    train_dataset = dataset_generator(y_train, u_train, g_train, batch_size=batch)

    ### INNER FUNCTION FOR TRAIN STEP ###
    def train_step(data):
        """single train inner function

        Args:
            model (_type_): _description_
            data (_type_): _description_
            loss_tracker (_type_): _description_
        """
        y_inputs, u_inputs, target = data

        with tf.GradientTape() as tape:
            preds = model([y_inputs, u_inputs], training=True)
        
            # Extract upper and lower probabilities
            preds_lo, preds_up = preds

            if loss_type == "linex":
                # Compute the linear-exponential loss
                term1 = linex_loss(np.array(target[...,0], dtype=float) - preds_lo, a=5)
                term2 = linex_loss(preds_up - np.array(target[...,1], dtype=float), a=5)
                loss = tf.reduce_mean(term1+term2)

            elif loss_type == "mse":
                # classical loss
                term1 = tf.square(np.array(target[...,0], dtype=float) - preds_lo)
                term2 = tf.square(np.array(target[...,1], dtype=float) - preds_up)
                loss = tf.reduce_mean(term1+term2)

            else:
                raise ValueError("Unknown loss function")

        grads = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        loss_tracker.update_state(loss)

        return loss, loss_tracker
    ### END INNER FUNCTION BLOCK ###

    best_loss = np.inf
    best_weights = None
    for epoch in range(epochs):
        
        if scheduler is not None:
            new_lr = scheduler(epoch)
            model.optimizer.lr.assign(new_lr)

        t1 = default_timer()
        for data in train_dataset:
            loss, _ = train_step(data)
        result_history.append(loss_tracker.result().numpy())

        if u_val is None:
            t2 = default_timer()
            template = ("Epoch {}, Loss: {:.4f}, Time: {} s")
            if verbose:
                print(template.format(epoch+1, loss_tracker.result(), t2-t1))

            if epoch > 100 and loss_tracker.result() < best_loss:
                best_loss = loss_tracker.result()
                best_epoch = epoch
                best_weights = model.get_weights()  # store a copy of weights in memory
                print(f"Best model updated at epoch {best_epoch+1} with loss {best_loss:.4f}")

        else:
            pred_val = model.predict([y_val, u_val], verbose=0)
            preds_lo_val, preds_up_val = pred_val
            term1_val = tf.square(np.array(g_val[...,0], dtype=float) - preds_lo_val)
            term2_val = tf.square(np.array(g_val[...,1], dtype=float) - preds_up_val)
            loss_val = tf.reduce_mean(term1_val + term2_val)
            val_history.append(loss_val)

            t2 = default_timer()
            template = ("Epoch {}, Loss: {:.4f}, Val Loss: {:.4f}, Time: {} s")
            if verbose:
                print(template.format(epoch+1, loss_tracker.result(), loss_val, t2-t1))
            
            if epoch > 100 and loss_val < best_loss:
                best_loss = loss_val
                best_epoch = epoch
                best_weights = model.get_weights()  # store a copy of weights in memory
                print(f"Best model updated at epoch {best_epoch+1} with val_loss {best_loss:.4f}")

    if best_weights is not None:
        model.set_weights(best_weights)

    history = [result_history, val_history]
    return model, history

def model_evaluate(model, x_test, y_test):
    """_summary_

    Args:
        model (_type_): _description_
        x_test (_type_): _description_
        y_test (_type_): _description_
    """

    pred = model.predict([x_test, x_test])

    mid_target = helper.interval_midpoint(y_test)
    mid_pred = helper.interval_midpoint(pred)
    midpoint_loss = keras.losses.MSE(mid_target, mid_pred)

    return pred, midpoint_loss