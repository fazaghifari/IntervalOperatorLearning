import math
import joblib
import keras
import h5py
import scipy

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import activations, layers, initializers
from keras.models import Model

from timeit import default_timer
from src.inn_layers_v2 import IntAdd, IntDense, IntConv2D, IntDropout, IntFlatten, IntMultiply, IntSilu, IntGelu
from src.inn_layers_v2 import IntZeroPadding2D, IntBatchNormalization, IntRelu, IntTanh, IntMultiply_donet
from src.deeponet_wrapper import trainer

from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from itertools import product
import argparse

class MatReader(object):
    def __init__(self, file_path, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_float(self, to_float):
        self.to_float = to_float

def create_data(path, n_train = 1000, n_val=500, n_test=200, rand = 42):
    reader = MatReader(path)

    r = 2
    h = int(((101 - 1)/r) + 1)
    s = h

    x_read_min = np.array(reader.read_field('boundCoeff_min')[:,::r,::r][:,:s,:s])
    x_read_max = np.array(reader.read_field('boundCoeff_max')[:,::r,::r][:,:s,:s])
    y_read_min = np.array(reader.read_field('sol_min')[:,::r,::r][:,:s,:s])
    y_read_max = np.array(reader.read_field('sol_max')[:,::r,::r][:,:s,:s])

    # Training data

    n_total = x_read_min.shape[0]
    n_test_all = n_total - n_train

    idxs = np.arange(x_read_min.shape[0])
    idx_train, idx_test = train_test_split(idxs, test_size=n_test_all/n_total,
                                            random_state=rand)
    idx_val, idx_test = train_test_split(idx_test, train_size=n_val/n_test_all,
                                            random_state=rand)

    x_min = x_read_min[:n_total].reshape((n_total, 1, s, s))  # X input
    x_max = x_read_max[:n_total].reshape((n_total, 1, s, s))  # X input
    y_min = y_read_min[:n_total].reshape((n_total, s, s, 1))  # HF output
    y_max = y_read_max[:n_total].reshape((n_total, s, s, 1))  # HF output

    idx_test = idx_test[:n_test, ...] # Get the actual number "n_test" test data

    u_train_min, g_train_min = x_min[idx_train, ...], y_min[idx_train, ...]
    u_train_max, g_train_max = x_max[idx_train, ...], y_max[idx_train, ...]
    u_val_min, g_val_min = x_min[idx_val, ...], y_min[idx_val, ...]
    u_val_max, g_val_max = x_max[idx_val, ...], y_max[idx_val, ...]
    u_test_min, g_test_min = x_min[idx_test, ...], y_min[idx_test, ...]
    u_test_max, g_test_max = x_max[idx_test, ...], y_max[idx_test, ...]

    # Get mid values for u_train, g_train, u_val, and g_val
    u_train_mid = 0.5 * (u_train_min + u_train_max)
    g_train_mid = 0.5 * (g_train_min + g_train_max)
    u_val_mid = 0.5 * (u_val_min + u_val_max)
    g_val_mid = 0.5 * (g_val_min + g_val_max)

    # x space
    n_train = u_train_min.shape[0]
    n_val = u_val_min.shape[0]
    n_test = u_test_min.shape[0]

    xspace = np.linspace(0,1,51,endpoint=False)
    yspace = xspace.copy()
    gridspace = np.meshgrid(xspace,xspace)
    x_grid = gridspace[0].reshape(51**2,1)
    y_grid = gridspace[1].reshape(51**2,1)

    x1_grid_train = np.tile(x_grid,(u_train_min.shape[0],1,1))
    x2_grid_train = np.tile(y_grid,(u_train_min.shape[0],1,1))
    x1_grid_test = np.tile(x_grid,(n_test,1,1))
    x2_grid_test = np.tile(y_grid,(n_test,1,1))

    space_train = np.concatenate([x2_grid_train,x1_grid_train], axis=-1)
    space_test = np.concatenate([x2_grid_test,x1_grid_test], axis=-1)

    # Stacking min and max
    u_train_int = np.stack([u_train_min, u_train_max], axis=-1)
    g_train_int = np.stack([g_train_min, g_train_max], axis=-1)

    u_test_int = np.stack([u_test_min, u_test_max], axis=-1)
    g_test_int = np.stack([g_test_min, g_test_max], axis=-1)

    u_val_int = np.stack([u_val_min, u_val_max], axis=-1)
    g_val_int = np.stack([g_val_min, g_val_max], axis=-1)

    return u_train_int, g_train_int, u_val_int, g_val_int, u_test_int, g_test_int, xspace, yspace

def data_formatter_nd_interval(u_data, g_data, sensor_coords, nsamp, eval_points=20):
    """
    Extended data formatter that supports upper/lower bounds in the last dimension.

    Parameters:
        u_data: ndarray, shape (nsamp, 1, ..., 2)
        g_data: ndarray, shape (nsamp, ..., 1, 2)
        sensor_coords: list of 1D arrays defining grid coordinates
        nsamp: number of samples to use
        eval_points: number of evaluation points per sample

    Returns:
        branch_tensor: (nsamp * eval_points, n_sensors_total, 2)
        trunk_tensor: (nsamp * eval_points, ndim)
        target_tensor: (nsamp * eval_points, 1, 2)
    """
    ndim = len(sensor_coords)
    sensor_grid = np.stack(np.meshgrid(*sensor_coords, indexing="ij"), axis=-1)  # (*sensor_shape, ndim)
    flat_coords = sensor_grid.reshape(-1, ndim)  # (n_sensors_total, ndim)
    n_sensors_total = flat_coords.shape[0]

    all_branch, all_trunk, all_target = [], [], []

    for i in range(nsamp):
        # Flatten both u and g but keep upper/lower as last dim
        u_i = u_data[i].reshape(-1, 2)  # (n_sensors_total, 2)
        g_i = g_data[i].reshape(-1, 2)  # (n_sensors_total, 2)

        # Random evaluation points
        indices = np.random.choice(n_sensors_total, size=eval_points, replace=False)

        for idx in indices:
            coord = flat_coords[idx]     # (ndim,)
            g_val = g_i[idx]             # (2,) — lower and upper values

            all_branch.append(u_i)       # entire input function with both bounds
            all_trunk.append(coord)
            all_target.append(g_val)     # both lower and upper bounds

    # Stack arrays into tensors
    branch_tensor = np.array(all_branch)   # (nsamp * eval_points, n_sensors_total, 2)
    trunk_tensor = np.array(all_trunk)     # (nsamp * eval_points, ndim)
    target_tensor = np.array(all_target)   # (nsamp * eval_points, 2)

    # Reshape target to (nsamp * eval_points, 1, 2)
    target_tensor = target_tensor.reshape(-1, 1, 2)

    return branch_tensor, trunk_tensor, target_tensor

def build_mlp(inputs, output_features: int, 
              hidden_features: int, num_hidden_layers: int):
    """
    Builds an MLP regression model.
    
    Parameters:
        x: Input tensor.
        output_features (int): Number of output features.
        hidden_features (int): Number of units in each hidden layer.
        num_hidden_layers (int): Number of hidden layers.
    
    Returns:
        array: keras tensor.
    """
    x = inputs
    
    # Create hidden layers with tanh activation
    for _ in range(num_hidden_layers):
        x = layers.Dense(hidden_features, activation='leaky_relu')(x)
    
    # Output layer with linear activation for regression
    outputs = layers.Dense(output_features, activation='linear')(x)
    
    return outputs

def build_imlp(inputs, output_features: int, 
              hidden_features: int, num_hidden_layers: int):
    """
    Builds an Interval MLP regression model.
    
    Parameters:
        x: Input tensor.
        output_features (int): Number of output features.
        hidden_features (int): Number of units in each hidden layer.
        num_hidden_layers (int): Number of hidden layers.
    
    Returns:
        array: keras tensor.
    """
    x = inputs
    
    # Create hidden layers with tanh activation
    for _ in range(num_hidden_layers):
        x = IntDense(hidden_features)(x)
        x = IntRelu()(x)
    
    # Output layer with linear activation for regression
    outputs = IntDense(output_features)(x)
    
    return outputs

def interval_deeponet(y_in_size, u_in_size, trunk_params, branch_params, out_feat=1):
    """_summary_

    Args:
        y_in_size (tuple): spatial input size
        u_in_size (tuple): function input size
        trunk_params (dict): trunk net parameters
        branch_params (dict): branch net parameters
        out_feat (int, optional): number of output feature. Defaults to 1.

    Returns:
        keras.Model: output model
    """
    y_input = layers.Input(y_in_size)
    u_input = layers.Input(u_in_size)

    trunk_out = build_mlp(y_input, trunk_params["output_features"], 
                          trunk_params["hidden_features"], trunk_params["num_hidden_layers"])
    
    # Branch layer
    u_in_lo = u_input[...,0]
    u_in_hi = u_input[...,1]
    u_in = [u_in_lo, u_in_hi]
    branch_out = build_imlp(u_in, branch_params["output_features"], 
                          branch_params["hidden_features"], branch_params["num_hidden_layers"])

    # Multiply trunk and branch
    mult_out = IntMultiply_donet()([trunk_out, branch_out])

    out = IntDense(out_feat)(mult_out)

    model = Model(inputs=[y_input, u_input], outputs=out, name="DeepONet")

    return model

def linex_loss(d,a):
    """Linear-exponential loss"""
    b = 2
    term1 = (tf.exp(-a*d)+(a*d)-1)
    return b*term1
    
def custom_loss(target, y_pred):
    term1 = tf.square((target[...,0] - y_pred[...,0]))
    term2 = tf.square((target[...,1] - y_pred[...,1]))
    loss = tf.reduce_mean(term1+term2)

    return loss


def scheduler(epoch):
    if epoch < 500:
        lr = 1e-3
    elif epoch < 2000:
        lr = 5e-4
    else:
        lr = 1e-4

    return lr


def run_experiments(trunk_params, branch_params, loss_fn, n_train=500, 
                    repetitions=5, epochs=100, batch=1024, verbose=0):
    
    PATH = "data/Darcy_Triangular_FNO_int.mat"

    n_val = 200
    n_test = 200

    results = []
    for seed in tqdm(range(repetitions)):
        print(f"###### Experiment {seed+1}/{repetitions} ######")
        # Preparing data
        data = dict()
        # Get data
        u_train_int, g_train_int, u_val_int, g_val_int, u_test_int, g_test_int, xspace, yspace = create_data(PATH, n_train=n_train,
                                                                                                     n_val=n_val, n_test=n_test, rand=seed)
        # Transform data shape
        u_train, space_train, g_train = data_formatter_nd_interval(u_train_int, g_train_int, [xspace,yspace], nsamp=u_train_int.shape[0], eval_points=200)
        u_val, space_val, g_val = data_formatter_nd_interval(u_val_int, g_val_int, [xspace,yspace], nsamp=u_val_int.shape[0], eval_points=100)

        # Compile model
        model = interval_deeponet(y_in_size=(2,), u_in_size=((51**2),2), trunk_params=trunk_params, 
                branch_params=branch_params, out_feat=1)
        model.compile(
            optimizer=keras.optimizers.legacy.Adam(learning_rate=1e-3),
            loss=custom_loss
        )

        # Train model
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
        t0 = default_timer()
        model, hist = trainer(model, space_train, u_train, g_train, 
                        y_val=space_val, u_val=u_val, 
                        g_val=g_val, loss_type="mse", scheduler=scheduler,
                        epochs=epochs, batch=64, verbose=verbose)
        data["training_time"] = default_timer() - t0
        print(f"Training time: {data['training_time']:.2f} seconds")

        # Evaluate on test set
        gridspace = np.meshgrid(xspace,xspace)
        x_grid = gridspace[0].reshape(51**2)
        y_grid = gridspace[1].reshape(51**2)
        sensor_data = np.stack([x_grid,y_grid],axis=0).T

        lb_pred_list = []
        ub_pred_list = []
        u_test_flat = u_test_int.reshape(n_test, 51**2, 2)

        t0=default_timer()
        for i in range(n_test):
            u_test_lo = u_test_flat[...,0]
            u_test_up = u_test_flat[...,1]
            u_lo = u_test_lo[i][None,:].repeat(2601,0)
            u_up = u_test_up[i][None,:].repeat(2601,0)

            u_int_test_in = np.stack([u_lo, u_up], axis=-1)
            pred_l, pred_u = model.predict([sensor_data, u_int_test_in])
            lb_pred_list.append(pred_l)
            ub_pred_list.append(pred_u)
        
        lb_pred = np.concatenate(lb_pred_list, axis=0).reshape(n_test,51,51).swapaxes(1, 2)
        ub_pred = np.concatenate(ub_pred_list, axis=0).reshape(n_test,51,51).swapaxes(1, 2)
        end_time = default_timer()

        data["inference_time"] = end_time - t0
        print(f"Inference time for {n_test} samples: {data['inference_time']:.2f} seconds")

        data["u_train"] = u_train_int
        data["y_train"] = xspace
        data["g_train"] = g_train_int
        data["u_test"] = u_test_int
        data["y_test"] = sensor_data
        data["g_test"] = g_test_int
        data["lb_pred"] = lb_pred
        data["ub_pred"] = ub_pred

        results.append(data)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Darcy 2D Naive Formatted")

    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Batch size for training"
    )
    parser.add_argument(
        "--n_train", type=int, default=10,
        help="Number of training data"
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--n_exp", type=int, default=2,
        help="Number of experiments"
    )

    args = parser.parse_args()

    trunk_params=dict()
    trunk_params["output_features"] = 128
    trunk_params["hidden_features"] = 128
    trunk_params["num_hidden_layers"] = 2

    branch_params=dict()
    branch_params["output_features"] = 128
    branch_params["hidden_features"] = 128
    branch_params["num_hidden_layers"] = 1

    results = run_experiments(trunk_params, branch_params, custom_loss, n_train=args.n_train, 
                                repetitions=args.n_exp, epochs=args.epochs, batch=args.batch_size, verbose=1)
    
    joblib.dump(results, f"output/ideal_naive_ntrain{args.n_train}_formatted_time.pkl")

