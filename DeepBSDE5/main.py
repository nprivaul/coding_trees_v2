"""
The main file to run BSDE solver to solve parabolic partial differential equations (PDEs).

"""
import logging

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')

if gpus:
    # choose GPU index safely
    desired_index = 2  # your preferred GPU

    gpu_index = desired_index if len(gpus) > desired_index else 0

    tf.config.set_visible_devices(gpus[gpu_index], 'GPU')

    tf_device = tf.config.list_physical_devices('GPU')[0]
    tf_name = tf.config.experimental.get_device_details(tf_device).get('device_name', 'Unknown')

    print(f"TensorFlow is using: {tf_name} (GPU {gpu_index})")
else:
    print("No GPU detected — using CPU")

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from equation import AllenCahn
from solver import BSDESolver


class Dict2Class(object):
    """ Turns dict into a class """
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])

def main(
    total_time=0.3,
    dim=5,
    nu=1,
    num_time_interval=20,
#    y_init_range=[0,1],
    y_init_range=[-1, 0],
    num_hiddens=[110, 110],
    lr_values=[5e-4, 5e-4],
    lr_boundaries=[2000],
    num_iterations= 2000,
    batch_size= 64,
    valid_size= 256,
    logging_frequency=100,
    dtype="float64",
    verbose=True,
):

    eqn_config = Dict2Class({
        "_comment": "Allen-Cahn equation",
        "eqn_name": "Allen-Cahn",
        "total_time": total_time,
        "dim": dim,
        "nu": nu,
        "num_time_interval": num_time_interval,
    })
    net_config = Dict2Class({
        "y_init_range": y_init_range,
        "num_hiddens": num_hiddens,
        "lr_values": lr_values,
        "lr_boundaries": lr_boundaries,
        "num_iterations": num_iterations,
        "batch_size": batch_size,
        "valid_size": valid_size,
        "logging_frequency": logging_frequency,
        "dtype": dtype,
        "verbose": verbose,
    })
    config = Dict2Class({
        "eqn_config": eqn_config,
        "net_config": net_config,
    })

    bsde = AllenCahn(eqn_config)
    tf.keras.backend.set_floatx(net_config.dtype)

    formatter = "%(asctime)s | %(name)s |  %(levelname)s: %(message)s"
    logging.basicConfig(
        level=logging.DEBUG,
        format=formatter,
    )
    logging.info('Begin to solve %s ' % eqn_config.eqn_name)
    bsde_solver = BSDESolver(config, bsde)
    training_history = bsde_solver.train()
    if bsde.y_init:
        logging.info('Y0_true: %.4e' % bsde.y_init)
        logging.info('relative error of Y0: %s',
                     '{:.2%}'.format(abs(bsde.y_init - training_history[-1, 2])/bsde.y_init))
    return training_history


if __name__ == '__main__':
    _ = main()
