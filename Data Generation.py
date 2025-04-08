import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import random
import os

from tensorflow.keras.models import Model   # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.losses import categorical_crossentropy    # type: ignore
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AveragePooling2D, Activation, GlobalAveragePooling2D # type: ignore
from tensorflow.keras.initializers import HeUniform, HeNormal, Zeros, Constant  # type: ignore

from tensorflow.keras import datasets   # type: ignore
from tensorflow.keras.utils import to_categorical   # type: ignore

from tqdm import tqdm


# Number of models to train
number_of_models = 10


# Data preprocessing

# Load MNIST data
(x_train,y_train),(x_test,y_test) = datasets.mnist.load_data()

# Specify number of classes and examples to use (equal examples of each class)
K = 10
n = 300

# Randomly select n/K indices from each of the K first classes' indices
# Size of rest set is same size as the training set
selected_indices_train = []
selected_indices_test = []
for i in range(K):
  selected_indices_train = np.concatenate([selected_indices_train,
      np.random.choice(np.where(y_train == i)[0], int(n/10), replace=False)])
  selected_indices_test = np.concatenate([selected_indices_test,
      np.random.choice(np.where(y_test == i)[0], int(n/10), replace=False)])

# Make the training data randomly shuffled
random.shuffle(selected_indices_train)

x_train = x_train[selected_indices_train.astype(int)]
y_train = y_train[selected_indices_train.astype(int)]

x_test = x_test[selected_indices_test.astype(int)]
y_test = y_test[selected_indices_test.astype(int)]


# Normalise the greyscale values
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Add a new axis
x_train = x_train[:, :, :, np.newaxis]
x_test = x_test[:, :, :, np.newaxis]


# Convert class vectors to binary class matrices.
y_train = to_categorical(y_train, K)
y_test = to_categorical(y_test, K)


# Create the CNN
class SimpleCNN(Sequential):
    def __init__(self, nb_classes=2, use_bn=False):
        super().__init__()

        self.add(Conv2D(
            filters=6,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='valid',
            kernel_initializer=HeNormal(),
            bias_initializer='zeros',
            activation = 'relu'
            ))
        self.add(Conv2D(
            filters=10,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='valid',
            kernel_initializer=HeNormal(),
            bias_initializer='zeros',
            activation = 'relu'
            ))
        
        if use_bn:
            self.add(tf.keras.layers.BatchNormalization())

        self.add(GlobalAveragePooling2D())
        self.add(Flatten())
        self.add(Dense(
            nb_classes,
            kernel_initializer=HeNormal(),
            bias_initializer='zeros',
            activation='relu'
            ))
        self.add(Dense(
            nb_classes,
            kernel_initializer=HeNormal(),
            use_bias=False,
            activation='softmax'
            ))

def reinitialize_weights(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Layer) and layer.weights:
            # Reinitialize kernel weights
            if hasattr(layer, 'kernel_initializer'):
              initializer = type(layer.kernel_initializer)()
              layer.kernel.assign(initializer(layer.kernel.shape))
            # Reinitialize biases
            if hasattr(layer, 'bias_initializer') and layer.bias is not None:
              initializer = type(layer.bias_initializer)()
              layer.bias.assign(initializer(layer.bias.shape))


tf.config.run_functions_eagerly(True)


# Check if base Results directory exists, create it if not
if not os.path.exists('./Results/'):
    print("HAVE NOT FOUND RESULTS DIRECTORY")

# Path for the saved results
save_path = './Results/' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '/'
os.makedirs(save_path, exist_ok=True)

# Create a dataframe for the results of the models
# Creating the default values to align with the layout of the scaleGMN paper's results
default_vals = {
    'Unnamed: 0': 'path', 
    'config.activation': 'relu', 
    'config.b_init': 'zeros', 
    'config.dataset': 'cnn_zoo_new', 
    'config.dnn_architecture': 'cnn', 
    'config.dropout': np.float64(0), 
    'config.epochs': np.int64(86), 
    'config.epochs_between_checkpoints': np.int64(20), 
    'config.init_std': np.float64(1), 
    'config.l2reg': np.float64(0), 
    'config.learning_rate': np.float64(0), 
    'config.num_layers': np.int64(2), 
    'config.num_units': np.int64(6), 
    'config.optimizer': 'adam', 
    'config.random_seed': np.int64(0), 
    'config.train_fraction': np.float64(1.0), 
    'config.w_init': 'he_normal', 
    'modeldir': 'path', 
    'step': np.int64(86), 
    'test_accuracy': 0.1, 
    'test_loss': np.float64(2.308499574661255), 
    'train_accuracy': np.float64(0.0847092494368553), 
    'train_loss': np.float64(2.308099881889894), 
    'laststep': np.True_
}

# A helper function that merges defaults with provided data and adds the row to the DataFrame
def add_row(data, frame, defaults=default_vals):

    # Provided data will override the defaults.
    new_row = {**defaults, **data}

    new_row_df = pd.DataFrame([new_row])
    return pd.concat([frame, new_row_df], ignore_index=True)

# Create an empty DataFrame with the desired columns
metrics = pd.DataFrame(columns=default_vals.keys())

# Build and compile the CNN architecture
model = SimpleCNN(nb_classes=10)
input_shape = (None, 28, 28, 1)  # None is for the batch dimension
model.build(input_shape)
model.compile(optimizer='adam',
                    loss=categorical_crossentropy,
                    metrics=['accuracy'])
model.summary()

flat_param_list = []
for i in tqdm(range(number_of_models)):
    # Randomly reinitialise the weights of the model and train it for 30 epochs
    reinitialize_weights(model)
    model.fit(x_train, y=y_train,
                      batch_size=30,
                      epochs=30,
                      verbose=0)

    # Evaluate the model
    stats = model.evaluate(x_test, y_test, verbose=0)
    test_accuracy = stats[1]
    loss_value = stats[0]


    # Add a row where only a few columns are specified, rest will be defaults
    metrics = add_row({'test_accuracy': test_accuracy, 'test_loss': loss_value}, metrics)

    print(f"model_ id  {i}  ::  test accuracy  {test_accuracy}  :: loss  {loss_value}")

    # Get Vector of parameters and save in a list
    flat_params = tf.concat([tf.reshape(w, [-1]) for w in model.trainable_weights], axis=0)
    flat_param_list.append(flat_params)



# Stack all vectors into a matrix of shape (k, m)
parameter_matrix = tf.stack(flat_param_list, axis=0)

# Save as numpy array in drive
np_parameter_matrix = parameter_matrix.numpy()
np.save(save_path + '/weight.npy', np_parameter_matrix)
print("Weights saved.")


# Create a dataframe for the layout of the model parameters
layout = pd.DataFrame(columns=[
    'varname',
    'start_idx',
    'end_idx',
    'size',
    'shape'
])

start_idx = 0
for layer in model.layers:
    if layer.trainable_variables:
        for variable in layer.trainable_variables[::-1]:
             varname = f"sequential/{layer.name}/{variable.name}:0"
             shape = variable.shape
             size = np.prod(shape)
             end_idx = start_idx + size

             layout = pd.concat([layout, pd.DataFrame({
                 'varname': [varname],
                 'start_idx': [start_idx],
                 'end_idx': [end_idx],
                 'size': [size],
                 'shape': [str(shape)]
                 })
             ])
             start_idx = end_idx


# Save layout as csv to path
layout.to_csv(save_path + '/layout.csv', index=False)
print("Layout saved.")

# Save results as csv in drive
metrics.to_csv(save_path + '/metrics.csv', index=False)
print("Metrics saved.")

# Save the indexes file for the scaleGMN paper layout
cnn_zoo_new_splits = pd.DataFrame([i for i in range(number_of_models)])
cnn_zoo_new_splits.to_csv(save_path + '/cnn_zoo_new_splits.csv', index=False)
print("Splits saved.")

print("All results saved.")

