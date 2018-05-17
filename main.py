import matplotlib.pyplot as plt
import pandas
import math
import numpy
from keras.utils import np_utils

import tensorflow as tf

import categorizer

numpy.random.seed(7)

train_input_file = 'train.csv'
# Columnas
#     0          1       2      3   4   5    6     7     8      9   10      11
# PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
train_df = pandas.read_csv(train_input_file, usecols=[1, 2, 4, 5, 11])
train_ds = train_df.values
COL_COUNT = train_ds.shape[1]

SURVIVED_COL, PCLASS_COL, SEX_COL, AGE_COL, EMBARK_COL = range(COL_COUNT)

a = train_ds[:,AGE_COL].astype('float32')
b = numpy.isnan(a)
c = b == False
mean_age = a[c].mean()


train_ds = categorizer.replace_nans(train_ds , AGE_COL , mean_age)
train_ds , SEX_CATEGORIES = categorizer.categorize_col(train_ds , SEX_COL)
train_ds , EMBARK_CATEGORIES = categorizer.categorize_col(train_ds , EMBARK_COL)


# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 1000
display_step = 1

# Usaremos dos capas ocultas de 256 neuronas cada una
n_hidden_1 = 256
n_hidden_2 = 256

train_in_ds = mnist.train.images  # dataset de entrnamiento de entrada
train_out_ds = mnist.train.labels  # dataset de entrenamiento de salida o clases
ds_size = len(train_in_ds)

test_in_ds = mnist.test.images  # dataset de pruebas de entrada
test_out_ds = mnist.test.labels  # dataset de pruebas de salida

# La entrada son imagenes de 28*28 pixeles (tensor de 28*28 elementos)
n_input = train_in_ds.shape[1]  # es decir, que la capa de entrada tendra 784 neuronas
n_classes = train_out_ds.shape[1]  # cantidad de clases. En este caso son 10 digitos del 0 al 9


def build_random_weight_tensor(in_size, out_size):
    """ Construye un tensor de in_size * out_size elementos con valores aleatorios siguiendo la distribucion normal.
    Cada valor representa el peso de la entrada 'i' hacia la neurona 'j' (siendo i una fila y j una columna) """
    return tf.random_normal(shape=[in_size, out_size])


def build_random_bias_tensor(out_size):
    """ Crea un tensor de umbrales de activacion para una capa de la red neuronal """
    return tf.random_normal(shape=[out_size])


# NOTAR QUE LOS PESOS Y BIASES SON VARIABLES
# Esto se debe a que las variables son los unicos componentes del modelo que pueden ser reasignados y por lo tanto, optimizados

# Store layers weight & bias
# creo tres tensores. Cada tensor representa los pesos de la entrada para cada neurona
# hidden_1 representa los pesos de las neuronas de entrada con la primera capa oculta (hidden_1)
# hidden_2 representa los pesos de las salidas de hidden_1 con la segunda capa
# hidden_3 representa los pesos de las salidas de hidden_2 con la salida
weights = {
    'hidden_1': tf.Variable(build_random_weight_tensor(n_input, n_hidden_1)),
    'hidden_2': tf.Variable(build_random_weight_tensor(n_hidden_1, n_hidden_2)),
    'out': tf.Variable(build_random_weight_tensor(n_hidden_2, n_classes))
}

# defino los umbrales de activacion
biases = {
    'bias_1': tf.Variable(build_random_bias_tensor(n_hidden_1)),
    'bias_2': tf.Variable(build_random_bias_tensor(n_hidden_2)),
    'out': tf.Variable(build_random_bias_tensor(n_classes))
}


# Create model
def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    # Cada capa resuelve: x * w + b
    layer_1 = tf.matmul(x, weights['hidden_1']) + biases['bias_1']
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.matmul(layer_1, weights['hidden_2']) + biases['bias_2']
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


# El placeholder X sera un tensor de dimensiones [? , n_input]
# El placeholder Y sera un tensor de dimensiones [? , n_classes]
# Los placeholders permiten inyectar valores al modelo
# Suelen usarse para inyectar los valores de entrada y salida para el entrenamiento
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# Construct model
logits = multilayer_perceptron(X)

# Defino la funcion de perdida y el optimizador que debe minimizar dicha funcion
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Esta funcion define un step u operacion de tensorflow que inicializara variables (los tf.Variable)
init_op = tf.global_variables_initializer()


def next_batch(input_ds, output_ds, batch_idx):
    start_idx = batch_idx * batch_size
    end_idx = start_idx + batch_size
    return (input_ds[start_idx:end_idx], output_ds[start_idx:end_idx])


sess = tf.Session()

sess.run(init_op)
# Training cycle
for epoch in range(training_epochs):
    avg_cost = 0.0
    batch_count = int(ds_size / batch_size)
    # Loop over all batches
    for batch_idx in range(batch_count):
        batch_x, batch_y = next_batch(train_in_ds, train_out_ds, batch_idx)
        #batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop) and cost op (to get loss value)
        _, cost = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
        # Compute average loss
        avg_cost += cost / batch_count
    # Display logs per epoch step
    if (epoch % display_step == 0):
        print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(avg_cost))

print("Optimization Finished!")

# Test model
pred = tf.nn.softmax(logits)  # Apply softmax to logits
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
acc_value = sess.run(accuracy, feed_dict={X: test_in_ds, Y: test_out_ds})
print("Accuracy:", acc_value)

sess.close()