# This is the main file where we train and test the ys
import pandas as pd
import tensorflow as tf
import numpy as np
from timeit import default_timer as timer

# Config options for the y etc
TRAIN_SIZE = 0.6

HIDDEN_NEURONS = 3

# Read in the csv
df = pd.read_csv('norm_iris.csv')

df = df.reindex(np.random.permutation(df.index))

petal_input = [[row['s_length'], row['s_width'], row['p_length'], row['p_width']] for _, row in df.iterrows()]
output = [[row['setosa'], row['versicolor'], row['virginica']] for _, row in df.iterrows()]

# Number of inputs
NUM_INPUTS = len(petal_input[0])
NUM_OUTPUTS = len(output[0])

# Training input is what we are going to train the y on
training_input = petal_input[:int(len(petal_input) * TRAIN_SIZE)]
# Test input is what we we use to test how well the y performs
test_input = petal_input[int(len(petal_input) * TRAIN_SIZE):]

# Likewise we will split the expected output in the same way
training_output = output[:int(len(petal_input) * TRAIN_SIZE)]
test_output = output[int(len(petal_input) * TRAIN_SIZE):]

input_data = training_input
output_data = training_output

# Set up the tf placeholders
x = tf.placeholder(dtype=np.float32, shape=[None, NUM_INPUTS], name='x')
y_ = tf.placeholder(dtype=np.float32, shape=[None, NUM_OUTPUTS], name='y_')

# Setting up the hidden layers:
W = tf.Variable(tf.zeros([NUM_INPUTS, HIDDEN_NEURONS]))
b = tf.Variable(tf.zeros([HIDDEN_NEURONS]))

y = tf.nn.softmax(tf.matmul(x, W) + b)  # Soft max

# Loss function:
cost_function = -tf.reduce_sum(y_ * tf.log(y))
optimiser = tf.train.AdamOptimizer(0.01).minimize(cost_function)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Init all the tf variables etc
init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
sess.run(init)

# Train the data:

data = []
data_2 = []

start = timer()

for i in range(5000):
    cvalues = sess.run([optimiser, cost_function], feed_dict={x: training_input, y_: training_output})

    if i % 100:
        print(cvalues[1])

end = timer()

print("Time taken to train: {}".format(end-start))

correct = 0

print("")
print(sess.run(accuracy, feed_dict={x: test_input, y_: test_output}))
