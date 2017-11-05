# This is the main file where we train and test the models
import pandas as pd
import tensorflow as tf
import numpy as np

# Config options for the model etc
TRAIN_SIZE = 0.6

HIDDEN_NEURONS = 12

# Read in the csv
df = pd.read_csv('norm_iris.csv')

df = df.reindex(np.random.permutation(df.index))

petal_input = [[row['s_length'], row['s_width'], row['p_length'], row['p_width']] for _, row in df.iterrows()]
output = [[row['setosa'], row['versicolor'], row['virginica']] for _, row in df.iterrows()]

# Number of inputs
NUM_INPUTS = len(petal_input[0])
NUM_OUTPUTS = len(output[0])

# Training input is what we are going to train the model on
training_input = petal_input[:int(len(petal_input) * TRAIN_SIZE)]
# Test input is what we we use to test how well the model performs
test_input = petal_input[int(len(petal_input) * TRAIN_SIZE):]

# Likewise we will split the expected output in the same way
training_output = output[:int(len(petal_input) * TRAIN_SIZE)]
test_output = output[int(len(petal_input) * TRAIN_SIZE):]

input_data = training_input
output_data = training_output

# Set up the tf placeholders
input_x = tf.placeholder(dtype=np.float32, shape=[None, NUM_INPUTS], name='input_x')
output_x = tf.placeholder(dtype=np.float32, shape=[None, NUM_OUTPUTS], name='output_x')

# Setting up the hidden layers:
hidden_W = tf.Variable(tf.random_normal([NUM_INPUTS, HIDDEN_NEURONS]))
hidden_b = tf.Variable(tf.random_normal([HIDDEN_NEURONS]))

# Calculate the hidden layer values
hidden = tf.sigmoid(tf.matmul(input_x, hidden_W) + hidden_b)

# Calculate the output
output_W = tf.Variable(tf.random_normal([HIDDEN_NEURONS, NUM_OUTPUTS]))
output = tf.sigmoid(tf.matmul(hidden, output_W))

# Loss function:
cost = tf.reduce_mean(tf.square(output_x - output))
optimiser = tf.train.AdamOptimizer(0.01)
train = optimiser.minimize(cost)

# Init all the tf variables etc
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

# Train the data:

for i in range(10001):
    cvalues = sess.run([train, cost, hidden_W, output_x], feed_dict={input_x: input_data, output_x: output_data})

correct = 0

for j, val in enumerate(test_input):

    conf = sess.run(output, feed_dict={input_x: [val]}).tolist()
    guess = conf[0].index(max(conf[0]))
    correct_output = test_output[j].index(max(test_output[j]))

    if guess == correct_output:
        print('Guess: {} | Correct output: {}'.format(guess, correct_output))
        print("")
        correct = correct + 1
    else:
        print('Guess: {} | Correct output: {}'.format(guess, correct_output))
        print("")

print(correct/len(test_input))
