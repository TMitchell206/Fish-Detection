import tensorflow as tf
from fish_data_handler import *
from output_handler import *

testing = True
learning_rate = 0.01
training_epochs = 3000
batch_size = 16
display_step = 300

#Constants:
width = 32
height = 32
channels = 1
greyscale = True
imgsize = [width,height]

n_input = width*height*channels
n_hidden = 64
n_classes = 8
dropout = 0.5

print "Processing traning data..."
trainIMG, trainLBL = prepare_data(testing=testing, greyscale=greyscale)
trainIMG, trainLBL = shuffle_training_data(trainIMG, trainLBL)
num_train_egs = len(trainIMG)
print len(trainIMG[0])

print "Tensorflow initializations..."
weights = {
    'w1': tf.Variable(tf.random_normal([n_input, n_hidden], stddev=0.1)),
    'w2': tf.Variable(tf.random_normal([n_hidden, n_classes], stddev=0.1))
    }

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden], stddev=0.1)),
    'b2': tf.Variable(tf.random_normal([n_classes], stddev=0.1))
    }

def layer(x, W, b):
    x = tf.matmul(x, W)
    return tf.nn.bias_add(x, b)

def sigmoid_layer(x, W, b):
    y = layer(x, W, b)
    return tf.nn.sigmoid(y)

def feed_forward(x, weights, biases):

    hidden1 = sigmoid_layer(x, weights['w1'], biases['b1'])
    hidden2 = sigmoid_layer(hidden1, weights['w2'], biases['b2'])
    out_raw = hidden2
    out_sm = tf.nn.softmax(hidden2)
    out_tensors = {'out_raw': out_raw, 'out_sm': out_sm}
    return out_sm


x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

# Build the network:
pred = feed_forward(x, weights, biases)
#class_vector = conv_network(x, weights, biases, keep_prob)['out_sm']

# Loss and Optimization:
cost = tf.reduce_mean(tf.pow(y - pred,2))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

print "Starting optimization..."
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):

        num_batch = int(num_train_egs/batch_size)+1
        avg_cost = 0.
        avg_acc = 0.
        trainIMG, trainLBL = shuffle_training_data(trainIMG, trainLBL)
        index = 0

        for i in range(num_batch):

            batch_x, batch_y, index = gen_batch(trainIMG, trainLBL, batch_size, index)

            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            avg_cost += sess.run(cost, feed_dict={x: batch_x, y: batch_y})/num_batch
            avg_acc += sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})/num_batch

        if epoch % display_step == 0 or epoch == training_epochs-1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})

            print ("Epoch: %03d/%03d cost: %.9f" %(epoch, training_epochs, avg_cost))
            print ("   Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(avg_acc))

    print("Optimization Finished!")
    del trainIMG, trainLBL

    # Calculate accuracy for 256 mnist test images
    print("Running classification...")

    # Load test images:
    classIMG_paths, classIMG = load_classification_images(testing=testing, greyscale=greyscale)
    classifications = []

    for im in classIMG:
        im = im.reshape(1, len(im))
        classifications.append(sess.run(pred, feed_dict={x: im}))
    print("Classification complete...")

    gen_output_csv(classifications, classIMG_paths)
    print("Submission file created...")

    sess.close()
print "Session closed, exiting..."
