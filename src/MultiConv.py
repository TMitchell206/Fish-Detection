import tensorflow as tf
from fish_data_handler import *
from output_handler import *

# Parameters
testing = False
learning_rate = 0.0001
training_epochs = 30
batch_size = 16
display_step = 2

#Constants:
width = 32
height = 32
channels = 3
imgsize = [width,height]

n_input = width*height*channels
n_classes = 8
dropout = 0.5

print "Processing traning data..."
trainIMG, trainLBL = prepare_data(testing=testing)
trainIMG, trainLBL = shuffle_training_data(trainIMG, trainLBL)
num_train_egs = len(trainIMG)

print "Images loaded..."
print "Number images in training set:", num_train_egs

# Wrapper functions for Tensor operators
def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1,strides,strides,1], padding='VALID')
    x = tf.nn.bias_add(x,b)
    return tf.nn.relu(x)

def max_pool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1,k,k,1], strides=[1,k,k,1], padding='VALID')

def layer_output(x, W, b):
    x = tf.matmul(x, W)
    return tf.nn.bias_add(x, b)

def dense_relu(x, W, b):
    x = layer_output(x, W, b)
    return tf.nn.relu(x)

def dense_softmax(x, W, b):
    x = layer_output(x, W, b)
    return tf.nn.softmax(x)

def pad_layer(x, width=1, height=1):
    return tf.pad(x, [[0,0],[width,width],[height,height],[0,0]], 'CONSTANT')

def preprocess(x):
    return tf.reshape(x, shape=[-1, imgsize[0], imgsize[1], channels])

def conv_network(x, weights, biases, dropout):

    # Verbose code for easy reading of the network
    x = preprocess(x)
    x = pad_layer(x)

    # Double convolution layer 1
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = pad_layer(conv1)
    conv1 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv1 = max_pool2d(conv1)

    # Double convolution layer 2
    conv2 = pad_layer(conv1)
    conv2 = conv2d(conv2, weights['wc3'], biases['bc3'])
    conv2 = pad_layer(conv2)
    conv2 = conv2d(conv2, weights['wc4'], biases['bc4'])
    conv2 = max_pool2d(conv2)

    # Dense Layer
    dense = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    dense = dense_relu(dense, weights['wd1'], biases['bd1'])
    dense = tf.nn.dropout(dense, dropout)
    dense = dense_relu(dense, weights['wd2'], biases['bd2'])
    dense = tf.nn.dropout(dense, dropout)
    out = tf.nn.bias_add(tf.matmul(dense, weights['out'] ), biases['out'])
    out_sm = dense_softmax(dense, weights['out'], biases['out'])

    out_tensors = { 'raw_out': out, 'sm_out': out_sm}

    return out_tensors

# Tensorflow variables:
print "Tensorflow initializations..."
weights = {
    'wc1': tf.Variable(tf.random_normal([3,3,3,4], stddev=0.1)),
    'wc2': tf.Variable(tf.random_normal([3,3,4,4], stddev=0.1)),
    'wc3': tf.Variable(tf.random_normal([3,3,4,8], stddev=0.1)),
    'wc4': tf.Variable(tf.random_normal([3,3,8,8], stddev=0.1)),
    'wd1': tf.Variable(tf.random_normal([8*8*8,32], stddev=0.1)),
    'wd2': tf.Variable(tf.random_normal([32,32], stddev=0.1)),
    'out': tf.Variable(tf.random_normal([32,8], stddev=0.1)),
    }

biases = {
    'bc1': tf.Variable(tf.random_normal([4], stddev=0.1)),
    'bc2': tf.Variable(tf.random_normal([4], stddev=0.1)),
    'bc3': tf.Variable(tf.random_normal([8], stddev=0.1)),
    'bc4': tf.Variable(tf.random_normal([8], stddev=0.1)),
    'bd1': tf.Variable(tf.random_normal([32], stddev=0.1)),
    'bd2': tf.Variable(tf.random_normal([32], stddev=0.1)),
    'out': tf.Variable(tf.random_normal([8], stddev=0.1))
    }

x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)
WEIGHT_DECAY_FACTOR = 0.000001

# Build the network:
pred = conv_network(x, weights, biases, keep_prob)['raw_out']
class_vector = conv_network(x, weights, biases, keep_prob)['sm_out']

# Loss and Optimization:
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,y))
l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
cost = cost + WEIGHT_DECAY_FACTOR*l2_loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
print "Starting optimization..."
with tf.Session() as sess:
    sess.run(init)
    step = 1

    for epoch in range(training_epochs):

        num_batch = int(num_train_egs/batch_size)+1
        avg_cost = 0.
        trainIMG, trainLBL = shuffle_training_data(trainIMG, trainLBL)

        for i in range(num_batch):

            randidx = np.random.randint(num_train_egs, size=batch_size)
            batch_x = trainIMG[randidx, :]
            batch_y = trainLBL[randidx, :]

            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
            avg_cost += sess.run(cost, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})/num_batch

        if epoch % display_step == 0 or epoch == training_epochs:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
            print ("Epoch: %03d/%03d cost: %.9f" %(epoch, training_epochs, avg_cost))
            print ("   Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))

    print("Optimization Finished!")
    del trainIMG, trainLBL

    # Calculate accuracy for 256 mnist test images
    print("Running classification...")

    # Load test images:
    classIMG_paths, classIMG = load_classification_images(testing=testing)
    classifications = []

    for im in classIMG:
        im = im.reshape(1, len(im))
        classifications.append(sess.run(class_vector, feed_dict={x: im, keep_prob: 1.0}))
    print("Classification complete...")

    gen_output_csv(classifications, classIMG_paths)
    print("Submission file created...")

    sess.close()
print "Session closed, exiting..."
