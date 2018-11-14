import tensorflow as tf
import input_data_labeling
import math
import os
import sys
import csv
import numpy
import pandas
from tqdm import tqdm

#%%
#layer_sizes = [784, 1000, 500, 250, 250, 250, 10]
#layer_sizes = [128, 1024, 512, 256, 256, 10]
# 63%: layer_sizes = [128,100, 25, 25, 10], starter_learning_rate = 0.06, decay_after = 30
# 69%: layer_sizes = [128,64, 32, 32, 10], starter_learning_rate = 0.06, decay_after = 30
# 24%: layer_sizes = [128, 1, 10]

# Default values
layer_sizes =[128, 2000, 2000, 2000, 10]
check_path = "checkptNew"
num_epochs = 150
starter_learning_rate = 0.03#0.02
decay_after = 30#15  # epoch after which to begin learning rate decay
denoising_cost = 0.1 * numpy.ones(len(layer_sizes))
denoising_cost[0] *= 10000
denoising_cost[1] *= 100

narg = len(sys.argv)
if narg < 7:
    print("Usage: my_ladder.py <checkpath> <layersizes> <num epochs> <learning rate> <decay after> <denoising cost factor>")
    print("Use given input and set rest to default.")
if narg > 1:
    check_path = sys.argv[1]
if narg > 2:
    lay_s =sys.argv[2]
    l = lay_s.strip("[]")
    layer_sizes = list(int(a) for a in l.split(","))
    denoising_cost = 0.1*numpy.ones(len(layer_sizes))
if narg > 3: 
    num_epochs = int(sys.argv[3])
if narg > 4:
    starter_learning_rate = float(sys.argv[4])
if narg > 5:
    decay_after = int(sys.argv[5])
if narg > 6:
    denoising_cost[0] *= 10000
    denoising_cost[1] *= 100
    denoising_cost *= float(sys.argv[6])

# Status 
print(3*"\n* * * * *")
print("Checkpoint path =", check_path)
print("Layer sizes =", layer_sizes)
print("Num epochs =", num_epochs)
print("Init learning rate =", starter_learning_rate)
print("Decay learning rate after", decay_after, "epochs")
print("Denoising cost", denoising_cost)
print(3*"* * * * *\n")


L = len(layer_sizes) - 1  # number of layers

testlen = 100

num_examples = 21000 + 9000 - testlen #unlabeled+labeled-test
num_labeled = 30000 - testlen#9000 - testlen



batch_size = 100
num_iter = int((num_examples/batch_size)) * num_epochs  # number of loop iterations

inputs = tf.placeholder(tf.float32, shape=(None, layer_sizes[0]))
outputs = tf.placeholder(tf.float32)


def bi(inits, size, name):
    return tf.Variable(inits * tf.ones([size]), name=name)


def wi(shape, name):
    return tf.Variable(tf.random_normal(shape, name=name)) / math.sqrt(shape[0])

# python3 zip is an iterator -> can only be read once
#shapes = zip(layer_sizes[:-1], layer_sizes[1:])  # shapes of linear layers
shapes = list(zip(layer_sizes[:-1], layer_sizes[1:]))  # shapes of linear layers

weights = {'W': [wi(s, "W") for s in shapes],  # Encoder weights
           'V': [wi(s[::-1], "V") for s in shapes],  # Decoder weights
           # batch normalization parameter to shift the normalized value
           'beta': [bi(0.0, layer_sizes[l+1], "beta") for l in range(L)],
           # batch normalization parameter to scale the normalized value
           'gamma': [bi(1.0, layer_sizes[l+1], "beta") for l in range(L)]}

noise_std = 0.3  # scaling factor for noise used in corrupted encoder

# hyperparameters that denote the importance of each layer
#denoising_cost = [100000.0, 100.0, 0.10, 0.10, 0.10, 0.10, 0.10]

join = lambda l, u: tf.concat([l, u], 0)
labeled = lambda x: tf.slice(x, [0, 0], [batch_size, -1]) if x is not None else x
unlabeled = lambda x: tf.slice(x, [batch_size, 0], [-1, -1]) if x is not None else x
split_lu = lambda x: (labeled(x), unlabeled(x))

training = tf.placeholder(tf.bool)

ewma = tf.train.ExponentialMovingAverage(decay=0.99)  # to calculate the moving averages of mean and variance
bn_assigns = []  # this list stores the updates to be made to average mean and variance


def batch_normalization(batch, mean=None, var=None):
    if mean is None or var is None:
        mean, var = tf.nn.moments(batch, axes=[0])
    return (batch - mean) / tf.sqrt(var + tf.constant(1e-10))

# average mean and variance of all layers
running_mean = [tf.Variable(tf.constant(0.0, shape=[l]), trainable=False) for l in layer_sizes[1:]]
running_var = [tf.Variable(tf.constant(1.0, shape=[l]), trainable=False) for l in layer_sizes[1:]]


def update_batch_normalization(batch, l):
    "batch normalize + update average mean and variance of layer l"
    mean, var = tf.nn.moments(batch, axes=[0])
    assign_mean = running_mean[l-1].assign(mean)
    assign_var = running_var[l-1].assign(var)
    bn_assigns.append(ewma.apply([running_mean[l-1], running_var[l-1]]))
    with tf.control_dependencies([assign_mean, assign_var]):
        return (batch - mean) / tf.sqrt(var + 1e-10)


def encoder(inputs, noise_std):
    h = inputs + tf.random_normal(tf.shape(inputs)) * noise_std  # add noise to input
    d = {}  # to store the pre-activation, activation, mean and variance for each layer
    # The data for labeled and unlabeled examples are stored separately
    d['labeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
    d['unlabeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
    d['labeled']['z'][0], d['unlabeled']['z'][0] = split_lu(h)
    for l in range(1, L+1):
        print("Layer ", str(l), ": ", str(layer_sizes[l-1]), " -> ", str(layer_sizes[l]))
        d['labeled']['h'][l-1], d['unlabeled']['h'][l-1] = split_lu(h)
        z_pre = tf.matmul(h, weights['W'][l-1])  # pre-activation
        z_pre_l, z_pre_u = split_lu(z_pre)  # split labeled and unlabeled examples

        m, v = tf.nn.moments(z_pre_u, axes=[0])

        # if training:
        def training_batch_norm():
            # Training batch normalization
            # batch normalization for labeled and unlabeled examples is performed separately
            if noise_std > 0:
                # Corrupted encoder
                # batch normalization + noise
                z = join(batch_normalization(z_pre_l), batch_normalization(z_pre_u, m, v))
                z += tf.random_normal(tf.shape(z_pre)) * noise_std
            else:
                # Clean encoder
                # batch normalization + update the average mean and variance using batch mean and variance of labeled examples
                z = join(update_batch_normalization(z_pre_l, l), batch_normalization(z_pre_u, m, v))
            return z

        # else:
        def eval_batch_norm():
            # Evaluation batch normalization
            # obtain average mean and variance and use it to normalize the batch
            mean = ewma.average(running_mean[l-1])
            var = ewma.average(running_var[l-1])
            z = batch_normalization(z_pre, mean, var)
            # Instead of the above statement, the use of the following 2 statements containing a typo
            # consistently produces a 0.2% higher accuracy for unclear reasons.
            # m_l, v_l = tf.nn.moments(z_pre_l, axes=[0])
            # z = join(batch_normalization(z_pre_l, m_l, mean, var), batch_normalization(z_pre_u, mean, var))
            return z

        # perform batch normalization according to value of boolean "training" placeholder:
        z = tf.cond(training, training_batch_norm, eval_batch_norm)

        if l == L:
            # use softmax activation in output layer
            h = tf.nn.softmax(weights['gamma'][l-1] * (z + weights["beta"][l-1]))
        else:
            # use ReLU activation in hidden layers
            h = tf.nn.relu(z + weights["beta"][l-1])
        d['labeled']['z'][l], d['unlabeled']['z'][l] = split_lu(z)
        d['unlabeled']['m'][l], d['unlabeled']['v'][l] = m, v  # save mean and variance of unlabeled examples for decoding
    d['labeled']['h'][l], d['unlabeled']['h'][l] = split_lu(h)
    return h, d

print("=== Corrupted Encoder ===")
y_c, corr = encoder(inputs, noise_std)

print("=== Clean Encoder ===")
y, clean = encoder(inputs, 0.0)  # 0.0 -> do not add noise

print("=== Decoder ===")


def g_gauss(z_c, u, size):
    "gaussian denoising function proposed in the original paper"
    wi = lambda inits, name: tf.Variable(inits * tf.ones([size]), name=name)
    a1 = wi(0., 'a1')
    a2 = wi(1., 'a2')
    a3 = wi(0., 'a3')
    a4 = wi(0., 'a4')
    a5 = wi(0., 'a5')

    a6 = wi(0., 'a6')
    a7 = wi(1., 'a7')
    a8 = wi(0., 'a8')
    a9 = wi(0., 'a9')
    a10 = wi(0., 'a10')

    mu = a1 * tf.sigmoid(a2 * u + a3) + a4 * u + a5
    v = a6 * tf.sigmoid(a7 * u + a8) + a9 * u + a10

    z_est = (z_c - mu) * v + mu
    return z_est

# Decoder
z_est = {}
d_cost = []  # to store the denoising cost of all layers

for l in range(L, -1, -1):
    print("Layer ", str(l), ": ", str(layer_sizes[l+1]) if l+1 < len(layer_sizes) else None, " -> ", str(layer_sizes[l]), ", denoising cost: ", str(denoising_cost[l]))
    z, z_c = clean['unlabeled']['z'][l], corr['unlabeled']['z'][l]
    m, v = clean['unlabeled']['m'].get(l, 0), clean['unlabeled']['v'].get(l, 1-1e-10)
    if l == L:
        u = unlabeled(y_c)
    else:
        u = tf.matmul(z_est[l+1], weights['V'][l])
    u = batch_normalization(u)
    z_est[l] = g_gauss(z_c, u, layer_sizes[l])
    z_est_bn = (z_est[l] - m) / v
    # append the cost of this layer to d_cost
    d_cost.append((tf.reduce_mean(tf.reduce_sum(tf.square(z_est_bn - z), 1)) / layer_sizes[l]) * denoising_cost[l])

# calculate total unsupervised cost by adding the denoising cost of all layers
u_cost = tf.add_n(d_cost)

y_N = labeled(y_c)
cost = -tf.reduce_mean(tf.reduce_sum(outputs*tf.log(y_N), 1))  # supervised cost
loss = cost + u_cost  # total cost

pred_cost = -tf.reduce_mean(tf.reduce_sum(outputs*tf.log(y), 1))  # cost used for prediction

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(outputs, 1))  # no of correct predictions
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) * tf.constant(100.0)

learning_rate = tf.Variable(starter_learning_rate, trainable=False)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# add the updates of batch normalization statistics to train_step
bn_updates = tf.group(*bn_assigns)
with tf.control_dependencies([train_step]):
    train_step = tf.group(bn_updates)

#%%
print("===  Loading Data ===")
mnist = input_data_labeling.read_data_sets("data/", n_labeled=num_labeled, one_hot=True)

#%%
saver = tf.train.Saver()

print("===  Starting Session ===")
sess = tf.Session()

i_iter = 0

#file_path = 'C:/Users/jonas/OneDrive/ETH/Machine Learning/iml-projects/task4/ladder-master/checkpoints/'
ckpt = tf.train.get_checkpoint_state(check_path + "/")  # get latest checkpoint (if any)
if ckpt and ckpt.model_checkpoint_path:
    # if checkpoint exists, restore the parameters and set epoch_n and i_iter
    saver.restore(sess, ckpt.model_checkpoint_path)
    epoch_n = int(ckpt.model_checkpoint_path.split('-')[-1])
    i_iter = int((epoch_n+1) * int(num_examples/batch_size))
    print("Restored Epoch ", str(epoch_n))
else:
    # no checkpoint exists. create checkpoints directory if it does not exist.
    if not os.path.exists(check_path):
        os.makedirs(check_path)
    init = tf.global_variables_initializer()
    sess.run(init)


print("=== Training ===")
print("Initial Accuracy: ", sess.run(accuracy, feed_dict={inputs: mnist.test.images, outputs: mnist.test.labels, training: False}), "%")

#feed_dict = {} # want to use this for eval
for i in tqdm(range(int(i_iter), int(num_iter))):
    images, labels = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={inputs: images, outputs: labels, training: True})
    if (i > 1) and ((i+1) % int(num_iter/num_epochs) == 0):
        epoch_n = int(i/int(num_examples/batch_size))
        if (epoch_n+1) >= decay_after:
            # decay learning rate
            # learning_rate = starter_learning_rate * ((num_epochs - epoch_n) / (num_epochs - decay_after))
            ratio = 1.0 * (num_epochs - (epoch_n+1))  # epoch_n + 1 because learning rate is set for next epoch
            ratio = max(0, ratio / (num_epochs - decay_after))
            sess.run(learning_rate.assign(starter_learning_rate * ratio))
        saver.save(sess, check_path + '/model.ckpt', epoch_n)
        # print "Epoch ", epoch_n, ", Accuracy: ", sess.run(accuracy, feed_dict={inputs: mnist.test.images, outputs:mnist.test.labels, training: False}), "%"
        ##with open('train_log', 'ab') as train_log:

        with open('train_log', 'at', newline='') as train_log: # changed type from 'ab' to 'at' (string instead of bytes for python3)
            # write test accuracy to file "train_log"
            train_log_w = csv.writer(train_log)
            log_i = [epoch_n] + sess.run([accuracy], feed_dict={inputs: mnist.test.images, outputs: mnist.test.labels, training: False})
            train_log_w.writerow(log_i)

print("Final Accuracy: ", sess.run(accuracy, feed_dict={inputs: mnist.test.images, outputs: mnist.test.labels, training: False}), "%")

sess.close()
 #%%
##################################
#
# Eval
#
##################################

#print(mnist.test.images)
sess = tf.Session()
ckpt = tf.train.get_checkpoint_state(check_path + "/")  # get latest checkpoint (if any)
saver.restore(sess, ckpt.model_checkpoint_path)
epoch_n = int(ckpt.model_checkpoint_path.split('-')[-1])
i_iter = int((epoch_n+1) * int(num_examples/batch_size))
print("Restored Epoch ", str(epoch_n))
print("=== Predicting ===")
#%%
pred = sess.run(tf.argmax(y, 1),feed_dict={inputs: mnist.evalu.images, training: False}) #feed_dict={inputs: mnist.test.images[:,0,:,0], training: False})
#%%
idx = numpy.arange(len(pred)) + 30000
pred_DF = pandas.DataFrame(data=pred, columns=['y'], index=idx)
pred_DF.to_csv("pred_unl_data_labels"+str(check_path[-2:])+str(layer_sizes).strip()+".csv",index_label=["Id"]) #,index=False)
#%%
sess.close()
