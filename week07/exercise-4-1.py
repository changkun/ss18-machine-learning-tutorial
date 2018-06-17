import h5py
import numpy as np
import tensorflow as tf
np.random.seed(42)

def load_data():
    # in case the file is stored not in the same folder as this notebook, please adjust the path
    h5f = h5py.File('dataset.h5','r+')
    X_train = h5f['X_train'][:]
    X_test = h5f['X_test'][:]
    h5f.close()    

    # in case the file is stored not in the same folder as this notebook, please adjust the path
    h5f = h5py.File('labels.h5','r+')
    y_train = h5f['y_train'][:]
    y_test = h5f['y_test'][:]
    h5f.close()  
    
    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = load_data()


map_characters= {
    0: 'abraham_grampa_simpson',
    1: 'apu_nahasapeemapetilon',
    2: 'bart_simpson',
    3: 'charles_montgomery_burns',
    4: 'chief_wiggum',
    5: 'comic_book_guy',
    6: 'edna_krabappel',
    7: 'homer_simpson',
    8: 'kent_brockman',
    9: 'krusty_the_clown',
    10:'lisa_simpson',
    11:'marge_simpson',
    12:'milhouse_van_houten',
    13:'moe_szyslak',
    14:'ned_flanders',
    15:'nelson_muntz',
    16:'principal_skinner',
    17:'sideshow_bob'
}
pic_size = 64

### Set neural network hyperparameters
epochs = 20
batch_size = 128
wt_init = tf.contrib.layers.xavier_initializer() # weight initializer

# input layer (64x64)
# n_input = 4096
n_input = 64*64

# first convolutional layer
n_conv_1 = 32
k_conv_1 = 3

# second convolutional layer
n_conv_2 = 64
k_conv_2 = 3

# max pooling layer:
pool_size = 2
mp_layer_dropout = .25

# dense layer:
n_dense = 128
dense_layer_dropout = .5

# output layer:
n_classes = len(map_characters)



# definition of x
x = tf.placeholder(tf.float32, [None, pic_size, pic_size, 3])
# definition of y
y = tf.placeholder(tf.float32, [None, n_classes])

# dense layer with ReLU activation:
def dense (x, W, b):
    return tf.nn.relu(tf.add(tf.matmul(x, W), b))

# convolutional layer with ReLU activation:
def conv2d(x, W, b, stride_length=1):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, W, strides=[1, stride_length, stride_length, 1], padding='SAME'), b))

# max-pooling layer:
def maxpooling2d(x, p_size):
    return tf.nn.max_pool(x, ksize=[1, p_size, p_size, 1], strides=[1, p_size, p_size, 1], padding='SAME')

def net(x, weights, biases, n_in, mp_psize, mp_dropout, dense_dropout):
    # first convolutional layer
    with tf.name_scope("conv1"):
        conv_1 = conv2d(x, weights['W_c1'], biases['b_c1'])
    
    # second convolutional layer
    with tf.name_scope("conv2"):
        conv_2 = conv2d(conv_1, weights['W_c2'], biases['b_c2'])

    # maxpool layer
    with tf.name_scope("maxpool1"):
        pool_1 = maxpooling2d(conv_2, mp_psize)

    # dropout layer
    with tf.name_scope("dropout1"):
        dropout_1 = tf.nn.dropout(pool_1, 1 - mp_dropout)

    # dense layer (first we have to flatten out the output of the previous layer)
    with tf.name_scope("dense1"):
        flat_1 = tf.reshape(dropout_1, [-1, weights['W_d1'].get_shape().as_list()[0]])
        dense_1 = dense(flat_1, weights['W_d1'], biases['b_d1'])
    
    # dropout layer
    with tf.name_scope("dropout2"):
        dropout_2 = tf.nn.dropout(dense_1, 1 - dense_dropout)
    
    # output layer
    with tf.name_scope("output"):
        return tf.add(tf.matmul(dropout_2, weights['W_out']), biases['b_out'])

# definition of dict for biases
bias_dict = {
    'b_c1': tf.Variable(tf.zeros([n_conv_1])),
    'b_c2': tf.Variable(tf.zeros([n_conv_2])),
    'b_d1': tf.Variable(tf.zeros([n_dense])),
    'b_out': tf.Variable(tf.zeros([n_classes]))
}

# calculate number of inputs to dense layer:
full_square_length = np.sqrt(n_input)
pooled_square_length = int(full_square_length / pool_size)
dense_inputs = pooled_square_length ** 2 * n_conv_2

# definition of dict for weights
weight_dict = {
    'W_c1': tf.get_variable('W_c1', [k_conv_1, k_conv_1, 3, n_conv_1], initializer=wt_init),
    'W_c2': tf.get_variable('W_c2', [k_conv_2, k_conv_2, n_conv_1, n_conv_2], initializer=wt_init),
    'W_d1': tf.get_variable('W_d1', [dense_inputs, n_dense], initializer=wt_init), 
    'W_out': tf.get_variable('W_out', [n_dense, n_classes],  initializer=wt_init)
}

# definition for our predictions
predictions = net(x, weight_dict, bias_dict, n_input, pool_size, mp_layer_dropout, dense_layer_dropout)

# definition of the cost function
cost = tf.reduce_mean (tf.nn.softmax_cross_entropy_with_logits_v2(logits=predictions, labels=y))

# defintion of the optimizer
optimizer = tf.train.AdamOptimizer().minimize(cost)

# defintion of accuracy (in percentage)
correct_pred = tf.equal(tf.argmax(predictions, 1), tf.argmax(y,1))
acc_pct = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) * 100

# definition of initializer
init_op = tf.global_variables_initializer()

tf.summary.scalar("cost", cost)
tf.summary.scalar("accuracy_percentage", acc_pct)

def next_batch(num, data, labels):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


with tf.Session() as session:
    # run initializer
    session.run(init_op)

    # Define FileWriter for logs
    train_writer = tf.summary.FileWriter( './logs/1/train ', session.graph)
    
    # iterate epochs and batches and run statement for TensorFlow calculations
    # use  merge = tf.summary.merge_all() to get values for tf.summaries defined aboove
    # and attach it to param list in session.run() to actually retrieve the results
    # with train_writer.add_summary(summary, cnt) we can write for each batch within 
    # an epoch the results to the log files. Note that the counter cnt is an indicator
    # for the log-id
    cnt = 0
    for epoch in range(100):
        avg_cost = avg_acc_pct = 0.0        
        n_batches = int(len(X_train) / batch_size)
        for i in range(n_batches):
            batch_x , batch_y = next_batch(batch_size, X_train, y_train)
            
            cnt+=1
            merge = tf.summary.merge_all()
            
            summary, _ , batch_cost, batch_acc = session.run([merge, optimizer, cost, acc_pct],
                                               feed_dict={x:batch_x, y:batch_y})

            train_writer.add_summary(summary, cnt)
            
            # aggregate cost and acc for each batch in epoch
            avg_cost += batch_cost / n_batches
            avg_acc_pct += batch_acc / n_batches
            print("Batch {:03} is finished".format(i+1))
            
        
        # verbose
        print ("Epoch {:03}: cost = {:.3f} , acc = {:.2f} %".format(
            epoch+1, avg_cost, avg_acc_pct))

    print("Training Complete.")
    
    # Test model within session
    print ("Test Model...")
    test_cost = cost.eval({x: X_test, y: y_test})
    test_accuracy_pct = acc_pct.eval({x: X_test, y: y_test})
    
    print("Test Cost: {:.3f}".format(test_cost))
    print("Test Accuracy: {:.2f} %".format(test_accuracy_pct))