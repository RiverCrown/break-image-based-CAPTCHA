import tensorflow as tf
import numpy as np
from my_get_batch import get_one_batch
from my_get_batch import get_one_image_batch
from calculate_distance import pairwise_distance
from triplet_loss import batch_hard_triplet_loss
from scipy import optimize
import json
import sys

try:
    json_file = open('params.json', 'r')
    params = json.load(json_file)
except IOError:
    print('找不到配置文件params.json！')
    sys.exit(0)
else:
    json_file.close()

TOTAL_COUNT = params['totalCount']
BATCH_SIZE = params['batchSize']
MODEL_DIR = params['modelDir']
if TOTAL_COUNT % BATCH_SIZE != 0:
	print('检查params.json，totalCount不能整除batchSize')
	sys.exit(0)

X = tf.placeholder(tf.float32, [None, 1665])
Y = tf.placeholder(tf.float32, [None])
keep_prob = tf.placeholder(tf.float32)
is_training = tf.placeholder(tf.bool)

def get_prediction(out, squared=True):
    distance = pairwise_distance(out, squared=squared)
    distance = distance[:4,4:]
    prediction = tf.cast(tf.argmin(distance, axis=-1), tf.float32)
    return prediction

def vgg_net_test(w_alpha=0.01, b_alpha=0.1, embedding_size=2048):
    x = tf.reshape(X, shape=[-1, 45, 37, 1])

    w_c1 = tf.Variable(w_alpha*tf.random_normal([3, 3, 1, 64]))
    b_c1 = tf.Variable(b_alpha*tf.random_normal([64]))
    conv1 = tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1)
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    w_c2 = tf.Variable(w_alpha*tf.random_normal([3, 3, 64, 128]))
    b_c2 = tf.Variable(b_alpha*tf.random_normal([128]))
    conv2 = tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2)
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    w_c3 = tf.Variable(w_alpha*tf.random_normal([3, 3, 128, 256]))
    b_c3 = tf.Variable(b_alpha*tf.random_normal([256]))
    conv3 = tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3)
    conv3 = tf.nn.relu(conv3)
    
    w_c4 = tf.Variable(w_alpha*tf.random_normal([3, 3, 256, 256]))
    b_c4 = tf.Variable(b_alpha*tf.random_normal([256]))
    conv4 = tf.nn.bias_add(tf.nn.conv2d(conv3, w_c4, strides=[1, 1, 1, 1], padding='SAME'), b_c4)
    conv4 = tf.nn.relu(conv4)
    conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    w_c5 = tf.Variable(w_alpha*tf.random_normal([3, 3, 256, 512]))
    b_c5 = tf.Variable(b_alpha*tf.random_normal([512]))
    conv5 = tf.nn.bias_add(tf.nn.conv2d(conv4, w_c5, strides=[1, 1, 1, 1], padding='SAME'), b_c5)
    conv5 = tf.nn.relu(conv5)

    w_c6 = tf.Variable(w_alpha*tf.random_normal([3, 3, 512, 512]))
    b_c6 = tf.Variable(b_alpha*tf.random_normal([512]))
    conv6 = tf.nn.bias_add(tf.nn.conv2d(conv5, w_c6, strides=[1, 1, 1, 1], padding='SAME'), b_c6)
    conv6 = tf.nn.relu(conv6)
    conv6 = tf.nn.max_pool(conv6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    w_c7 = tf.Variable(w_alpha*tf.random_normal([3, 3, 512, 512]))
    b_c7 = tf.Variable(b_alpha*tf.random_normal([512]))
    conv7 = tf.nn.bias_add(tf.nn.conv2d(conv6, w_c7, strides=[1, 1, 1, 1], padding='SAME'), b_c7)
    conv7 = tf.nn.relu(conv7)

    w_c8 = tf.Variable(w_alpha*tf.random_normal([3, 3, 512, 512]))
    b_c8 = tf.Variable(b_alpha*tf.random_normal([512]))
    conv8 = tf.nn.bias_add(tf.nn.conv2d(conv7, w_c8, strides=[1, 1, 1, 1], padding='SAME'), b_c8)
    conv8 = tf.nn.relu(conv8)
    conv8 = tf.nn.max_pool(conv8, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    w_d1 = tf.Variable(w_alpha*tf.random_normal([2048, 4096]))
    b_d1 = tf.Variable(b_alpha*tf.random_normal([4096]))
    dense1 = tf.layers.flatten(conv8)
    dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, w_d1), b_d1))
    dense1 = tf.nn.dropout(dense1, keep_prob)

    w_d2 = tf.Variable(w_alpha*tf.random_normal([4096, 4096]))
    b_d2 = tf.Variable(b_alpha*tf.random_normal([4096]))
    dense2 = tf.nn.relu(tf.add(tf.matmul(dense1, w_d2), b_d2))
    dense2 = tf.nn.dropout(dense2, keep_prob)

    w_out = tf.Variable(w_alpha*tf.random_normal([4096, embedding_size]))
    b_out = tf.Variable(b_alpha*tf.random_normal([embedding_size]))
    out = tf.add(tf.matmul(dense2, w_out), b_out)

    return out

def vgg_net_test_2(w_alpha=0.01, b_alpha=0.1, embedding_size=1024):
    x = tf.reshape(X, shape=[-1, 45, 37, 1])

    w_c1 = tf.Variable(w_alpha*tf.random_normal([3, 3, 1, 64]))
    b_c1 = tf.Variable(b_alpha*tf.random_normal([64]))
    conv1 = tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1)
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    w_c2 = tf.Variable(w_alpha*tf.random_normal([3, 3, 64, 128]))
    b_c2 = tf.Variable(b_alpha*tf.random_normal([128]))
    conv2 = tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2)
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    w_c3 = tf.Variable(w_alpha*tf.random_normal([3, 3, 128, 256]))
    b_c3 = tf.Variable(b_alpha*tf.random_normal([256]))
    conv3 = tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3)
    conv3 = tf.nn.relu(conv3)
    
    w_c4 = tf.Variable(w_alpha*tf.random_normal([3, 3, 256, 256]))
    b_c4 = tf.Variable(b_alpha*tf.random_normal([256]))
    conv4 = tf.nn.bias_add(tf.nn.conv2d(conv3, w_c4, strides=[1, 1, 1, 1], padding='SAME'), b_c4)
    conv4 = tf.nn.relu(conv4)
    conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    w_c5 = tf.Variable(w_alpha*tf.random_normal([3, 3, 256, 512]))
    b_c5 = tf.Variable(b_alpha*tf.random_normal([512]))
    conv5 = tf.nn.bias_add(tf.nn.conv2d(conv4, w_c5, strides=[1, 1, 1, 1], padding='SAME'), b_c5)
    conv5 = tf.nn.relu(conv5)

    w_c6 = tf.Variable(w_alpha*tf.random_normal([3, 3, 512, 512]))
    b_c6 = tf.Variable(b_alpha*tf.random_normal([512]))
    conv6 = tf.nn.bias_add(tf.nn.conv2d(conv5, w_c6, strides=[1, 1, 1, 1], padding='SAME'), b_c6)
    conv6 = tf.nn.relu(conv6)
    conv6 = tf.nn.max_pool(conv6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    w_c7 = tf.Variable(w_alpha*tf.random_normal([3, 3, 512, 512]))
    b_c7 = tf.Variable(b_alpha*tf.random_normal([512]))
    conv7 = tf.nn.bias_add(tf.nn.conv2d(conv6, w_c7, strides=[1, 1, 1, 1], padding='SAME'), b_c7)
    conv7 = tf.nn.relu(conv7)

    w_c8 = tf.Variable(w_alpha*tf.random_normal([3, 3, 512, 512]))
    b_c8 = tf.Variable(b_alpha*tf.random_normal([512]))
    conv8 = tf.nn.bias_add(tf.nn.conv2d(conv7, w_c8, strides=[1, 1, 1, 1], padding='SAME'), b_c8)
    conv8 = tf.nn.relu(conv8)
    conv8 = tf.nn.max_pool(conv8, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    w_d1 = tf.Variable(w_alpha*tf.random_normal([2048, 2048]))
    b_d1 = tf.Variable(b_alpha*tf.random_normal([2048]))
    dense1 = tf.layers.flatten(conv8)
    dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, w_d1), b_d1))
    dense1 = tf.nn.dropout(dense1, keep_prob)

    w_d2 = tf.Variable(w_alpha*tf.random_normal([2048, 2048]))
    b_d2 = tf.Variable(b_alpha*tf.random_normal([2048]))
    dense2 = tf.nn.relu(tf.add(tf.matmul(dense1, w_d2), b_d2))
    dense2 = tf.nn.dropout(dense2, keep_prob)

    w_out = tf.Variable(w_alpha*tf.random_normal([2048, embedding_size]))
    b_out = tf.Variable(b_alpha*tf.random_normal([embedding_size]))
    out = tf.add(tf.matmul(dense2, w_out), b_out)

    return out


def train_crack_captcha_cnn():
    out = vgg_net_test()
    #out = vgg_net_test_2()
    test = tf.reshape(out, [-1, 13, 2048])
    triplet_loss = batch_hard_triplet_loss(labels=Y, embeddings=out, margin=6.0)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0003).minimize(triplet_loss)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, 'vgg_test_model/crack_capcha.model-100000')
        step = 0
        while True:
            batch_x, batch_y = get_one_batch(is_random=False, index=int(step%7500))
            _, loss_ = sess.run([optimizer, triplet_loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75, is_training: True})
            
            if step % 100 == 0:
                print(step, loss_)
            if step % 10000 == 0:
                success = 0
                for i in range(int(1000/25)):
                    images = []
                    labels = []
                    for j in range(25):
                        x, y = get_one_batch()
                        for image in x:
                            images.append(image)
                        label = y[0:4]
                        labels.append(label)
                    predictions = tf.map_fn(get_prediction, test, dtype=tf.float32)
                    predict = sess.run(tf.cast(predictions, tf.int32), feed_dict={X: images, keep_prob: 1.0})
            
                    for j in range(25):
                        if (labels[j] == predict[j]).all():
                            success += 1
                print('success rate')
                acc = success/1000
                if acc > 0.99 or (step % 10000 == 0 and step > 0):
                    saver.save(sess, "/home/rivercrown/try/train_5_net_work/triplet_network/vgg_test_margin_6/crack_capcha.model", global_step=step)
                print(acc)
            step += 1

def crack_capcha():
    #out = vgg_net()
    out = vgg_net_test()
    out = tf.reshape(out, [-1, 13, 2048])
    success = 0
    saver = tf.train.Saver()
    with tf.Session() as sess:
        #saver.restore(sess, tf.train.latest_checkpoint('.'))
        saver.restore(sess, 'vgg_test_model/crack_capcha.model-60000')
        for a in range(int(2500/25)):
            images = []
            labels = []
            for i in range(25):
                x, y = get_one_batch(is_random=False, index=7500+i+25*a)
                for image in x:
                    images.append(image)
                label = y[0:4]
                labels.append(label)
                
            predictions = tf.map_fn(get_prediction, out, dtype=tf.float32)
            predict = sess.run(tf.cast(predictions, tf.int32), feed_dict={X: images, keep_prob: 1.0})
            
            for i in range(25):
                if (labels[i] == predict[i]).all():
                    success += 1
                '''
                else:
                    print(labels[i])
                    print(predict[i])
                    print(' ')
                '''
            
    print('success rate:', float(success/2500))

def crack_capcha_without_validation():
    with tf.device('/cpu:0'):
        mapping = open('mappings.txt', 'w')
        out = vgg_net_test()
        out = tf.reshape(out, [-1, 13, 2048])
        saver = tf.train.Saver()
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            saver.restore(sess, tf.train.latest_checkpoint(MODEL_DIR))
            for a in range(int(TOTAL_COUNT/BATCH_SIZE)):
                images = []
                for i in range(BATCH_SIZE):
                    x = get_one_image_batch(is_random=False, index=i+BATCH_SIZE*a)
                    for image in x:
                        images.append(image)
                    
                predictions = tf.map_fn(get_prediction, out, dtype=tf.float32)
                predict = sess.run(tf.cast(predictions, tf.int32), feed_dict={X: images, keep_prob: 1.0})
                
                for i in range(BATCH_SIZE):
                    predict_str = [str(x) for x in predict[i]]
                    predict_str = ''.join(predict_str)
                    predict_str = str(i+BATCH_SIZE*a).zfill(4) + ',' + predict_str + '\n'
                    mapping.write(predict_str)
                
        mapping.close()

#train_crack_captcha_cnn()
#crack_capcha()
crack_capcha_without_validation()
