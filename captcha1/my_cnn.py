from .my_get_captcha import get_text_and_image
from .my_get_captcha import get_ans_and_image
from .my_get_captcha import get_image
import numpy as np
import tensorflow as tf
import sys
import json

try:
    json_file = open('params.json', 'r')
    params = json.load(json_file)
    print('成功载入json')
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

char_set = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*']
# text, image = get_text_and_image()
# print("验证码图像channel:", image.shape)  # (60, 160, 3)
# 图像大小
IMAGE_HEIGHT, IMAGE_WIDTH = 80, 79
# MAX_CAPTCHA = len(text)
MAX_CAPTCHA = 1
CHAR_SET_LEN = len(char_set)
text_to_pos = {
    '0': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,
    '+': 10,
    '-': 11,
    '*': 12
}

pos_to_text = {
    0: '0',
    1: '1',
    2: '2',
    3: '3',
    4: '4',
    5: '5',
    6: '6',
    7: '7',
    8: '8',
    9: '9',
    10: '+',
    11: '-',
    12: '*'
}


def text2vec(text):
    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
    vector[text_to_pos[text]] = 1
    return vector


# 向量转回文本
def vec2text(vec):
    for i, mark in enumerate(vec):
        if mark == 1:
            text = pos_to_text[i]
            break
    return text


def get_next_batch(batch_size=128):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])

    for i in range(batch_size):
        text, image = get_text_and_image()

        batch_x[i, :] = image.flatten() / 255  # (image.flatten()-128)/128  mean为0
        batch_y[i, :] = text2vec(text)

    return batch_x, batch_y


X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])
keep_prob = tf.placeholder(tf.float32)  # dropout


def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    # w_c1_alpha = np.sqrt(2.0/(IMAGE_HEIGHT*IMAGE_WIDTH)) #
    # w_c2_alpha = np.sqrt(2.0/(3*3*32))
    # w_c3_alpha = np.sqrt(2.0/(3*3*64))
    # w_d1_alpha = np.sqrt(2.0/(8*32*64))
    # out_alpha = np.sqrt(2.0/1024)

    # 3 conv layer
    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)

    w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)

    # Fully connected layer
    w_d = tf.Variable(w_alpha * tf.random_normal([10 * 10 * 64, 1024]))
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    w_out = tf.Variable(w_alpha * tf.random_normal([1024, MAX_CAPTCHA * CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha * tf.random_normal([MAX_CAPTCHA * CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    # out = tf.nn.softmax(out)
    return out


def train_crack_captcha_cnn():
    output = crack_captcha_cnn()
    # loss
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, Y))
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('loss', loss)
    merged = tf.summary.merge_all()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter('.' + '/train', sess.graph)
        test_writer = tf.summary.FileWriter('.' + '/test')
        sess.run(tf.global_variables_initializer())
        step = 0
        while True:
            batch_x, batch_y = get_next_batch(64)
            summary, _, loss_ = sess.run([merged, optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})
            train_writer.add_summary(summary, step)
            print(step, '损失:', loss_)

            # 每100 step计算一次准确率
            if step % 10 == 0:
                batch_x_test, batch_y_test = get_next_batch(100)
                summary, acc = sess.run([merged, accuracy], feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                test_writer.add_summary(summary, step)
                print(step, '正确率:', acc)
                # 如果准确率大于50%,保存模型,完成训练
                if step > 5000:

                    if acc > 0.999:
                        saver.save(sess, "/home/rivercrown/try/train_1_cnn/crack_capcha.model", global_step=step)
                        break
            step += 1


def crack_single_captcha():
    output = crack_captcha_cnn()

    success = 0
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))
        for x in range(int(10000 / 200)):
            texts = []
            images = []

            for i in range(200):
                text, image = get_text_and_image(is_random=False, index=i + 200 * x)
                texts.append(text)
                image = image.flatten() / 255
                images.append(image)

            predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
            text_list = sess.run(predict, feed_dict={X: images, keep_prob: 1})
            predict_texts = []
            for text in text_list:
                vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
                i = 0
                for n in text:
                    vector[i * CHAR_SET_LEN + n] = 1
                    i += 1
                predict_texts.append(vec2text(vector))
            for i in range(200):
                if texts[i] == predict_texts[i]:
                    success += 1
                else:
                    print("{} 正确: {}  预测: {}".format(i, texts[i], predict_texts[i]))
    # print("{} 正确: {}  预测: {}".format(199, texts[199], predict_texts[199]))
    print('success rate: ', float(success / 10000))


def crack_captcha():
    output = crack_captcha_cnn()
    mappings = open('mappings.txt', 'w')
    success = 0
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))
        for x in range(int(TOTAL_COUNT / BATCH_SIZE)):
            answers = []
            images = []
            lens = []

            for i in range(BATCH_SIZE):
                ans, image_array, char_len = get_ans_and_image(is_random=False, index=i + BATCH_SIZE * x)
                answers.append(ans)
                lens.append(char_len)
                for image in image_array:
                    image = image.flatten() / 255
                    images.append(image)

            predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
            text_list = sess.run(predict, feed_dict={X: images, keep_prob: 1})
            single_char_predict = []
            for text in text_list:
                vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
                i = 0
                for n in text:
                    vector[i * CHAR_SET_LEN + n] = 1
                    i += 1
                single_char_predict.append(vec2text(vector))

            predict_texts = []
            for char_len in lens:
                equation = ''.join(single_char_predict[:char_len])
                ans = str(eval(equation))
                predict_texts.append(equation + '=' + ans)
                single_char_predict = single_char_predict[char_len:]
            for i in range(BATCH_SIZE):
                predict_str = str(i + BATCH_SIZE * x).zfill(4) + ',' + predict_texts[i] + '\n'
                mappings.write(predict_str)
                if answers[i] == predict_texts[i]:
                    success += 1
                else:
                    print("{} 正确: {}  预测: {}".format(i, answers[i], predict_texts[i]))
    # print("{} 正确: {}  预测: {}".format(24, answers[24], predict_texts[24]))
    print('success rate: ', float(success / TOTAL_COUNT))
    mappings.close()


def crack_captcha_without_validation():
    with tf.device('/cpu:0'):
        output = crack_captcha_cnn()
        mappings = open('mappings.txt', 'w')
        saver = tf.train.Saver()
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            saver.restore(sess, tf.train.latest_checkpoint(MODEL_DIR))
            for x in range(int(TOTAL_COUNT / BATCH_SIZE)):
                images = []
                lens = []

                for i in range(BATCH_SIZE):
                    image_array, char_len = get_image(is_random=False, index=i + BATCH_SIZE * x)
                    lens.append(char_len)
                    for image in image_array:
                        image = image.flatten() / 255
                        images.append(image)

                predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
                text_list = sess.run(predict, feed_dict={X: images, keep_prob: 1})
                single_char_predict = []
                for text in text_list:
                    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
                    i = 0
                    for n in text:
                        vector[i * CHAR_SET_LEN + n] = 1
                        i += 1
                    single_char_predict.append(vec2text(vector))

                predict_texts = []
                for char_len in lens:
                    equation = ''.join(single_char_predict[:char_len])
                    ans = str(eval(equation))
                    predict_texts.append(equation + '=' + ans)
                    single_char_predict = single_char_predict[char_len:]
                for i in range(BATCH_SIZE):
                    predict_str = str(i + BATCH_SIZE * x).zfill(4) + ',' + predict_texts[i] + '\n'
                    mappings.write(predict_str)
        mappings.close()


# crack_single_captcha()
# crack_captcha()
# train_crack_captcha_cnn()
crack_captcha_without_validation()
