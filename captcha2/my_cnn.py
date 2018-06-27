from my_get_captcha import get_text_and_image
from my_get_captcha import get_image
import numpy as np
import tensorflow as tf
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

number = ['0','1','2','3','4','5','6','7','8','9']
alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
ALPHABET = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
#text, image = get_text_and_image()
#print("验证码图像channel:", image.shape)  # (60, 160, 3)
# 图像大小
IMAGE_HEIGHT, IMAGE_WIDTH = 60, 160
#MAX_CAPTCHA = len(text)
MAX_CAPTCHA = 5
#print("验证码文本最长字符数", MAX_CAPTCHA)

# 文本转向量
char_set = number + alphabet + ALPHABET + ['_']  
CHAR_SET_LEN = len(char_set)
def text2vec(text):
	text_len = len(text)
	if text_len > MAX_CAPTCHA:
		raise ValueError('验证码最长5个字符')
 
	vector = np.zeros(MAX_CAPTCHA*CHAR_SET_LEN)
	def char2pos(c):
		if c =='_':
			k = 62
			return k
		k = ord(c)-48
		if k > 9:
			k = ord(c) - 55
			if k > 35:
				k = ord(c) - 61
				if k > 61:
					raise ValueError('No Map') 
		return k
	for i, c in enumerate(text):
		idx = i * CHAR_SET_LEN + char2pos(c)
		vector[idx] = 1
	return vector
# 向量转回文本
def vec2text(vec):
	char_pos = vec.nonzero()[0]
	text=[]
	for i, c in enumerate(char_pos):
		char_at_pos = i #c/63
		char_idx = c % CHAR_SET_LEN
		if char_idx < 10:
			char_code = char_idx + ord('0')
		elif char_idx <36:
			char_code = char_idx - 10 + ord('A')
		elif char_idx < 62:
			char_code = char_idx-  36 + ord('a')
		elif char_idx == 62:
			char_code = ord('_')
		else:
			raise ValueError('error')
		text.append(chr(char_code))
	return "".join(text)

def get_next_batch(batch_size=128):
	batch_x = np.zeros([batch_size, IMAGE_HEIGHT*IMAGE_WIDTH])
	batch_y = np.zeros([batch_size, MAX_CAPTCHA*CHAR_SET_LEN])
 
	for i in range(batch_size):
		text, image = get_text_and_image()
 
		batch_x[i,:] = image.flatten() / 255 # (image.flatten()-128)/128  mean为0
		batch_y[i,:] = text2vec(text)
 
	return batch_x, batch_y

X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT*IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA*CHAR_SET_LEN])
keep_prob = tf.placeholder(tf.float32) # dropout

def vgg_net(w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

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

    w_c3 = tf.Variable(w_alpha*tf.random_normal([3, 3, 128, 128]))
    b_c3 = tf.Variable(b_alpha*tf.random_normal([128]))
    conv3 = tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3)
    conv3 = tf.nn.relu(conv3)
    
    w_c4 = tf.Variable(w_alpha*tf.random_normal([3, 3, 128, 128]))
    b_c4 = tf.Variable(b_alpha*tf.random_normal([128]))
    conv4 = tf.nn.bias_add(tf.nn.conv2d(conv3, w_c4, strides=[1, 1, 1, 1], padding='SAME'), b_c4)
    conv4 = tf.nn.relu(conv4)
    conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    w_d1 = tf.Variable(w_alpha*tf.random_normal([20480, 1024]))
    b_d1 = tf.Variable(b_alpha*tf.random_normal([1024]))
    dense1 = tf.layers.flatten(conv4)
    dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, w_d1), b_d1))
    dense1 = tf.nn.dropout(dense1, keep_prob)

    w_out = tf.Variable(w_alpha*tf.random_normal([1024, MAX_CAPTCHA*CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha*tf.random_normal([MAX_CAPTCHA*CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense1, w_out), b_out)

    return out

def train_crack_captcha_cnn():
	#output = crack_captcha_cnn()
	output = vgg_net()
	# loss
	#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, Y))
	loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
	optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
 
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
		#saver.restore(sess, 'new_model/crack.model-2000')

		step = 0
		while True:
			batch_x, batch_y = get_next_batch(64)
			summary, _, loss_ = sess.run([merged, optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})
			train_writer.add_summary(summary, step)
			print(step, loss_)
			
			# 每100 step计算一次准确率
			if step % 1000 == 0:
				batch_x_test, batch_y_test = get_next_batch(100)
				summary, acc = sess.run([merged, accuracy], feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
				test_writer.add_summary(summary, step)
				print(step, acc)
				# 保存模型
				if acc == 1.0:
					saver.save(sess, "/home/rivercrown/try/train_2_cnn/new_model_2/crack.model", global_step=step)
			
			step += 1

def crack_captcha():
	output = vgg_net()
	mapping = open('mappings.txt', 'w')
	success = 0
	saver = tf.train.Saver()
	with tf.Session() as sess:
		#saver.restore(sess, tf.train.latest_checkpoint('.'))
		saver.restore(sess, 'new_model/crack.model-5000')
		for x in range(int(2500/100)):
			texts = []
			images = []
			for i in range(100):
				text, image = get_text_and_image(is_random = False, index =7500 + i + 100 * x)
				texts.append(text)
				image = image.flatten() / 255
				images.append(image) 
		
			predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
			text_list = sess.run(predict, feed_dict={X: images, keep_prob: 1})
			predict_texts = []
			for text in text_list:
				vector = np.zeros(MAX_CAPTCHA*CHAR_SET_LEN)
				i = 0
				for n in text:
					vector[i*CHAR_SET_LEN + n] = 1
					i += 1
				predict_texts.append(vec2text(vector))
			for i in range(100):
				predict_str = str(i + 100 * x).zfill(4) + ',' + predict_texts[i] + '\n'
				mapping.write(predict_str)
				if texts[i] == predict_texts[i]:
					success += 1
				else:
					print("{} 正确: {}  预测: {}".format(i, texts[i], predict_texts[i]))
	print('success rate: ', float(success/2500))
	mapping.close()

def crack_captcha_without_validation():
	with tf.device('/cpu:0'):
		output = vgg_net()
		mapping = open('mappings.txt', 'w')
		saver = tf.train.Saver()
		with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
			#saver.restore(sess, tf.train.latest_checkpoint('.'))
			saver.restore(sess, tf.train.latest_checkpoint(MODEL_DIR))
			for x in range(int(TOTAL_COUNT/BATCH_SIZE)):
				images = []
				for i in range(BATCH_SIZE):
					image = get_image(is_random = False, index = i + BATCH_SIZE * x)
					image = image.flatten() / 255
					images.append(image) 
			
				predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
				text_list = sess.run(predict, feed_dict={X: images, keep_prob: 1})
				predict_texts = []
				for text in text_list:
					vector = np.zeros(MAX_CAPTCHA*CHAR_SET_LEN)
					i = 0
					for n in text:
						vector[i*CHAR_SET_LEN + n] = 1
						i += 1
					predict_texts.append(vec2text(vector))
				for i in range(BATCH_SIZE):
					predict_str = str(i + BATCH_SIZE * x).zfill(4) + ',' + predict_texts[i] + '\n'
					mapping.write(predict_str)
		
		mapping.close()


crack_captcha_without_validation()
#crack_captcha()
#train_crack_captcha_cnn()
