import os  # 디렉토리 경로 호출 용도
import cv2  # 이미지 파일 불러올 때 사용
import numpy as np  # 다양한 행렬 연산 (데이터 처리) 용도
from sklearn.preprocessing import LabelEncoder
# 데이터 전처리 (문자로된 폴더 리스트를 숫자형 array로 변환)
from sklearn.preprocessing import OneHotEncoder
# one-hot-encoding을 위해 OneHotEncoder 함수를 불러옴
from numpy import array  # 리스트를 array형태로 만들떄 사용하는 함수
import tensorflow as tf


TRAIN_DIR = "./train_data/"

train_folder_list = array(os.listdir(TRAIN_DIR))
# print(train_folder_list)
# ben, no_ben
train_input = []
train_label = []

label_encoder = LabelEncoder()  # LabelEncoder Class 호출

# 문자열로 구성된 train_folder_list를 숫자형 리스트로 변환
integer_encoded = label_encoder.fit_transform(train_folder_list)  # 계수 추정과 자료 변환을 동시에
# print(integer_encoded)
# ben, no_ben

onehot_encoder = OneHotEncoder(sparse=False)
# print(onehot_encoder)
# OneHotEncoder(categorical_features=None, categories=None,dtype=<class 'numpy.float64'>,
# handle_unknown='error', n_values=None, sparse=False)

integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
# OneHotEncoder를 사용하기 위해 integer_encoded의 shape을 (2,)에서 (2,1)로 변환
# print(integer_encoded)
# [[0]
#  [1]]

onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
# OneHotEncoder를 사용하여 integer_encoded를 다음과 같이 변환하여 onehot_encoded 변수에 저장
# print(onehot_encoded)
# [[1. 0.]
#  [0. 1.]]

for index in range(len(train_folder_list)):
    path = os.path.join(TRAIN_DIR, train_folder_list[index])
    path = path + '/'
    img_list = os.listdir(path)
    for img in img_list:
        img_path = os.path.join(path, img)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        train_input.append([np.array(img)])  # 이미지를 array 형태로 바꾸고 리스트에 넣음
        train_label.append([np.array(onehot_encoded[index])])


train_input = np.reshape(train_input, (-1, 4096))
# print(train_input[0])
# [255 251 248 ...  62  76  64]
# print('train_input.shape:', train_input.shape)
# train_input.shape: (1328, 4096)


train_label = np.reshape(train_label, (-1, 2))
# print('train_label:', train_label.shape)
# train_label: (1328, 2)


train_input = np.array(train_input).astype(np.float32)
train_label = np.array(train_label).astype(np.float32)
# np.save("train_data.npy", train_input)
# np.save("train_label.npy", train_label)


TEST_DIR = './test_data/'
test_folder_list = array(os.listdir(TEST_DIR))

test_input = []
test_label = []

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(test_folder_list)

onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

for index in range(len(test_folder_list)):
    path = os.path.join(TEST_DIR, test_folder_list[index])
    path = path + '/'
    img_list = os.listdir(path)
    for img in img_list:
        img_path = os.path.join(path, img)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        test_input.append([np.array(img)])
        test_label.append([np.array(onehot_encoded[index])])

test_input = np.reshape(test_input, (-1, 4096))
test_label = np.reshape(test_label, (-1, 2))
test_input = np.array(test_input).astype(np.float32)
test_label = np.array(test_label).astype(np.float32)
# np.save("test_input.npy", test_input)
# np.save("test_label.npy", test_label)

# set hyper parameters
batch_size = 100
learning_rate = 0.001
training_epochs = 33

# set random_seed
tf.set_random_seed(1)

# input placeholder of batch normalization
# batch_prob = tf.placeholder(tf.bool)

# input placeholders
X = tf.placeholder(dtype=tf.float32, shape=[None, 4096])  # 64x64=4096 [None, 4096] shape의 데이터를 불러올 수 있음
X_img = tf.reshape(X, [-1, 64, 64, 1])  # img 64x64x1 (black/white)
Y = tf.placeholder(dtype=tf.float32, shape=[None, 2])


################### Layer 1 ###################
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
# L1 = tf.layers.batch_normalization(L1, center=True, scale=True, training=batch_prob)  # 배치정규화
L1 = tf.nn.relu(L1)  # 활성화 함수로 렐루 사용
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


################### Layer 2 ###################
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
# [14,14,32] --> [14,14,64] (64개의 feature map)
# L2 = tf.layers.batch_normalization(L2, center=True, scale=True, training=batch_prob)
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# [14,14,64] --> [7,7,64]


################### Layer 3 ###################
W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
# L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


L3_flat = tf.reshape(L3, [-1, 16 * 16 * 128])  # 다시 평평하게
W4 = tf.get_variable("W4", shape=[16 * 16 * 128, 2], initializer=tf.contrib.layers.xavier_initializer())


# set bias of filter
b = tf.Variable(tf.random_normal([2]))
logits = tf.matmul(L3_flat, W4) + b

# define cost function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
# loss function을 최소화하는 경사하강법 종류 중 adam optimizer 을 사용
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())  # 모든 변수의 weight값을 초기화 합니다.
# result = sess.run(W1)
# print('필터', result)


# train my model
print('Learning started. It takes sometime.')

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(len(train_input) / batch_size)  # 1328 / 100 = 13

    for i in range(total_batch):
        start = ((i + 1) * batch_size) - batch_size  # 0, 100, 200, ..
        end = ((i + 1) * batch_size)  # 100, 200, 300 ..
        batch_xs = train_input[start:end]
        batch_ys = train_label[start:end]
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')

# Test model and check accuracy
correct_prediction = logits
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
a = sess.run(correct_prediction, feed_dict={X: test_input})


# 소프트맥스 함수
def softmax(x):
    c = np.max(x)
    return np.exp(x - c) / np.sum(np.exp(x - c))


predict_result = []

for i in range(len(a)):
    sm = softmax(a[i])
    print(sm)
    predict_result.append(np.argmax(sm))

print(predict_result)
