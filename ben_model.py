import os  # 디렉토리 경로 호출 용도
import cv2  # 이미지 파일 불러올 때 사용
import numpy as np  # 다양한 행렬 연산 (데이터 처리) 용도
from sklearn.preprocessing import LabelEncoder
# 데이터 전처리 (문자로된 폴더 리스트를 숫자형 array로 변환)
from sklearn.preprocessing import OneHotEncoder
# one-hot-encoding을 위해 OneHotEncoder 함수를 불러옴
from numpy import array  # 리스트를 array형태로 만들떄 사용하는 함수
import tensorflow as tf

# tf.set_random_seed(777)
tf.set_random_seed(1)

TRAIN_DIR = "./train_data/"

train_folder_list = array(os.listdir(TRAIN_DIR))
# print(train_folder_list)
# ben, no_ben
train_input = []
train_label = []

label_encoder = LabelEncoder()  # LabelEncoder Class 호출

# 문자열로 구성된 train_folder_list를 숫자형 리스트로 변환
integer_encoded = label_encoder.fit_transform(train_folder_list) # 계수 추정과 자료 변환을 동시에
# print(integer_encoded)
# ben, no_ben

onehot_encoder = OneHotEncoder(sparse=False)
# print(onehot_encoder)
# # OneHotEncoder(categorical_features=None, categories=None,dtype=<class 'numpy.float64'>, handle_unknown='error', n_values=None, sparse=False)

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
'''현재 train_input은 list 형태이므로 shape이 (-1, 4096)인 형태의 np.array로 변환합니다.
(-1, 4096)에서 -1은 데이터의 정확한 개수를 모를 때 사용하는 숫자이며 4096는 이미지의
형태가 64*64이므로 정사각형 모양의 데이터 형태를 1자 형태로 바꾸기 위해 64의 제곱인 4096를 사용'''
# print(train_input[0])
# [255 251 248 ...  62  76  64]
# print('train_input.shape= ', train_input.shape)
# train_input.shape=  (1342, 4096)


train_label = np.reshape(train_label, (-1, 2))
# print('train_label=', train_label.shape)
# train_label= (1342, 2)


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

# hyper parameters
learning_rate = 0.0001

# input place holders
X = tf.placeholder(dtype=tf.float32, shape=[None, 4096])
# [None, 4096]는 4096의 shape을 갖는 데이터를 0~무한대 까지 불러올 수 있다는 뜻

X_img = tf.reshape(X, [-1, 64, 64, 1])  # img 64x64x1 (black/white)
'''
X라는 변수의 shape은 [-1, 4096]이므로 원래 데이터의 shape([-1, 64, 64, 1)]으로 바꾸어
X_img에 저장합니다. [-1, 64, 64, 1] = [batch size, 너비(weight), 높이(height), channel 수]
batch size는 가변할 수 있으므로 대부분 -1로 지정합니다.
grayscale로 이미지를 불러왔으므로 1로 설정되었으며, RGB로 불러왔을 경우에는 3을 기입해야합니다.
'''

Y = tf.placeholder(dtype=tf.float32, shape=[None, 2])
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
'''
필터의 [가로,세로,채널,개수]
Convolution layer의 필터 크기와 개수를 선언하여 W1에 저장합니다.
tf.random_normal[3, 3, 1, 32]에서 tf.random_normal은 정규분포에서 난수를 추출하여 저장한다는 의미이며,
3,3은 필터 크기 즉, 3*3필터를 쓰겠다고 선언한 것이며, 1은 input data의 channel인 1을 지정했습니다. (흑백)
마지막 32는 3*3필터를 총 32개 쓰겠다고 선언
'''

L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
# [28,28,1] --> [28,28,32] (32개의 featuremap을 가짐)
'''
 Convoluton layer를 선언하여 L1에 저장합니다.
 X_img는 해당 Convolution layer의 input이며, W1은 앞서 선언한 Convolution layer의 필터입니다.
 즉, X_img에 W1 필터를 활용하여 Convolution layer를 구성하겠다는 뜻입니다.

 padding=’SAME’은 convolution 연산 후 shape이 줄어드는 것을 방지하기 위하여 설정하는 구문입니다.
 만약 padding=’VALID’로 설정한다면,3*3필터가 28*28이미지를 한칸 씩 움직이며 연산을 수행하므로
 convoultion 연산 이후 출력된 결과의 shape은 28*28이 아닌 26*26이 됩니다.
'''
L1 = tf.nn.relu(L1)  # 활성화 함수는 렐루를 사용
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# [28,28,32] --> [14,14,32] (2x2 max_pooling을 거침)
'''
ksize=[1,2,2,1]에 대한 설명은 다음과 같습니다. ksize는 kernel 사이즈를 의미하며,
Conolution layer의 filter와 동일한 개념이라고 생각하면 됩니다.
첫번째 1은 모든 batch에 대하여 kernel을 적용하겠다는 의미이며,
[2,2]는 2*2크기의 kernel을 사용하겠다는 의미입니다.
마지막 1은 모든 채널에 대하여 kernel을 적용하겠다는 의미
'''

W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
# [14,14,32] --> [14,14,64] (64개의 featuremap)
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# [14,14,64] --> [7,7,64]


L2_flat = tf.reshape(L2, [-1, 16 * 16 * 64])  # 다시 평평하게
'''
처음 input data의 shape은 28*28*1이었습니다.
input data가 convoultion layer(L1)을 거쳐 32개의 3*3의 필터와 연산되었으므로
이 때의 연산 결과는 28*28*32가 됩니다(padding=’SAME’으로 지정하였으므로 data의 크기는
변하지 않습니다). L1은 max-pooling layer(L1)를 거쳐 2*2의 kernel으로 max-pooling 연산을
수행했으므로 이 때의 연산 결과는 14*14*32가 됩니다. 마찬가지로 convolution layer(L2)에서
3*3*64 연산을 수행하면 14*14*64가 되며 max-pooling(L2)에서 2*2 kernel으로 max-pooling
연산을 수행하면 7*7*64가 됩니다.
'''

W3 = tf.get_variable("W3", shape=[16 * 16 * 64, 2], initializer=tf.contrib.layers.xavier_initializer())
'''
 fully-connected 연산을 위해 weight를 선언하여 W3에 저장합니다.
 이 때, W3의 shape을 [7*7*64, 10]으로 지정했는데, input data의 shape이 7*7*64이며,
 output data의 shape이 10(0~9)이기 때문에 W3의 shape을 [7*7*64, 10]으로 설정합니다.
 Weigth의 초기값은 성능이 우수한 것으로 알려진 Xavier initializer를 사용합니다.
'''

b = tf.Variable(tf.random_normal([2]))
# bias를 선언하여 b에 저장합니다. output의 shape이 10이므로 shape을 10으로 설정
logits = tf.matmul(L2_flat, W3) + b

# define cost function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
# loss function을 최소화하는 경사하강법 종류 중 adam optimizer 을 사용
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


training_epochs = 30
'''
 학습 횟수를 설정합니다.
'''
batch_size = 200
'''
효과적인 모델 학습을 위해 batch size를 설정합니다.
batch size는 학습 할 때 몇개의 데이터를 한번에 학습하는가에 관한 설정입니다.
본 실험에서는 42,000개의 데이터를 학습하므로, batch size는 1~42,000까지 설정할 수 있습니다.
200으로 설정했으므로 한번 학습하는데 200개의 데이터를 사용한다는 의미입니다.
'''

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())  # 모든 변수의 weight값을 초기화 합니다.
# result = sess.run(W1)
# print('필터',result)

# train my model
print('Learning started. It takes sometime.')
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(len(train_input) / batch_size)
    '''
    1 epoch에 몇회 학습할 것인지를 설정합니다.
    train_input에 저장된 데이터는 42,000개이므로 len(train_input)은 42,000을,
    batch_size는 200으로 설정했으므로 total_batch는 210이 됩니다.
    '''

    for i in range(total_batch):
        start = ((i + 1) * batch_size) - batch_size  # 0, 200, 400, ..
        end = ((i + 1) * batch_size)  # 200, 400, 600 ..
        batch_xs = train_input[start:end]
        batch_ys = train_label[start:end]
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')

# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={X: test_input, Y: test_label}))