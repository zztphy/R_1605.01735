import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt


#transfer the 1-D array to one-hot form
def to_one_hot(datasetY):
    max = np.max(datasetY)
    one_hot = np.zeros([datasetY.shape[0], max+1])
    for i in range(datasetY.shape[0]):
        one_hot[i, datasetY[i]] = 1
    return one_hot

#get data randomly from dataset
def get_data(ndata, datasetX, datasetY, set_one_hot):
    index = np.random.randint(low=0, high=datasetX.shape[0], size=ndata)
    if set_one_hot == True:
        return datasetX[index,:], to_one_hot(datasetY[index])
    else:
        return datasetX[index,:], datasetY[index]

#number of sample for test per T
sample_num = 250

#number of sampled T
T_num = 41

#size of the lattice is L*L
L = 10
#number of neurals in hidden layer
H = 100
#number of iterations
NSteps = 10000
#batch size
batch_size = 100

#Reading data
#np.loadtxt returns numpy.ndarray
SaveName = 'L_10'
PathToData = './DataIsing/L_10/'
DataTrainX = np.loadtxt(PathToData+'Xtrain.txt',dtype='int')
DataTrainY = np.loadtxt(PathToData+'ytrain.txt',dtype='int')
DataTestX = np.loadtxt(PathToData+'Xtest.txt',dtype='int')
DataTestY = np.loadtxt(PathToData+'Ytest.txt',dtype='int')
T_test = np.loadtxt('./DataIsing/T_test.txt', dtype='float32')

#placeholders
X = tf.placeholder(tf.float32, shape=[None, L*L])
Y_acc = tf.placeholder(tf.float32, shape=[None, 2])

#varialbes
W1 = tf.Variable(tf.truncated_normal([L*L, H], stddev=0.1))
W2 = tf.Variable(tf.truncated_normal([H, 2], stddev=0.1))
b1 = tf.Variable(tf.zeros([H]))
b2 = tf.Variable(tf.zeros([2]))

#forword propagation
Y1 = tf.sigmoid(tf.matmul(X, W1) + b1)
Ylogits = tf.matmul(Y1, W2) + b2
Y_pred = tf.nn.softmax(Ylogits)

#Loss function:cross entropy
Loss = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_acc)
Loss = tf.reduce_mean(Loss)*100

#calculating accuracy
correct_prediction = tf.equal(tf.argmax(Y_acc, 1), tf.argmax(Y_pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#train step
#learning rate
lr = 0.003
train_step = tf.train.AdamOptimizer(lr).minimize(Loss)

#init run
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#Loop
for i in range(NSteps):
    batch_X, batch_Y = get_data(batch_size, DataTrainX, DataTrainY, 1)

    sess.run(train_step, feed_dict={X: batch_X, Y_acc: batch_Y})
    acc_train, loss_train = sess.run([accuracy, Loss], feed_dict={X: batch_X, Y_acc: batch_Y})

    acc_test, loss_test = sess.run([accuracy, Loss], feed_dict={X: DataTestX, Y_acc: to_one_hot(DataTestY)})

    if i%100 == 0:
        print('#####%i#####' %(i))
        print('Accuracy and loss for training is:%f, %f\n\n' %(acc_train, loss_train))
        print('Accuracy and loss for test is:%f, %f\n' %(acc_test, loss_test))

pred_result = sess.run(Y_pred, feed_dict={X: DataTestX})

result_statis = np.array(np.zeros([T_num, 5]), dtype=np.float32)
for i in range(T_num):
    ave_temp = np.mean(pred_result[i*sample_num: (i+1)*sample_num, :], axis=0)
    std_temp = np.std(pred_result[i*sample_num: (i+1)*sample_num, :], axis=0)
    result_statis[i, 0] = T_test[i]
    result_statis[i, 1] = ave_temp[0]
    result_statis[i, 2] = std_temp[0]
    result_statis[i, 3] = ave_temp[1]
    result_statis[i, 4] = std_temp[1]

np.savetxt(SaveName, result_statis)



