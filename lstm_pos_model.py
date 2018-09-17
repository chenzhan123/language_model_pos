import tensorflow as tf
import collections
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.corpus import brown
from tensorflow.contrib import rnn

raw_sents=brown.tagged_sents(tagset="universal")
raw_words=brown.tagged_words(tagset="universal")
x_length=[len(sent) for sent in raw_sents]
words=[];words_pos=[]
for word,pos in raw_words:
    words_pos.append(pos)

embedding_size = 80
max_time = max(x_length) #一共180
lstm_size = 80 #隐层单元
n_classes = len(set(words_pos))   # 12个分类
batch_size = 50 #每批次50个样本
vocab_size=5000
layer_num=2
keep_prob=0.3
labels=['NUM', 'X', 'ADV', 'ADJ', 'CONJ', 'NOUN', 'ADP', 'PRON', '.', 'PRT', 'VERB', 'DET']
labels_dict=dict()
for i in range(len(labels)):
    labels_dict[labels[i]]=i
reverse_labels_dict=dict(zip(labels_dict.values(),labels_dict.keys()))

def build_dataset(sents):
    words=[];words_pos=[];data_x=[];data_y=[]
    for items in sents:
        for word,pos in items:
            words.append(word)
            words_pos.append(pos)
    count=[['UNK',-1]]
    count.extend(collections.Counter(words).most_common(vocab_size-1))
    dictionary=dict()
    for word,_ in count:
        dictionary[word]=len(dictionary)
    unk_count=0
    for items in sents:
        temp_x=[];temp_y=[]
        for word,pos in items:
            if word in dictionary:
                temp_x.append(dictionary[word])
            else:
                temp_x.append(0)
                unk_count+=1
            temp_y.append(labels_dict[pos])
        temp_x.extend([0] * (max_time - len(temp_x)))
        temp_y.extend([0] * (max_time - len(temp_y)))
        data_x.append(temp_x)
        data_y.append(temp_y)
    count[0][1]=unk_count
    reverse_dictionary=dict(zip(dictionary.values(),dictionary.keys()))
    return data_x,data_y,count,dictionary,reverse_dictionary

data_x,data_y,count,dicionary,reverse_dictionary=build_dataset(raw_sents)
#
data_y=np.array(data_y).flatten()
data_y=np.reshape(np.array(pd.get_dummies(data_y)),[-1,180,12])
x_train,x_test,y_train,y_test,x_length_train,x_length_test=train_test_split(data_x,data_y,x_length,test_size=0.2)

# begining
tf.reset_default_graph()
# 这里的none表示第一个维度可以是任意的长度
X_input = tf.placeholder(tf.int32, [None, max_time])
Y_input = tf.placeholder(tf.int32, [None, max_time, 12])
sents_length=tf.placeholder(tf.int32,[None])

# 初始化权值
weights = tf.Variable(tf.truncated_normal([lstm_size, n_classes], stddev=0.1))
# 初始化偏置值
biases = tf.Variable(tf.constant(0.1, shape=[n_classes]))

# **步骤1：RNN 的输入shape = (batch_size, timestep_size, input_size)
embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1, 1))
inputs = tf.nn.embedding_lookup(embedding, X_input)
inputs = tf.reshape(inputs, [-1, max_time, embedding_size])

# **步骤2：定义一层 LSTM_cell，只需要说明 hidden_size, 它会自动匹配输入的 X 的维度
# lstm_cell = rnn.BasicLSTMCell(num_units=lstm_size, forget_bias=0.0,state_is_tuple=True)

# **步骤4：调用 MultiRNNCell 来实现多层 LSTM
def lstm_cell():
    return rnn.BasicLSTMCell(num_units=lstm_size, forget_bias=0.0,state_is_tuple=True)

drop_cell=[rnn.DropoutWrapper(lstm_cell(),output_keep_prob=keep_prob) for _ in range(layer_num)]
mlstm_cell = rnn.MultiRNNCell(drop_cell,  state_is_tuple=True)

# **步骤5：用全零来初始化state
init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)
state=init_state
outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=inputs, initial_state=state, time_major=False)

# **步骤6：计算损失及准确率
loss = 0;
accuarcy = 0
for i in range(batch_size):
    pre = tf.matmul(outputs[i, :sents_length[i], :], weights) + biases
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=pre, labels=Y_input[i, :sents_length[i], :]))
    correct_prediction = tf.equal(tf.argmax(Y_input[i, :sents_length[i], :], 1), tf.argmax(pre, 1))
    accuracy_batch = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))  #
    loss += cross_entropy
    accuarcy += accuracy_batch
accuarcy = accuarcy / tf.cast(tf.reduce_sum(sents_length), tf.float32)
train_step=tf.train.AdagradOptimizer(0.1).minimize(loss)





