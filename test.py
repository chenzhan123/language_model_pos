from sklearn.model_selection import train_test_split
from nltk.corpus import brown
from traditional_pos_model import *
from lstm_pos_model import *

# def pre_real_accuarcy(x_test,real,Model):
#     pre=[]
#     for item in x_test:
#         result=Model.tag(item)
#         for i in range(len(result)):
#             pre.append(result[i][1])
#     reality=[]
#     for items in real:
#         for item in items:
#             reality.append(item)
#     acc=0
#     for i in range(len(reality)):
#         if pre[i]==reality[i]:
#             acc+=1
#     return acc/len(reality)

def data_split(train_data,rate):
    x_train,x_test=train_test_split(train_data,test_size=rate)
    nx_test=[];ny_test=[]
    for items in x_test:
        temp_x=[];temp_y=[]
        for word,pos in items:
            temp_x.append(word)
            temp_y.append(pos)
        nx_test.append(temp_x)
        ny_test.append(temp_y)
    return x_train,nx_test,ny_test

## load data
# there are train data, test data. If you want to acquire accuracy, you can define function yourself
sents=brown.tagged_sents(tagset="universal")
train_data,x_test,y_test=data_split(sents,0.2)

####################
###     test     ###
####################
# when you have trained a model, use: model.load(r"./Model/Averaged_perceptron.pickle")
# there are some models trained before in file Model. And you can load them
## 1.averaged perceptron(trained)
mapping = load_universal_map(r"./Model/en-ptb.map")
model1=PerceptronTagger(load=True)   # when load is true, then apply trained averaged perceptron to the data
model1.tag(x_test[0])   # third column is the confidence weight: the number is larger, the result is more confident

## 2.averaged perceptron
model2=PerceptronTagger(load=False)  # when load is False, you need train a model for your data
model2.train(train_data)
model2.tag(x_test[0])

## 3.HMM
model3=HMM(train_data)
model3.tag(x_test[0])

## 4.LSTM
#### training
# if you have finished training, you skip this step to the next step that loading model.
saver = tf.train.Saver()
init=tf.global_variables_initializer()
start=time.time()
sess=tf.Session()
sess.run(init)
for epoch in range(801):
    acc_rate = 0
    for i in range(len(x_train)//batch_size):
        x=x_train[i*batch_size:(i+1)*batch_size]
        y=y_train[i*batch_size:(i+1)*batch_size]
        length=x_length_train[i*batch_size:(i+1)*batch_size]
        sess.run(train_step,feed_dict={X_input:x,Y_input:y,sents_length:length})
        acc=sess.run(accuarcy,feed_dict={X_input:x,Y_input:y,sents_length:length})
        acc_rate +=acc
    acc_rate=acc_rate/(len(x_train)//batch_size)
    print("step: %s , training accuracy: %s" %(epoch,acc_rate))
    if epoch%100==0:
        saver.save(sess, 'Model/lstm_2_80_second.ckpt')

acc_rate=0
for i in range(len(y_test)//batch_size):
    x = x_test[i * batch_size:(i + 1) * batch_size]
    y = y_test[i * batch_size:(i + 1) * batch_size]
    length = x_length_test[i * batch_size:(i + 1) * batch_size]
    acc=sess.run(accuarcy, feed_dict={X_input: x, Y_input: y, sents_length: length})
    acc_rate+=acc
acc_rate=acc_rate / (len(y_test) // batch_size)
print("test accuracy: %s" %(acc_rate))
spend_time=time.time()-start
saver.save(sess,'Model/lstm_2_80.ckpt')
sess.close()
### loading
saver = tf.train.Saver()
sess1=tf.Session()
saver.restore(sess1,'Model/lstm_2_80.ckpt')
train_data=[x_train[0] for _ in range(50)]
result=[]
for i in range(50):
    result.append(sess1.run(tf.matmul(outputs[i,:,:],weights)+biases,feed_dict={X_input:train_data}))

