import pickle
from oprs import fc, conv
import numpy as np

a = pickle.load(open('../MNIST_data/mnist.pkl','rb'), encoding='latin1')
train, valid, test = a
train_img, train_label = train
valid_img, valid_label = valid
test_img, test_label = test
"""
train_img = train_img.reshape(50000,1,28,28)
valid_img = valid_img.reshape(10000,1,28,28)
test_img = test_img.reshape(10000,1,28,28)
"""
num_examples = train_img.shape[0]
input_dims = train_img.shape[1]
output_dims = 10
epsilon = 1 
hidden_dims = 100

conv1 = conv(1, 3, (3,3), (1,1))

fc1 = fc(input_dims, hidden_dims)
fc2 = fc(hidden_dims, output_dims)

def sample(_bs):
    idx = np.random.choice(np.array(range(50000)),_bs)
    noise = np.random.normal(0,1e-4,(_bs, input_dims))
    return train_img[idx]+noise, train_label[idx]

def acc(img, label):
    #d = conv1.forward(img)
    #e = d.reshape(d.shape[0], -1)
    z1 = fc1.forward(img)
    z2 = fc2.forward(z1)
    scores = np.exp(z2)
    probs = scores / np.sum(scores)
    pred = np.argmax(probs, axis=1)
    error = (np.array(label) == pred).mean()
    return error

prev_acc = 0.0
bs = 10000
for i in range(1000000):
    sample_img, sample_label = sample(bs)
    #d = conv1.forward(sample_img)
    #e = d.reshape(d.shape[0], -1)
    z1 = fc1.forward(sample_img)
    z2 = fc2.forward(z1)
    z2 = z2 - np.max(z2, axis=1, keepdims=True)
    exp_scores = np.exp(z2)
    probs = exp_scores/np.sum(exp_scores, axis=1, keepdims=True)

    delta3 = probs
    delta3[range(bs), np.array(sample_label)] -= 1
    delta2 = fc2.backprop(z1, delta3, epsilon)
    delta1 = fc1.backprop(sample_img, delta2, epsilon)
    #delta1 = delta1.reshape(d.shape)
    #delta0 = conv1.backprop(sample_img, delta1, epsilon*1e-2)
    acc1 = acc(valid_img[:1024], valid_label[:1024])
    acc0 = acc(sample_img, sample_label)
    print(i, epsilon, 'train:{}, valid:{}'.format(acc0, acc1))
    if acc1 - prev_acc < 1e-6:
        epsilon *= 0.9
    if epsilon < 1e-3:
        break
    prev_acc = acc1

acc2 = acc(test_img, test_label)
print(acc2)
