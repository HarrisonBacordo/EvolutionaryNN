import numpy as np
import tensorflow as tf
from


def generate_dataset():
    data = np.zeros((10000, 2))
    answers = list()
    for n in range(10000):
        rand = np.random.randint(0, 2, 2)
        data[n] = rand
        if 1 in rand and 0 in rand:
            answers.append(1)
        else:
            answers.append(0)
    return data, answers


input_layer = tf.placeholder(shape=[None, 2], dtype=tf.float64)
labels = tf.placeholder(shape=[None, 2], dtype=tf.int32)
dense_1 = tf.layers.dense(inputs=input_layer, units=20, activation=tf.nn.relu)
output = tf.layers.dense(inputs=dense_1, units=2)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=output))
optimizer = tf.train.GradientDescentOptimizer(.2).minimize(loss)


def main():
    features, answers = generate_dataset()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(features.size):
            feature = np.reshape(features[i], newshape=[1, 2])
            answer = sess.run(tf.one_hot(answers[i], depth=2))
            answer = np.reshape(answer, newshape=[1, 2])
            _, c = sess.run([optimizer, loss], feed_dict={input_layer: feature, labels: answer})
            print(c)


if __name__ == '__main__':
    main()