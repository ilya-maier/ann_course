import tensorflow as tf
import matplotlib.pyplot as plt


# just a simple script that helps to understand core concepts of NNs such as:
# loss function, gradient, learning rate and epochs

def make_noisy_data(w=0.1, b=0.5, n=100):
    uniform = tf.random.uniform((n,))
    noise = tf.random.normal((n,), stddev=0.1)
    data = w * uniform + b + noise
    # in order to understand what uniform and normal functions are doing:
    # plt.plot(uniform, 'g.')
    # plt.plot(noise, 'ro')
    plt.plot(uniform, data, 'bo')
    plt.plot(uniform, data - noise)
    # plt.show()
    return uniform, data


x, y = make_noisy_data()
w = tf.Variable(0.)
b = tf.Variable(0.)


# returns a predict result using a weight and a bias
def predict():
    return w * x + b


# calculates the error -> loss function
def mean_squared_error(y_pred, y_true):
    return tf.reduce_mean(tf.square(y_pred - y_true))


epochs = 200
learning_rate = 0.1

for i in range(epochs):
    # forward propagation
    with tf.GradientTape() as tape:
        predictions = predict()
        loss = mean_squared_error(y_pred=predictions, y_true=y)

    # backpropagation
    gradients = tape.gradient(target=loss, sources=[w, b])

    # changing the weight and bias
    w.assign_sub(gradients[0] * learning_rate)
    b.assign_sub(gradients[1] * learning_rate)

    if i == 50:
        learning_rate -= 0.01

    print(f"After {i} steps we learned that w={w.numpy()} and b={b.numpy()}")

plt.plot(x, predict(), 'r-')
plt.show()
