import tensorflow as tf
# import dataset 'mnist'
from tensorflow.examples.tutorials.mnist import input_data
mnist_ds = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Define parameters
iterations = 50
learning_rate = 0.001
img_w = img_h = 28
batch_size = 100
display_step = 2

# Calc total # of pixels in each img.
num_pixels = img_w * img_h

# X = input parameters. (X equals # of pixels in this case)
X = tf.placeholder(tf.float32, [None, num_pixels])

# Y = output parameters. (Y equals # of numbers we want to classify)
Y = tf.placeholder(tf.float32, [None, 10])

# Init weight matrix by random ~ normal distribution
w = tf.Variable(tf.random_normal([num_pixels, 10], stddev = 0.25), name = "weights")

# Init bias matrix by random ~ normal distribution
b = tf.Variable(tf.random_normal([10], stddev = 0.25), name = "biases")

# Init our softmax model:
#
# Y_ = softmax( X * W + b)
#              X: matrix for 100 grayscale images of 28x28 pixels, flattened (there are 100 images in a mini-batch)
#              W: weight matrix with 784 lines and 10 columns
#              b: bias vector with 10 dimensions
#              +: add with broadcasting: adds the vector to each line of the matrix (numpy)
#              softmax(matrix) applies softmax on each line
#              softmax(line) applies an exp to each value then divides by the norm of the resulting line
#              Y_: output matrix with 100 lines and 10 columns
Y_ = tf.nn.softmax(tf.matmul(X, w) + b)


# loss function: cross-entropy = - sum( Yi * log(Y_i) )
#                           Y_: the computed output vector
#                           Y: the desired output vector
# cross-entropy
# log takes the log of each element, * multiplies the tensors element by element
# reduce_mean will add all the components in the tensor
# so here we end up with the total cross-entropy for all images in the batch
loss_function = -tf.reduce_sum(Y * tf.log(Y_))

# Training step
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)

# init
init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	for i in range(iterations):
		avg_loss = 0.
		total_batch = int(mnist_ds.train.num_examples/batch_size)

		for j in range(total_batch):
			# Fetch training data from mnist dataset
			batch_X, batch_Y = mnist_ds.train.next_batch(batch_size)

			# Fit training model using batch data
			sess.run(optimizer, feed_dict = {X: batch_X, Y: batch_Y})

			# Compute the average loss
			avg_loss += sess.run(loss_function, feed_dict = {X: batch_X, Y: batch_Y})/total_batch

		if i % display_step == 0:
			print("Iteration: ", '%04d' % (i + 1), "loss=", "{:.9f}".format(avg_loss))

	print("Training Complete!")

	# Start testing
	correct_prediction = tf.equal(tf.argmax(Y_, 1), tf.argmax(Y, 1))

	# Calc accuracy
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print("Accuracy:", accuracy.eval({X: mnist_ds.test.images, Y: mnist_ds.test.labels}))



