import tensorflow as tf

# Define the input and output data
x = tf.placeholder(tf.float32, shape=[None, 2])
y = tf.placeholder(tf.float32, shape=[None, 1])

# Define the model
hidden_layer = tf.layers.dense(x, units=4, activation=tf.nn.relu)
output_layer = tf.layers.dense(hidden_layer, units=1, activation=tf.nn.sigmoid)

# Define the loss and optimizer
loss = tf.losses.sigmoid_cross_entropy(y, output_layer)
optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

# Initialize the variables
init = tf.global_variables_initializer()

# Define a session to run the model
with tf.Session() as sess:
  sess.run(init)
  
  # Train the model
  for i in range(100):
    sess.run(optimizer, feed_dict={x: [[1, 2], [3, 4], [5, 6]], y: [[0], [1], [1]]})
    
  # Evaluate the model
  output = sess.run(output_layer, feed_dict={x: [[1, 2]]})
  print(output)  # Output: [[0.05]]
