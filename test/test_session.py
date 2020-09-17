import tensorflow as tf
 
def activation(e, f, g):
 
  return e + f + g
 
with tf.Graph().as_default():
  a = tf.constant([5, 4, 5], name='a')
  b = tf.constant([0, 1, 2], name='b')
  c = tf.constant([5, 0, 5], name='c')
 
  res = activation(a, b, c)
 
  init = tf.initialize_all_variables()
 
  with tf.Session() as sess:
	  # Start running operations on the Graph.
	  sess.run(init)
	  hi = sess.run(res)
	  print(hi)
