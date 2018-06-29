import tensorflow as tf
#y=wx+b
w=tf.Variable([2.0,3.4],tf.float32,name="w")
b=tf.Variable([5.0,2.0],name="b")
x=tf.placeholder(tf.float32,name="x")
y=w*x+b
init=tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)


	print("value of y:",sess.run(y,feed_dict={x:[2.2,3.0]}))


number=tf.Variable(2)
multiplier=tf.Variable(1)

init=tf.global_variables_initializer()
result=number.assign(tf.multiply(number,multiplier))#number.assign multiplies and assigns answer to result


with tf.Session() as sess:

	sess.run(init)
	for i in range(10):

	   print("result of number*multiplier:",sess.run(result))
	   print("increament of multiplier by 1:",sess.run(multiplier.assign_add(1)))



   


