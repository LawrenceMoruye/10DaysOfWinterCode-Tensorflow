import tensorflow as tf

A=tf.constant([4],tf.int32,name="A")

B=tf.constant([5],tf.int32,name="B")

C=tf.constant([6],tf.int32,name="D")

x=tf.placeholder(tf.int32,name="x")

#y=Ax**2+Bx+c
with tf.name_scope("Equation1"):

	Ax2_1=tf.multiply(A,tf.pow(x,2),name="Ax2_1")
	Bx_1=tf.multiply(B,x,name="Bx_1")
	y1=tf.add_n([Ax2_1,Bx_1,C],name="y1")

#y2=Ax**+Bx**2
with tf.name_scope("Equation2"):

	Ax_2=tf.multiply(A,tf.pow(x,2),name="Ax_2")
	Bx_2=tf.multiply(B,tf.pow(x,2),name="Bx_2")
	y2=tf.add_n([Ax_2,Bx_2],name="y2")

with tf.name_scope("final_sum"):
	y=y1+y2

	

with tf.Session() as sess:
	print("value of y:",sess.run(y,feed_dict={x:[10]}))
	writer=tf.summary.FileWriter("./m5_example5",sess.graph)
	writer.close()