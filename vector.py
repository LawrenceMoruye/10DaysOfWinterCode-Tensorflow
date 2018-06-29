import tensorflow as tf
#y=wx+b
w=tf.constant([10,100],name="constant_w")

x=tf.placeholder(tf.int32,name="x")
b=tf.placeholder(tf.int32,name="b")

wx=tf.multiply(w,x,name="wx")
y=tf.add(wx,b,name="y")

with tf.Session() as sess:
	print("wx:",sess.run(wx,feed_dict={x:[5,500]}))
	print("y",sess.run(y,feed_dict={x:[5,500],b:[7,9]}))
	print("y using intermediate",sess.run(fetches=y,feed_dict={wx:[50,50000],b:[7,9]}))

	

writer=tf.summary.FileWriter("./m4_example4",sess.graph)
writer.close()