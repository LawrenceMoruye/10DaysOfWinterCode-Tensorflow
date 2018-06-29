import tensorflow as tf
import matplotlib.pyplot as plt
import os
import matplotlib.image as mp_img

filename ="./mypic.JPG"
image=mp_img.imread(filename)
print("image shape",image.shape)
print("image array",image)
plt.imshow(image)
plt.show()

x=tf.Variable(image,name="x")

init=tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	transpose=tf.image.transpose_image(x)

	result=sess.run(transpose)

	print("the shape is",result.shape)
	plt.imshow(result)
	plt.show()



