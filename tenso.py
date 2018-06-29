import tensorflow as tf
a=tf.constant(2.5,name="constant_a")
b=tf.constant(8.5,name="constant_b")
c=tf.constant(1.5,name="constant_c")
d=tf.constant(100.5,name="constant_d")

square=tf.square(a,name="square_a")
power=tf.pow(b,c,name="power_b_c")
squart=tf.sqrt(d,name="squaret_d")

final_sum=tf.add_n([square,power,squart],name="final_sum")
final_sum2=tf.add_n([a,b,c,power,squart],name="final_sum")

sess=tf.Session()
writer=tf.summary.FileWriter("./m2_example2",sess.graph)
print(sess.run(squart))
print(sess.run(power))
print(sess.run(square))
writer.close()
sess.close()

