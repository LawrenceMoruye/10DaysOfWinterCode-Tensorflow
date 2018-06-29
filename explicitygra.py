import tensorflow as tf
g1=tf.Graph()
with tf.Session() as sess:
    x=tf.placeholder(tf.int32,shape=[3],name="x")
    y=tf.placeholder(tf.int32,shape=[3],name="y")

    sum_x=tf.reduce_sum(x,name="sum_x")
    prod_y=tf.reduce_prod(y,name="prod_y")

    div_x_y=tf.div(sum_x,prod_y,name="div_x_y")
    final_mean=tf.reduce_mean([sum_x,prod_y],name="final_mean")

    print("sum of X:",sess.run(sum_x,feed_dict={x:[100,200,300]}))
    print("division of x and y",sess.run(div_x_y,feed_dict={x:[100,200,300],y:[1,2,3]}))
    print("the mean:",sess.run(final_mean,feed_dict={x:[100,200,300],y:[1,2,3]}))
    
    assert y.graph is g1