import tensorflow as tf
i=0
for example in tf.python_io.tf_record_iterator("landmark.tfrecord"):
    print(tf.train.Example.FromString(example))
    i+=1
    if(i>10):
    	break