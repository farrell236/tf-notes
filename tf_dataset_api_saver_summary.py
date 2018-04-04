import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets import inception

def _parse_function(serialized_example):

    features = tf.parse_single_example(
        serialized_example,
        features={
            'height':       tf.FixedLenFeature([], tf.int64),
            'width':        tf.FixedLenFeature([], tf.int64),
            'image':        tf.FixedLenFeature([], tf.string),
            'pose':         tf.FixedLenFeature([], tf.string),
            'pose_q':       tf.FixedLenFeature([], tf.string),
            'pose_x':       tf.FixedLenFeature([], tf.string)
        })

    image   = tf.decode_raw(features['image'], tf.float32)
    pose    = tf.decode_raw(features['pose'], tf.float32)
    pose_q  = tf.decode_raw(features['pose_q'], tf.float32)
    pose_x  = tf.decode_raw(features['pose_x'], tf.float32)

    height  = tf.cast(features['height'], tf.int32)
    width   = tf.cast(features['width'], tf.int32)

    image   = tf.reshape(image, (height, width, 3))
    pose.set_shape((6))
    pose_q.set_shape((4))
    pose_x.set_shape((3))

    # Resize Image operations
    image = tf.image.resize_images(image, size=[224, 224])

    #image = tf.image.resize_image_with_crop_or_pad(image=image,
    #                                               target_height=224,
    #                                               target_width=224)

    return image , pose_q , pose_x


# List of TFRecords for training
training_filenames = [
    'dataset/dataset_train_2.tfrecords',
    'dataset/dataset_train_3.tfrecords',
    'dataset/dataset_train_4.tfrecords',
    'dataset/dataset_train_5.tfrecords',
    'dataset/dataset_train_6.tfrecords',
    'dataset/dataset_train_7.tfrecords',
    'dataset/dataset_train_8.tfrecords',
    'dataset/dataset_train_9.tfrecords']

# List of TFRecords for validation
validation_filenames = [
    'dataset/dataset_test_0.tfrecords',
    'dataset/dataset_test_1.tfrecords']


# Create training dataset with iterator
training_dataset    = tf.data.TFRecordDataset(training_filenames)
training_dataset    = training_dataset.map(_parse_function)  # Parse the record into tensors.
training_dataset    = training_dataset.repeat()  # Repeat the input indefinitely.
training_dataset    = training_dataset.batch(2)
train_iterator      = training_dataset.make_one_shot_iterator()

# Create validation dataset with iterator
validation_dataset  = tf.data.TFRecordDataset(validation_filenames)
validation_dataset  = validation_dataset.map(_parse_function)  # Parse the record into tensors.
validation_dataset  = validation_dataset.repeat()  # Repeat the input indefinitely.
validation_dataset  = validation_dataset.batch(1)
validation_iterator = validation_dataset.make_one_shot_iterator()

# Create a feedable iterator that use a placeholder to switch between dataset
handle       = tf.placeholder(tf.string)

iterator     = tf.contrib.data.Iterator.from_string_handle(
    handle, 
    train_iterator.output_types, 
    train_iterator.output_shapes)

image , pose_q , pose_x = iterator.get_next()

####################################################################################################
# Construct model and encapsulating all ops into scopes, making
# Tensorboard's Graph visualization more convenient

with tf.name_scope('Model'):
    py_x , _ = inception.inception_v3(image)

    py_x = tf.nn.relu(py_x)

    weights = {
        'h1': tf.Variable(tf.random_normal([1000, 4]),name='w_wpqr_out'),
        'h2': tf.Variable(tf.random_normal([1000, 3]),name='w_xyz_out'),
    }
    biases = {
        'b1': tf.Variable(tf.zeros([4]),name='b_wpqr_out'),
        'b2': tf.Variable(tf.zeros([3]),name='b_xyz_out'),
    }

    cls3_fc_pose_wpqr = tf.add(tf.matmul(py_x, weights['h1']), biases['b1'])
    cls3_fc_pose_xyz = tf.add(tf.matmul(py_x, weights['h2']), biases['b2'])

with tf.name_scope('Loss'):
    # Minimize error using weighted Euclidean Distance
    cost_wpqr = tf.reduce_mean(tf.squared_difference(cls3_fc_pose_wpqr, pose_q))
    cost_xyz = tf.reduce_mean(tf.squared_difference(cls3_fc_pose_xyz, pose_x))

    cost = tf.add(cost_wpqr,cost_xyz)

with tf.name_scope('Adam'):
    # Adam Optimiser
    train_op = tf.train.AdamOptimizer(1e-4).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

####################################################################################################
# Main Tensorflow training session

with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    train_iterator_handle       = sess.run(train_iterator.string_handle())
    validation_iterator_handle  = sess.run(validation_iterator.string_handle())
    
    tf.summary.scalar('Loss', cost)
    #tf.summary.tensor('cls3_fc_pose_wpqr', cls3_fc_pose_wpqr)
    #tf.summary.tensor('cls3_fc_pose_xyz', cls3_fc_pose_xyz)
    #tf.summary.tensor('pose_q', pose_q)
    #tf.summary.tensor('pose_x', pose_x)
    tf.summary.image('image',image)
    for var in tf.trainable_variables():
        tf.summary.histogram(var.name, var)
    summary_op = tf.summary.merge_all()
    summaries_dir = 'logs/'
    train_writer = tf.summary.FileWriter(summaries_dir + '/training', sess.graph)
    test_writer = tf.summary.FileWriter(summaries_dir + '/validation')

    # Create tf.train.Saver Object
    saver = tf.train.Saver()
    
    for i in range(200000):

        _, _cost, summary = sess.run([train_op, cost, summary_op], feed_dict={handle: train_iterator_handle})
        train_writer.add_summary(summary, i)

        print 'training: ' , i , 'cost: ' , _cost

        if i % 10 == 0:

            # Save Model
            save_path = saver.save(sess, 'model/model'+str(i)+'.ckpt')

            _cost, summary = sess.run([cost, summary_op], feed_dict={handle: validation_iterator_handle})                
            test_writer.add_summary(summary, i)
    
            print 'testing:', i , 'cost: ' , _cost

