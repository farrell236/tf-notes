import tensorflow as tf


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
    'dataset_train_2.tfrecords',
    'dataset_train_3.tfrecords',
    'dataset_train_4.tfrecords',
    'dataset_train_5.tfrecords',
    'dataset_train_6.tfrecords',
    'dataset_train_7.tfrecords',
    'dataset_train_8.tfrecords',
    'dataset_train_9.tfrecords']

# List of TFRecords for validation
validation_filenames = [
    'dataset_test_0.tfrecords',
    'dataset_test_1.tfrecords']


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

next_element = iterator.get_next()

sess = tf.Session()

train_iterator_handle       = sess.run(train_iterator.string_handle())
validation_iterator_handle  = sess.run(validation_iterator.string_handle())

####################################################################################################
# Example

image   = next_element[0]
pose_q  = next_element[1]
pose_x  = next_element[2]

# Example graph with input elements
f = tf.multiply(pose_x,2)

for i in range(0,10):
    print 'idx:' , i , sess.run(f, feed_dict={handle: train_iterator_handle})

for i in range(0,20):
    print 'idx:' , i , sess.run(f, feed_dict={handle: validation_iterator_handle})

for i in range(0,20):
    print 'idx:' , i , sess.run(f, feed_dict={handle: train_iterator_handle})

for i in range(0,10):
    print 'idx:' , i , sess.run(f, feed_dict={handle: validation_iterator_handle})

####################################################################################################
# Scenario 1
# Train for 3 epochs, every epoch run 100 iters of training and 20 iters of validation

for _ in range(3):
    for _ in range(100):
        _ , _ , x = sess.run(next_element, feed_dict={handle: train_iterator_handle})
        #print 'train:' , x

    for _ in range(50):
        _ , _ , x = sess.run(next_element, feed_dict={handle: validation_iterator_handle})
        #print 'validation:' , x

####################################################################################################
# Scenario 2
# Train for 200000 iters, run 50 iters of validation every 100 iters of training

for i in range(200000):
    _ , _ , _ = sess.run(next_element, feed_dict={handle: train_iterator_handle})

    if i % 100 == 0:
        print 'testing'
        for _ in range(50):
            _ , _ , p_x = sess.run(next_element, feed_dict={handle: validation_iterator_handle})
            print 'val:' , p_x



