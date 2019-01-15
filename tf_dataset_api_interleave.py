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

# Create training dataset with iterator
files = tf.data.Dataset.list_files(training_filenames)
dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=8, block_length=2)
# cycle_length: number of files to interleave
# block_length: number of records in file
dataset = dataset.map(_parse_function)  # Parse the record into tensors.
dataset = dataset.repeat()  # Repeat the input indefinitely.
dataset = dataset.shuffle(10)  # Shuffle the dataset with 10 elements in buffer
dataset = dataset.batch(4)
example = dataset.make_one_shot_iterator().get_next()

sess = tf.Session()

for i in range(0,1000):
    out = sess.run(example)
    print(out[1],'\n')

