import tensorflow as tf

class Reader:

    def __init__(self, tfrecord_list):

        self.file_q = tf.train.string_input_producer(tfrecord_list)

    def read_and_decode(self):
        reader = tf.TFRecordReader()

        _, serialized_example = reader.read(self.file_q)

        features = tf.parse_single_example(
            serialized_example,
            features={
                #'height':      tf.FixedLenFeature([], tf.int64),
                #'width':       tf.FixedLenFeature([], tf.int64),
                'image':        tf.FixedLenFeature([], tf.string),
                'pose_q':       tf.FixedLenFeature([], tf.string),
                'pose_x':       tf.FixedLenFeature([], tf.string)
            })

        image = tf.decode_raw(features['image'], tf.float32)
        pose_q = tf.decode_raw(features['pose_q'], tf.float32)
        pose_x = tf.decode_raw(features['pose_x'], tf.float32)

        #height = tf.cast(features['height'], tf.int32)
        #width = tf.cast(features['width'], tf.int32)

        image = tf.reshape(image, (16, 9, 3))
        pose_q.set_shape((4))
        pose_x.set_shape((3))

        # Random transformations can be put here: right before you crop images
        # to predefined size. To get more information look at the stackoverflow
        # question linked above.
        '''
        image = tf.image.resize_images(image, size=[224, 224])

        image = tf.image.resize_image_with_crop_or_pad(image=image,
                                                       target_height=224,
                                                       target_width=224)
        '''

        image_batch , pose_q_batch , pose_x_batch = tf.train.shuffle_batch([image, pose_q, pose_x],
                                                                           batch_size=2,
                                                                           capacity=1024,
                                                                           num_threads=2,
                                                                           min_after_dequeue=10)

        return image_batch , pose_q_batch , pose_x_batch


# Files List
filename_train = [
    'dataset/dataset_train_2.tfrecords',
    'dataset/dataset_train_3.tfrecords',
    'dataset/dataset_train_4.tfrecords',
    'dataset/dataset_train_5.tfrecords',
    'dataset/dataset_train_6.tfrecords',
    'dataset/dataset_train_7.tfrecords',
    'dataset/dataset_train_8.tfrecords',
    'dataset/dataset_train_9.tfrecords']

# Create Reader Object
reader_eval = Reader(filename_train)

# Get Input Tensors
image, pose_q, pose_x = reader_eval.read_and_decode()

sess = tf.Session()

# Start Queue Threads
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord,sess=sess)



for i in range(0,1000):
    _image, _pose_q, _pose_x = sess.run([image, pose_q, pose_x])
    print 'idx:' , i
    print 'record:' , _pose_x , _pose_q




# 8 3 6 9 2 7 5 4 8 2 






