import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets import inception
from tensorflow.contrib.slim.python.slim.nets import alexnet

def my_input_fn(filenames):

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
    
        return image , pose

    # Create training dataset with iterator
    dataset    = tf.data.TFRecordDataset(filenames)
    dataset    = dataset.map(_parse_function)  # Parse the record into tensors.
    dataset    = dataset.repeat()  # Repeat the input indefinitely.
    dataset    = dataset.batch(2)
    iterator   = dataset.make_one_shot_iterator()
    
    return iterator.get_next()

####################################################################################################
# Construct model and encapsulating all ops into scopes, making
# Tensorboard's Graph visualization more convenient

def my_model_fn(
    features, # This is batch_features from input_fn
    labels,   # This is batch_labels from input_fn
    mode):    # And instance of tf.estimator.ModeKeys, see below

    print 'HIT! my_model_fn() CALLED!'

    if mode == tf.estimator.ModeKeys.PREDICT:
        tf.logging.info("my_model_fn: PREDICT, {}".format(mode))
    elif mode == tf.estimator.ModeKeys.EVAL:
        tf.logging.info("my_model_fn: EVAL, {}".format(mode))
    elif mode == tf.estimator.ModeKeys.TRAIN:
        tf.logging.info("my_model_fn: TRAIN, {}".format(mode))

    y_pred , _ = alexnet.alexnet_v2(features,num_classes=6,is_training=True)
    '''
    py_x = tf.nn.relu(py_x)
    
    weights = {
        'h1': tf.Variable(tf.random_normal([1000, 6]),name='w_pose'),
    }
    biases = {
        'b1': tf.Variable(tf.zeros([6]),name='b_pose'),
    }
    
    y_pred = tf.add(tf.matmul(py_x, weights['h1']), biases['b1'])
    '''

    print_out = tf.add(0.0,labels,name='print_out')

    # 1. Prediction mode
    # Return our prediction
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=y_pred)

    # Evaluation and Training mode

    # Calculate the loss
    loss = tf.reduce_mean(tf.squared_difference(y_pred, labels))

    # 2. Evaluation mode
    # Return our loss (which is used to evaluate our model)
    # Set the TensorBoard scalar my_accurace to the accuracy
    # Obs: This function only sets value during mode == ModeKeys.EVAL
    # To set values during training, see tf.summary.scalar
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode,
            loss=loss)

    # If mode is not PREDICT nor EVAL, then we must be in TRAIN
    assert mode == tf.estimator.ModeKeys.TRAIN, "TRAIN is only ModeKey left"

    # 3. Training mode

    # Default optimizer for DNNClassifier: Adagrad with learning rate=0.05
    # Our objective (train_op) is to minimize loss
    # Provide global step counter (used to count gradient updates)
    optimizer = tf.train.AdamOptimizer(1e-4)
    train_op = optimizer.minimize(
        loss,
        global_step=tf.train.get_global_step())

    # Set the TensorBoard scalar my_accuracy to the accuracy
    # Obs: This function only sets the value during mode == ModeKeys.TRAIN
    # To set values during evaluation, see eval_metrics_ops
    tf.summary.scalar('loss', loss)

    # Return training operations: loss and train_op
    return tf.estimator.EstimatorSpec(
        mode,
        loss=loss,
        train_op=train_op)


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

tensors_to_log = {"out_pose": "print_out"}
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=1)

# Create a custom estimator using my_model_fn to define the model
tf.logging.info("Before classifier construction")
classifier = tf.estimator.Estimator(
    model_fn=my_model_fn,
    model_dir='model/')  # Path to where checkpoints etc are stored
tf.logging.info("...done constructing classifier")


tf.logging.info("Before classifier.train")
classifier.train(
    input_fn=lambda: my_input_fn(training_filenames),
    steps=10,
    hooks=[logging_hook])
tf.logging.info("...done classifier.train")


tf.logging.info("Before classifier.evaluate")
classifier.evaluate(
    input_fn=lambda: my_input_fn(validation_filenames),
    steps=2)
tf.logging.info("...done classifier.evaluate")


tf.logging.info("Before classifier.train")
classifier.train(
    input_fn=lambda: my_input_fn(training_filenames),
    steps=10,
    hooks=[logging_hook])
tf.logging.info("...done classifier.train")

