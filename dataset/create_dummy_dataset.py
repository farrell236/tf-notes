import tensorflow as tf
import numpy as np


# Tensorflow feature wrapper function
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


# Create dummy databases
#
# Each element of the matrix/vector will have form X.Y
# X : TFRecord number
# Y : example number in TFRecord

n_database = 10
n_elements = 100

for i in range(0, n_database):

    # Dummy databases 0,1 are used for validation,
    # remaining databases will be used for training
    if i in [0, 1]:
        prefix = 'dataset_test_'
    else:
        prefix = 'dataset_train_'

    fname = prefix + str(i) + '.tfrecords'
    print 'creating: ' , fname

    # Create TFRecordWriter object
    writer = tf.python_io.TFRecordWriter(fname)

    for j in range(0, n_elements):

        # Each TFRecord is comprised of 4 elements:
        # X:        Dummy RGB image     ; dimension: 16x9x3 ; datatype: float32
        # pose:     SE3 vector          ; dimension: 1x6    ; datatype: float32
        # pose_q:   Quaternion vector   ; dimension: 1x4    ; datatype: float32
        # pose_x:   Translation vector  ; dimension: 1x3    ; datatype: float32

        X           = np.ones([16, 9, 3]) * i + np.ones([16, 9, 3]) * j/n_elements
        pose        = np.ones(6) * i + np.ones(6) * j/n_elements
        pose_q      = np.ones(4) * i + np.ones(4) * j/n_elements
        pose_x      = np.ones(3) * i + np.ones(3) * j/n_elements

        img_raw     = X.astype('float32').tostring()
        pose_raw    = pose.astype('float32').tostring()
        pose_q_raw  = pose_q.astype('float32').tostring()
        pose_x_raw  = pose_x.astype('float32').tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'height':   _int64_feature(X.shape[0]),
            'width':    _int64_feature(X.shape[1]),
            'image':    _bytes_feature(img_raw),
            'pose':     _bytes_feature(pose_raw),
            'pose_q':   _bytes_feature(pose_q_raw),
            'pose_x':   _bytes_feature(pose_x_raw)}))

        writer.write(example.SerializeToString())

    writer.close()
