"""
Usage:
  # From tensorflow/models/
  # Create train data:
  py generate_tfrecord.py --csv_input=data/train_labels.csv --output_path=data/train.record --image_dir=ha/train

  # Create test data:
  py generate_tfrecord.py --csv_input=data/test_labels.csv --output_path=data/test.record --image_dir=ha/test
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf
from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('image_dir', '', 'Path to images')
FLAGS = flags.FLAGS
csv_input_train = 'data//train_labels.csv'
image_dir_train = 'ha//train'
output_path_train ='data//train.record'

csv_input_test = 'data//test_labels.csv'
image_dir_test = 'ha//test'
output_path_test ='data//test.record'
# TO-DO replace this with label map
'''
def class_text_to_int(row_label):
    if row_label == 'sitting pig':
        return 1
    elif row_label == 'lying pig':
        return 2
    elif row_label == 'standing pig':
        return 3
    elif row_label == 'part of pig':
        return 4
    elif row_label == 'multi pig':
        return 5
    else:
        None'''


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    from application1 import class_text_to_int
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def export_train_record(path):
    writer = tf.python_io.TFRecordWriter(os.path.join(path,output_path_train))
    path_image = os.path.join(os.path.join(path,image_dir_train))
    examples = pd.read_csv(os.path.join(path,csv_input_train))
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path_image)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_train = os.path.join(path, output_path_train)
    print('Successfully created the TFRecords: {}'.format(output_train))
    
def export_test_record(path):
    writer = tf.python_io.TFRecordWriter(os.path.join(path,output_path_test))
    path_image = os.path.join(os.path.join(path,image_dir_test))
    examples = pd.read_csv(os.path.join(path,csv_input_test))
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path_image)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_test = os.path.join(path, output_path_test)
    print('Successfully created the TFRecords: {}'.format(output_test))

#export_train_record()
#export_test_record()
