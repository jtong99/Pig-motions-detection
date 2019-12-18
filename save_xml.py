import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import glob
from pathlib import Path
import xml.etree.cElementTree as ET
from PIL import Image



sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util

MODEL_NAME = 'inference_graph'
CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','object-detection.pbtxt')
#PATH_TO_IMAGE = os.path.join(CWD_PATH,'data1','*.jpg')
PATH_TO_IMAGE = glob.glob("E:\\nckh\\data_test_lan_1\\*.jpg")

NUM_CLASSES = 5

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

num_detections = detection_graph.get_tensor_by_name('num_detections:0')
k = 1
o = ['xmin','ymin','xmax','ymax']
for images in PATH_TO_IMAGE:
    name = os.path.splitext(os.path.basename(images))[0]
    y = 1
    image = cv2.imread(images)
    width = image.shape[1]
    height = image.shape[0]
    image_expanded = np.expand_dims(image, axis=0)
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})
    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.60
        )
    '''xml1 = Path('xml_train')
    xml = Path('xml_test')
    
    new2 = Path('sp_detected')'''
    new3 = Path('E:\\nckh\\train_latest')
    #os.makedirs(os.path.join(new,name))
    #path = os.path.join(new2,name)
    #cv2.imwrite(os.path.join(new3,name+'.jpg'),image)
    #new = Path('E:\\nckh\\sitting_folder')
    b = []
    order = []
    order_num = 0
    for index,value in enumerate(classes[0]):
        if scores[0,index] >= 0.6:
            b.append(category_index.get(value).get('name'))
            print(category_index.get(value).get('name'))
            #if category_index.get(value).get('name') == 'sitting pig':
                #order.append(order_num)
            #order_num += 1
            cv2.imwrite(os.path.join(new3,name+'.jpg'),image)
            print('write sp')
                #break
    
    a = []
    min_score_thresh=0.60
    true_boxes = boxes[0][scores[0] >= min_score_thresh]
    order_sit = 0
    for i in range(true_boxes.shape[0]):
        ymin = int(true_boxes[i,0]*height-30)
        xmin = int(true_boxes[i,1]*width)
        ymax = int(true_boxes[i,2]*height)
        xmax = int(true_boxes[i,3]*width)
        
        print('write detect jpg')
        a.append([xmin,ymin,xmax,ymax])
        roi = image[ymin:ymax,xmin:xmax].copy()
        for i in order:
            if order_sit == i:
                cv2.imwrite(os.path.join(new,"box_{}.jpg".format(str(y))), roi)
        y = y + 1
        order_sit += 1
    image_path = Path(images)
    
    
    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = str(image_path.parent.name)
    ET.SubElement(annotation, 'filename').text = str(image_path.name)
    ET.SubElement(annotation, 'path').text = str(image_path)

    source = ET.SubElement(annotation, 'source')
    ET.SubElement(source, 'database').text = 'Unknown'

    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str (image.shape[1])
    ET.SubElement(size, 'height').text = str(image.shape[0])
    ET.SubElement(size, 'depth').text = str(image.shape[2])

    ET.SubElement(annotation, 'segmented').text = '0'

    for j in range(len(a)):
        object = ET.SubElement(annotation, 'object')
        ET.SubElement(object, 'name').text = str(b[j])
        ET.SubElement(object, 'pose').text = 'Unspecified'
        ET.SubElement(object, 'truncated').text = '0'
        ET.SubElement(object, 'difficult').text = '0'
        
        
        bndbox = ET.SubElement(object, 'bndbox')
        for y in range(4):
            ET.SubElement(bndbox, str(o[y])).text = str(a[j][y])
        

    tree = ET.ElementTree(annotation)
    #xml_file_name = 'image_test//image_{}'.format(str(k))+'image_{}'.format(str(k))+'.xml'
    tree.write(os.path.join(new3,name+'.xml'))
    #tree.write(os.path.join(xml1,name+'.xml'))
    #print('write xml',k)
    print('writting',k)
    k = k + 1

    
    
    





