import PIL
import imageio
import threading
import sys
import Object_detection_image
import xml_to_csv2 as csv
import generate_tfrecord2
import cpfile
from PIL import Image, ImageTk
from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk
from functools import partial
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import subprocess
import shutil as cp
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util



def display(MODEL_NAME, VIDEO_NAME):
    # Grab path to current working directory
    os.chdir('E:\\nckh\\models-master\\research\\object_detection')
    CWD_PATH = os.getcwd()

    # Path to frozen detection graph .pb file, which contains the model that is used
    # for object detection.
    PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH,'training','object-detection.pbtxt')

    # Path to video
    PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME)

    # Number of classes the object detector can identify
    NUM_CLASSES = 5

    # Load the label map.
    # Label maps map indices to category names, so that when our convolution
    # network predicts `5`, we know that this corresponds to `king`.
    # Here we use internal utility functions, but anything that returns a
    # dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    # Define input and output tensors (i.e. data) for the object detection classifier

    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    FILE_OUTPUT = 'video_detected.avi'
    # Open video file
    video = cv2.VideoCapture(PATH_TO_VIDEO)
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    out = cv2.VideoWriter(FILE_OUTPUT, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                          10, (frame_width, frame_height))

    while(video.isOpened()):

        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        ret, frame = video.read()
        frame_expanded = np.expand_dims(frame, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        # Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.60)
        #if ret == True:
            #out.write(frame)
            #print('writting')
            
        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame)
       
        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'): 
            break
    # Clean up
    video.release()
    cv2.destroyAllWindows()
def clear():
    ilabel.config(image = "")
    cl_button.destroy()
    


def load_im(path):
    global img
    global ilabel
    global cl_button
    cl_button = Button(tab1, text="clear",command=clear)
    cl_button.place(x=50,y=357)
    i = PIL.Image.open(path)
    i.thumbnail((500,500))
    img = ImageTk.PhotoImage(i)
    ilabel = Label(tab1, image=img)
    ilabel.place(x=50, y=210, anchor="w")
    
def stop():
    global stop
    stop = True
    print('stop')
    vlabel.config(image = "")
    

    
def load_vid(path):
    global stop
    global frame_image
    global vlabel
    video = imageio.get_reader(path)
    frame = 0
    stop = Button(tab1, text="Stop",command=stop)
    stop.place(x=100,y=370,anchor="e")
    vlabel = Label(tab1)
    vlabel.place(x=50, y=210, anchor="w")
    for image in video.iter_data():
        frame += 1
        image_frame = PIL.Image.fromarray(image)
        image_frame.thumbnail((500,500))
        try:
            frame_image = ImageTk.PhotoImage(image_frame)
            vlabel.config(image=frame_image)
            vlabel.image = frame_image
            if stop == True:
                break
        except:
            sys.exit()
    
    

       
def browse():
    global file
    file = filedialog.askopenfilename(filetypes = (("Image files","*.jpg"),("Video files","*.mp4"),("All files","*.*")))
    combo['values']= (file)
    combo.current(0)
    if file.endswith(".jpg"):
        load_im(file)
    elif file.endswith(".mp4"):
        thread = threading.Thread(target=load_vid, args=(file,))
        thread.daemon = True
        thread.start()
    else:
        messagebox.showinfo('Error! File not supported!', 'This type of file is not supported!')

def browse2():
    global file2
    file2 = filedialog.askdirectory()
    combo2['values']= (file2)
    combo2.current(0)

def bye():
    w = Tk()
    w.geometry('400x150')
    w.after(2000, lambda: w.destroy()) # Destroy the widget after 30 seconds
    lb = Label(w, text=" THANK YOU FOR USING OUR APP ! ",font="Helvetica 16 bold italic")
    lb.pack()
    about = Label(w, text="Version v1.0\nMade with\n\t Tong Minh Duc\n\t Ha Khanh Quynh\n\t Nguyen Thi Xuan Tuoi\n\t Nguyen Phuc Hoang Long")
    about.place(x=20,y=80,anchor="w")
    w.mainloop()
def error():
    w = Tk()
    w.title("ERROR")
    w.geometry('400x150')
    #w.after(2000, lambda: w.destroy()) # Destroy the widget after 30 seconds
    error = Label(w, text="PLEASE BROWSE THE PATH\n OF IMAGE OR VIDEO!!!",font="Helvetica 16 bold italic")
    error.place(x=20,y=80,anchor="w")
    w.mainloop()
    
def close():
    try:
        window.destroy()
        sys.exit()
    except:
        bye()
def down():
    if combo2.get() == "":
        messagebox.showinfo('error','Please fill in the path of working folder!!!')
    else:
        global file2
        #cmd = subprocess.Popen('cmd.exe /K cd /d '+file2+'&& git clone https://github.com/tensorflow/models.git')
        file2 += 'models/research/object_detection/'
        combo2['values']= (file2)
        combo2.current(0)
    

def detect():
    if combo.get() == "":
        error()
    else:
        global MODEL_NAME
        global file
        if MODEL_NAME == "":
            MODEL_NAME = 'E:\\nckh\\models-master\\research\\object_detection\\inference_graph\\'
        if file.endswith(".jpg"):
            Object_detection_image.img_de(file,MODEL_NAME)
        else:
            #Object_detection_video.vid_de(file,MODEL_NAME)
            display(MODEL_NAME,file)


    
def train():
    if combo2.get() == "":
        messagebox.showinfo('ERROR! CANNOT CREATE RECORD FILE','Please fill the path of working folder!!!')
    else:
        w = Toplevel()
        w.geometry('250x100')
        w.title('training')
        lb = Label(w, text="Enter steps: ")
        lb.place(x=10,y=10)
        global entry_train
        entry_train = Entry(w)
        entry_train.place(x=90,y=10)  
        btn= ttk.Button(w,text='Start Training',command=change_step)
        btn.place(x=100,y=50)
        w.mainloop()
def ok():
        value = entry_train.get()
        print(value)
def change_step():
    n = entry_train.get()
    if combo2.get() == "":
        error()
    else:
        if n == "":
            messagebox.showinfo('error','Please add steps.')
        else:
            fname =file2+'/training/faster_rcnn_inception_v2_pets.config'
            s = '  num_steps: '+str(n)+'\n'
            with open(fname,'r') as file:
                data = file.readlines()
            f = open(fname,'r+')
            fread = f.readlines()
            x = 'num_steps:'
            a =[]
            i = 0
            for line in fread:
                i+=1
                if 'num_steps:' in line:
                    data[i-1]= s
                    print(i)
            with open(fname,'w') as file1:
                file1.writelines(data)
            messagebox.showinfo('success','Successfully added steps and Click OK to start training.')
            cmd = subprocess.Popen('cmd.exe /K cd /d '+file2+'&& py train.py --logtostderr --train_dir='+file2+'/training/ --pipeline_config_path='+file2+'/training/faster_rcnn_inception_v2_pets.config')
#E:\\nckh\\models-master\\research\\object_detection
obj_names = []   
def start_change():
    z = 1
    for i in ent:
        with open(file2+'/training/'+obj_name_fill.get()+'.pbtxt','a') as file:
            file.write('item{\n'+'  id: '+str(z)+'\n'+'  name: "'+i.get()+'"\n'+'}\n')
        z += 1
    global obj_file
    global num_object
    obj_file = obj_name_fill.get()
    file_label = file2 + 'training/'+obj_file+'.pbtxt'
    num_object = objects
    for e in ent:
        obj_names.append(e.get())
    fname =file2+'training/faster_rcnn_inception_v2_pets.config'
    num_classes = '    num_classes: '+str(objects)+'\n'
    label_path = '  label_map_path: '+'"'+file_label+'"\n'
    train = '    input_path: "'+file2+'data/train.record'+'"'
    test = '  input_path: "'+file2+'data/test.record'
    with open(fname,'r') as file:
        data = file.readlines()
    f = open(fname,'r+')
    fread = f.readlines()
    x = 'num_steps:'
    a =[]
    i = 0
    for line in fread:
        i+=1
        data[123] = train
        data[135] = test
        if 'num_classes:' in line:
            data[i-1]= num_classes
        if 'label_map_path:' in line:
            data[i-1] = label_path
        with open(fname,'w') as file1:
            file1.writelines(data)
    
def change_class():
    global wi
    wi = Toplevel()
    wi.geometry('250x500')
    wi.title('training')
    lb = Label(wi, text="Enter object: ")
    lb.place(x=10,y=10)
    global objects
    objects = Entry(wi)
    objects.place(x=90,y=10)
    btn= ttk.Button(wi,text='OK',command=display)
    btn.place(x=100,y=50)
    starting = ttk.Button(wi,text='start change',command=start_change)
    starting.place(x=100,y=300)
def display():
    global objects
    objects = objects.get()
    objects = int(objects)
    z = 0
    global ent
    ent = []
    lab = []
    le = 120
    le1 = 120
    obj_name = Label(wi,text='File name: ')
    obj_name.place(x=10,y=80)
    global obj_name_fill
    global o_name
    obj_name_fill = Entry(wi)
    obj_name_fill.place(x=90,y=80)
    for i in range(objects):
        name = 'object ' + str(z+1) +':'
        lab.append(Label(wi,text=name))
        ent.append(Entry(wi))
        z += 1
    for k in lab:
        k.place(x=10,y=le1)
        le1 = le1 + 30
    for i in ent:
        i.place(x = 90, y = le)
        le = le + 30
        
def create_csv():
    #global image
    if combo2.get() == "":
        messagebox.showinfo('ERROR! CANNOT CREATE CSV FILE','Please fill the path of working folder!!!')
    else:
        image = filedialog.askdirectory()
        for fname in os.listdir(file2+'/data/'):
            if fname.endswith('.csv'):
                messagebox.showinfo('ERROR! CANNOT CREATE CSV FILE','File CSV exist in folder DATA!!!')
                break
        else:
            csv.create_csv(image)
            messagebox.showinfo('success','Successfully created csv!')
def create_record():
    if combo2.get() == "":
        messagebox.showinfo('ERROR! CANNOT CREATE RECORD FILE','Please fill the path of working folder!!!')
    else:
        generate_tfrecord2.export_train_record(file2)
        generate_tfrecord2.export_test_record(file2)
        messagebox.showinfo('success','Successfully created records!')

def divide_img():
    if combo2.get() == "":
        messagebox.showinfo('ERROR! CANNOT DIVIDE IMAGE','Please fill the path of working folder!!!')
    else:
        dir_train = os.path.isdir(file2+'/ha/train')
        dir_test = os.path.isdir(file2+'/ha/test')
        if dir_train == True and dir_test == True:
            dir_img_train = os.listdir(file2+'/ha/train/')
            dir_img_test = os.listdir(file2+'/ha/test/')
            if len(dir_img_train) == 0 and len(dir_img_test) == 0:
                img = filedialog.askdirectory()
                cpfile.cp_img(img,file2+'/ha')
                messagebox.showinfo('success','Successfully divided img!')
            else:
                messagebox.showinfo('ERROR','Folder image train and test is not EMPTY!')
        else:
            messagebox.showinfo('ERROR','No folder image! Please click CREATE DIR or create folder image!')
def create_dir():
    if combo2.get() == "":
        messagebox.showinfo('error','Please fill in the path of working folder!!!')
    else:
        if os.path.isdir(file2+'/ha') and os.path.isdir(file2+'/inference_graph') and os.path.isdir(file2+'/data') and os.path.isdir(file2+'/training') and os.path.isfile(file2+'//training//faster_rcnn_inception_v2_pets.config') and os.path.isfile(file2+'//train.py') and os.path.isfile(file2+'//export_inference_graph.py'):
            messagebox.showinfo('exist','Data exist!! cannot create')
        else:
            os.makedirs(file2 + '//ha')
            os.makedirs(file2 + '//ha//train')
            os.makedirs(file2 + '//ha//test')
            os.makedirs(file2 + '//inference_graph')
            os.makedirs(file2 + '//data')
            os.makedirs(file2 + '//training')
            cp.copy('training//faster_rcnn_inception_v2_pets.config',file2+'//training//')
            cp.copy('training//object-detection.pbtxt',file2+'//training//')
            cp.copy('train.py',file2)
            cp.copy('export_inference_graph.py',file2)
            if os.path.isdir(file2+'/ha') and os.path.isdir(file2+'/inference_graph') and os.path.isdir(file2+'/data') and os.path.isdir(file2+'/training') and os.path.isfile(file2+'//training//faster_rcnn_inception_v2_pets.config') and os.path.isfile(file2+'//train.py') and os.path.isfile(file2+'//export_inference_graph.py'):
                messagebox.showinfo('success','Successfully created!')
            else:
                messagebox.showinfo('ERROR','Missing some file! Please check again!!!!')

def model():
    mod = filedialog.askdirectory()
    global MODEL_NAME
    MODEL_NAME = mod
def class_text_to_int(row_label):
    z = 1
    for i in obj_names:
        if row_label == i:
            return z
        z += 1
    return 0
    

def test_obj():
    global obj_names
    for j in obj_names:
        print(j)
    print(class_text_to_int('sitting pig'))
if __name__ == "__main__":
    #MODEL_NAME = 'C:\\Users\\Shan Tong\\Desktop\\NC-TONG HOP\\train 7_latest\\inference_graph'
    MODEL_NAME = ''
    global window
    window = Tk()
    window.title("Pigs Motions Detection & Training app")
    window.geometry('600x420')
    window.resizable(width=False, height=False)

    tab_control = Notebook(window)
    tab1 = Frame(tab_control)
    tab2 = Frame(tab_control)
    tab_control.add(tab1, text='Detect')
    tab_control.add(tab2, text='Train')
    lbl1 = Label(tab1, text=" ____ W E L C O M E ____ ")
    lbl1.place(x=300, y=25, anchor="center")
    lbl2 = Label(tab2, text=" ____ W E L C O M E ____ ")
    lbl2.place(x=300, y=25, anchor="center")
    tab_control.pack(expand=1, fill='both')
    stop = False
    combo = Combobox(tab1)
    combo.configure(width = 50)
    lab1 = Label(tab1, text="File Path: ")
    lab1.place(x=50, y=50, anchor="w")

    combo2 = Combobox(tab2)
    combo2.configure(width = 50)
    lab2 = Label(tab2, text="Working folder: ")
    lab2.place(x=40, y=50, anchor="w")



    
    btn = Button(tab1,text='Browse', command=browse)
    btn.place(x=553, y=50, anchor="e")
    sdetect = Button(tab1,text='Start Detecting', command=detect)
    sdetect.place(x=553, y=370, anchor="e")
    cancel = Button(tab1,text='Cancel',command=close)
    cancel.place(x=320, y=370, anchor="e")
    combo.place(x=290, y=50, anchor="center")
    combo2.place(x=290, y=50, anchor="center")
    btn2 = Button(tab2,text='Browse', command=browse2)
    btn2.place(x=553, y=50, anchor="e")

    createCSV = Button(tab2, text='Create CSV',command= create_csv)
    createCSV.place(x=100,y=100)

    trainRECORD = Button(tab2, text='Create records',command= create_record )
    trainRECORD.place(x=100,y=200)


    train = Button(tab2, text='Start Training',command= train)
    train.place(x=100,y=350)

    down = Button(tab2, text='Download',command= down)
    down.place(x=200,y=100)

    down = Button(tab2, text='Create dir',command= create_dir)
    down.place(x=400,y=100)

    divide = Button(tab2, text='Divide img',command= divide_img)
    divide.place(x=300,y=100)

    choose_graph = Button(tab1, text='Choose model',command=model)
    choose_graph.place(x=350,y=359)

    change_class = Button(tab2, text='Change class',command=change_class)
    change_class.place(x=400,y=359)

    teste = Button(tab2, text='test',command=test_obj)
    teste.place(x=400,y=270)


    window.mainloop()

