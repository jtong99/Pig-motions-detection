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
import math
import pathlib as Path
from datetime import datetime
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

def time(millis):
    millis = int(millis)
    seconds=(millis/1000)%60
    seconds = int(seconds)
    minutes=(millis/(1000*60))%60
    minutes = int(minutes)
    hours=(millis/(1000*60*60))%24
    #real = str(int(hours)) +':'+str(minutes)+':'+str(seconds)
    real = '{:02}'.format(int(hours)) +':'+ '{:02}'.format(minutes)+':'+'{:02}'.format(seconds)
    return real


def display(working_folder, VIDEO_NAME):
    # Grab path to current working directory
    #os.chdir('E:\\nckh\\models-master\\research\\object_detection')
    CWD_PATH = working_folder

    # Path to frozen detection graph .pb file, which contains the model that is used
    # for object detection.
    PATH_TO_CKPT = os.path.join(CWD_PATH,'inference_graph','frozen_inference_graph.pb')

    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH,'training','object-detection.pbtxt')

    # Path to video
    PATH_TO_VIDEO = VIDEO_NAME

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
    #out = cv2.VideoWriter(FILE_OUTPUT, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                          #10, (frame_width, frame_height))
    frame_rate = video.get(5)
    frame_rate = math.floor(frame_rate)*5
    num_sitting = 0
    date = str(datetime.today().strftime('%Y-%m-%d || %H:%M:%S'))
    with open(os.path.join(CWD_PATH,'log//log_file'+'.txt'),'a') as file_w:
        file_w.write(date+'\n')
        file_w.write('Video name: '+VIDEO_NAME+'\n')
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
        frame_id = video.get(1)
        if frame_id % frame_rate == 0:
            mili = int(video.get(cv2.CAP_PROP_POS_MSEC))
            print(time(mili))
            for index,value in enumerate(classes[0]):
                if scores[0,index] >= 0.6:
                    #print(category_index.get(value).get('name'))
                    if category_index.get(value).get('name') == 'sitting pig':
                        print(num_sitting)
                        cv2.imwrite(os.path.join(CWD_PATH,'log//SITTING PIG {}.jpg'.format(str(num_sitting))),frame)
                        messagebox.showinfo('Detected', 'Detecting Sitting pig')
                        num_sitting = num_sitting + 1
                        with open(os.path.join(CWD_PATH,'log//log_file'+'.txt'),'a') as file_w:
                            file_w.write('sitting pig at: '+str(time(mili))+'\n')
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
    combo.set('')

    


def load_im(path):
    global img
    global ilabel
    global cl_button
    cl_button = Button(tab1, text="clear",command=clear)
    cl_button.place(x=50,y=357)
    i = PIL.Image.open(path)
    i = i.resize((500,270),PIL.Image.ANTIALIAS)
    #i.thumbnail((500,500))
    img = ImageTk.PhotoImage(i)
    ilabel = Label(tab1, image=img)
    ilabel.place(x=50, y=210, anchor="w")
    
def clear_video():
    global stop_vid
    stop_vid = True
    combo.set('')
    print('stop')
    
    

    
def load_vid(path):
    global stop_vid
    global video_label
    global stop
    video = imageio.get_reader(path)
    frame = 0
    stop = Button(tab1, text="Stop",command=clear_video)
    stop.place(x=100,y=370,anchor="e")
    video_label = Label(tab1)
    video_label.place(x=50, y=210, anchor="w")
    for image in video.iter_data():
        if stop_vid == True:
            break
        frame += 1
        image_frame = PIL.Image.fromarray(image)
        image_frame.thumbnail((500,500))
        try:
            frame_image = ImageTk.PhotoImage(image_frame)
            video_label.config(image=frame_image)
            video_label.image = frame_image
        except:
            sys.exit()
    video_label.config(image = "")
    stop.destroy()
    
    

       
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
        for i in lab_tuto:
            i.destroy()
        tuto_notice.destroy()
        cmd = subprocess.Popen('cmd.exe /K cd /d '+file2+'&& git clone https://github.com/tensorflow/models.git && exit')
        file2 += 'models/research/object_detection/'
        combo2['values']= (file2)
        combo2.current(0)
        cmd.wait()
        downprg = Progressbar(tab2, orient = HORIZONTAL,length=320)
        downprg.place(x=290, y=150, anchor="center")
        downprg.config(mode='determinate')
        downprg['value']=100
        

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
    global num_step
    num_step = entry_train.get()
    if combo2.get() == "":
        error()
    else:
        if num_step == "":
            messagebox.showinfo('error','Please add steps.')
        else:
            fname =file2+'/training/faster_rcnn_inception_v2_pets.config'
            s = '  num_steps: '+str(num_step)+'\n'
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
            cmd = subprocess.Popen('cmd.exe /K cd /d '+file2+'&& py train.py --logtostderr --train_dir='+file2+'/training/ --pipeline_config_path='+file2+'/training/faster_rcnn_inception_v2_pets.config && exit')
#E:\\nckh\\models-master\\research\\object_detection
obj_names = []

def start_change():
    #try:
    global ent
    z = 1
    for i in ent:
        with open(file2+'/training/'+obj_name_fill.get()+'.pbtxt','a') as file:
            file.write('item{\n'+'  id: '+str(z)+'\n'+'  name: "'+i.get()+'"\n'+'}\n')
        z += 1
    global obj_file
    num_objects = int(objects.get())
    print(num_objects)
    obj_file = obj_name_fill.get()
    file_label = file2 + 'training/'+obj_file+'.pbtxt'
    for e in ent:
        obj_names.append(e.get())
    fname =file2+'training/faster_rcnn_inception_v2_pets.config'
    num_classes = '    num_classes: '+str(num_objects)+'\n'
    label_path = '  label_map_path: '+'"'+file_label+'"\n'
    train = '    input_path: "'+file2+'data/train.record'+'"\n'
    test = '    input_path: "'+file2+'data/test.record"\n'
    with open(fname,'r') as file:
        data = file.readlines()
    f = open(fname,'r+')
    fread = f.readlines()
    x = 'num_steps:'
    a =[]
    i = 0
    data[122] = train
    data[134] = test
    with open(fname,'w') as file1:
        file1.writelines(data)
    for line in fread:
        i+=1
        if 'num_classes:' in line:
            data[i-1]= num_classes
        if 'label_map_path:' in line:
            data[i-1] = label_path
        with open(fname,'w') as file1:
            file1.writelines(data)
    #except:
        #messagebox.showinfo('ERROR! CANNOT ADD OBJECT','Please try again!!!')
    if os.path.isfile(file2+'training/'+obj_file+'.pbtxt'):
        messagebox.showinfo('SUCCESS','CREATED OBJECTS')
        wi.destroy()
    else:
        messagebox.showinfo('ERROR','ERROR OCCURED')
    
def change_class():
    if combo2.get() == "":
        messagebox.showinfo('ERROR! CANNOT CREATE ADD CLASS','Please fill the path of working folder!!!')
    else:       
        global wi
        wi = Toplevel()
        wi.grab_set()
        wi.focus()
        wi.geometry('250x300')
        wi.resizable(False, False)
        wi.title('Training')
        lb = Label(wi, text="Enter amount: ")
        lb.place(x=10,y=10)
        global objects
        objects = Entry(wi)
        objects.place(x=90,y=10)
        btn= ttk.Button(wi,text='OK',command=enter_object)
        btn.place(x=40,y=50)
        starting = ttk.Button(wi,text='Confirm',command=start_change)
        starting.place(x=140,y=50)
def enter_object():
    global objects
    num_objects = objects.get()
    if(num_objects == ''):
        messagebox.showinfo('EMPTY NUMBER OF OBJECT','Please enter NUMBER OF OBJECTS!')
    else:
        num_objects = int(num_objects)
        if(num_objects > 6):
            messagebox.showinfo('MAX CLASS REACH!','Please enter less than 6 classes!')
        else:
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
            for i in range(num_objects):
                name = 'Object ' + str(z+1) +':'
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
        for fname in os.listdir(file2+'/data/'):
            if fname.endswith('.csv'):
                messagebox.showinfo('ERROR! CANNOT CREATE CSV FILE','File CSV exist in folder DATA!!!')
                break
        else:
            csv.create_csv(file2)
            messagebox.showinfo('success','Successfully created csv!')
def create_record():
    if combo2.get() == "":
        messagebox.showinfo('ERROR! CANNOT CREATE RECORD FILE','Please fill the path of working folder!!!')
    else:
        generate_tfrecord2.export_train_record(file2)
        generate_tfrecord2.export_test_record(file2)
        messagebox.showinfo('success','Successfully created records!')

def divide_img():
    flag = 0
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
                flag = 1
            else:
                messagebox.showinfo('ERROR','Folder image train and test is not EMPTY!')
        else:
            messagebox.showinfo('ERROR','No folder image! Please click CREATE DIR or create folder image!')
    if(flag == 1):
        divprg = Progressbar(tab2, orient = HORIZONTAL,length=320)
        divprg.place(x=290, y=250, anchor="center")
        divprg.config(mode='determinate')
        divprg['value']=100

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
            os.makedirs(file2 + '//log')
            cp.copy('training//faster_rcnn_inception_v2_pets.config',file2+'//training//')
            #cp.copy('training//object-detection.pbtxt',file2+'//training//')
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
    

def generate_graph():
    global num_step
    cmd = subprocess.Popen('cmd.exe /K cd /d '+file2+'&& py export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-'+str(num_step)+' --output_directory inference_graph && exit')
    messagebox.showinfo('SUCCESS','Created inference graph!!!!')
    
if __name__ == "__main__":
    #MODEL_NAME = 'C:\\Users\\Shan Tong\\Desktop\\NC-TONG HOP\\train 7_latest\\inference_graph'
    MODEL_NAME = ''
    global window
    global downprg
    global combo
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
    stop_vid = False
    combo = Combobox(tab1)
    combo.configure(width = 50)
    lab1 = Label(tab1, text="File Path: ")
    lab1.place(x=50, y=50, anchor="w")

    combo2 = Combobox(tab2)
    combo2.configure(width = 50)
    lab2 = Label(tab2, text="Working folder: ")
    lab2.place(x=40, y=50, anchor="w")
    tutorial = ["1. Download TF object detection folder","2. Browse the working PATH","3. Create neccessary directory","4. Add objects you want to train","5. Divide image at 8:2","6. Create CSV, RECORDS file and start training","7. Generate inference graph"]
    lab_tuto = []
    
    for i in tutorial:
        lab_tuto.append(Label(tab2,text=i,font="Helvetica 16 bold italic"))
    tuto_y = 70
    for z in lab_tuto:
        z.place(x=20,y=tuto_y)
        tuto_y+=40
    global tuto_notice
    tuto_notice = Label(tab2, text="(If you have your own folder please skip it and go to step 2)",font="Helvetica 10 italic")
    tuto_notice.place(x=30, y=94)

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
    createCSV.place(x=230, y=359, anchor="e")

    trainRECORD = Button(tab2, text='Create records',command= create_record )
    trainRECORD.place(x=350, y=359, anchor="e")


    train = Button(tab2, text='Start Training',command= train)
    train.place(x=453, y=359, anchor="e")
    
    down = Button(tab2, text='Download',command= down)
    down.place(x=553, y=150, anchor="e")
    
    down = Button(tab2, text='Create dir',command= create_dir)
    down.place(x=553, y=200, anchor="e")
    #x=553, y=200
    divide = Button(tab2, text='Divide img',command= divide_img)
    divide.place(x=553, y=250, anchor="e")

    choose_graph = Button(tab1, text='Choose model',command=model)
    choose_graph.place(x=350,y=359)

    change_class = Button(tab2, text='Add objects',command=change_class)
    change_class.place(x=40, y=359, anchor="w")

    create_graph = Button(tab2, text='Generate graph',command=generate_graph)
    create_graph.place(x=473, y=346)


    window.mainloop()
