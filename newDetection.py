# importing required libraries
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtWebEngineWidgets import *
from PyQt5.QtPrintSupport import *
import os
import sys
import time
import numpy as np
import numpy as asarray
from detectionWindow import Ui_MainWindow
import darknet
from yoloCfgGenerator import generateYoloCfg
import cv2
from multiprocessing import Process, Queue, Value, freeze_support

CLASSES=[]

try:
    with open('classes.names','r') as names:
        for name in names:
            CLASSES.append(name.strip('\n'))
            clsNum = len(CLASSES)
except FileNotFoundError as e:
    print("classes.names Missing")

class Inference_Thread(QThread):
    updateDetectionBox=pyqtSignal(np.ndarray)
    updateOriginalBox=pyqtSignal(np.ndarray)
    updateFps=pyqtSignal(str)
    def __init__(self):
        super(Inference_Thread,self).__init__()
        self.count=0
        
        config_file=generateYoloCfg(clsNum)
        weights = "80class.weights"
        batch_size=1
        
        self.network, self.class_names, self.class_colors = darknet.load_network(config_file,clsNum,weights,batch_size)
        self.width = darknet.network_width(self.network)
        self.height = darknet.network_height(self.network)
        self.darknet_image = darknet.make_image(self.width, self.height, 3)
        darknet.detect_image(self.network, self.class_names, self.darknet_image, thresh=0.25) ##Predict first dummy##
        t = time.time()
        

    def started(self):
        prev_frame_time = 0
        new_frame_time = 0
        cap = cv2.VideoCapture(self.path)
        while(cap.isOpened()):
            ret, image = cap.read()
            hNew, wNew, cNew = image.shape
            image_resized = cv2.resize(image, (self.width, self.height),interpolation=cv2.INTER_LINEAR)
            hr,wr,cr = image_resized.shape # h, w, channel

            ratiosH = hNew/hr
            ratiosW = wNew/wr

            darknet.copy_image_from_bytes(self.darknet_image, image_resized.tobytes())
            bboxes = darknet.detect_image(self.network, self.class_names, self.darknet_image, thresh=0.25)
    
            self.updateOriginalBox.emit(image)

            image = darknet.draw_boxes(bboxes, image, self.class_colors, CLASSES, ratiosH, ratiosW, self.confVal/100)
            
            
            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time

            fps = int(fps)
            fps = str(fps)

            self.updateFps.emit(f'{fps} FPS')
            self.updateDetectionBox.emit(image)
            

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cap.release()

    def pathDir(self,path):
        self.path=path
    
    def startInference(self):
        self.started()
    
    def confidenceValue(self, val):
        self.confVal = val


# creating main window class
class MainWindow(QMainWindow):
    # constructor
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.horizontalSlider.setValue(70)
        self.ui.horizontalSlider.setMinimum(0)
        self.ui.horizontalSlider.setMaximum(100)
        self.ui.horizontalSlider.sliderMoved.connect(self.sliderMoving)

        self.inferenceThread=Inference_Thread()
        self.inferenceThread.updateDetectionBox.connect(self.updateDetectionBox)
        self.inferenceThread.updateOriginalBox.connect(self.updateOriginalBox)
        self.inferenceThread.updateFps.connect(self.updateFps)

        self.inferenceThread.confidenceValue(self.ui.horizontalSlider.value())
        self.ui.label_confidence.setText(str(self.ui.horizontalSlider.value()))

        #self.inferenceThread.start(priority=4)

        self.ui.btn_upload.clicked.connect(self.pick_dir)
        self.ui.btn_start.clicked.connect(self.inferenceThread.startInference)

    def updateFps(self,txt):
        self.ui.label_fps.setText(txt)

    def sliderMoving(self):
        self.inferenceThread.confidenceValue(self.ui.horizontalSlider.value())
        self.ui.label_confidence.setText(str(self.ui.horizontalSlider.value()))

    def updateOriginalBox(self,image):
        hNew, wNew, cNew = image.shape
        bytesPerLine = cNew * wNew
        convertToQtFormat = QImage(image.data, wNew, hNew, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        p = convertToQtFormat.scaled(wNew, hNew, Qt.KeepAspectRatio)

        self.ui.label_vid_ori.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ui.label_vid_ori.setScaledContents(True)
        self.ui.label_vid_ori.setPixmap(QPixmap.fromImage(p))


    def updateDetectionBox(self,image):
        hNew, wNew, cNew = image.shape
        bytesPerLine = cNew * wNew
        convertToQtFormat = QImage(image.data, wNew, hNew, bytesPerLine, QImage.Format_RGB888).rgbSwapped() #image.tobytes()
        p = convertToQtFormat.scaled(wNew, hNew, Qt.KeepAspectRatio)
        self.ui.label_video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ui.label_video.setScaledContents(True)
        self.ui.label_video.setPixmap(QPixmap.fromImage(p))

        

    def pick_dir(self):
        dialog = QFileDialog()
        #self.folder_path = dialog.getExistingDirectory(None, "Select Folder")
        path, checked = dialog.getOpenFileName(None, "QFileDialog.getOpenFileName()","", "All Files (*);;Python Files (*.py);;Text Files (*.txt)")
        self.inferenceThread.pathDir(path)
        self.ui.lineEdit.setText(path)


app = QApplication(sys.argv) # creating a pyQt5 application
app.setApplicationName("AI Detector on Web Browser") # setting name to the application
window = MainWindow() # creating a main window object
window.setGeometry(50,50,1920,1080)
window.showMaximized()
app.exec_() # loop


