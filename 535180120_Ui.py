import sys
import cv2
from PyQt5.QtGui import QImage

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QLabel, QMainWindow, QApplication, QWidget, QVBoxLayout, QListWidget, QListWidgetItem, QFileDialog
from PyQt5.QtGui import QPixmap, QTransform
from PyQt5.QtCore import Qt
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("C:/Users/tanuj/Downloads/model/500_epoch.h5", compile=False)
img = Image.open("C:/Users/tanuj/Downloads/Data gambar/dataset1_gambar crop sesuai produk yang paling tinggi dan lebar\cola\cola (12).JPG")
img = img.resize((224, 224), Image.ANTIALIAS)  # resize the image to 224x224

# Convert the image to a numpy array and preprocess it
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.

# Use the model to classify the image
model.predict(img_array)
global listWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):

        self.setGeometry(300, 300, 300, 300)
        self.setWindowTitle('Aplikasi')
        
        self.label_1 = QtWidgets.QLabel(self)
        self.label_1.move(100, 50)
        self.label_1.setGeometry(100,40,100,50)
        self.label_1.setAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_1.setFont(font)
        self.label_1.setWordWrap(True)
        self.label_1.setText("Program Rekognisi")
        
        self.button1 = QtWidgets.QPushButton('Browse Image', self)
        self.button1.move(100, 100)
        self.button1.clicked.connect(self.show_browse)

        self.button2 = QtWidgets.QPushButton('About', self)
        self.button2.move(100, 140)
        self.button2.clicked.connect(self.show_about)
        
        self.button3 = QtWidgets.QPushButton('Exit', self)
        self.button3.move(100, 180)
        self.button3.clicked.connect(self.close)

        self.w = None
        self.x = None
        self.y = None
        self.z = None
        
        #self.setCentralWidget(self.image_label)

        self.show()

    def show_about(self):
        if self.x is None:
            self.x = About()
        self.x.show()
    
        
    def show_browse(self):
        if self.z is None:
            self.z = BrowseImg()
        self.z.show()

        
class BrowseImg(QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(0, 0, 1000, 840)
        self.setWindowTitle('Browse Image')
        self.setMinimumSize(QtCore.QSize(1250, 800))
        self.setMaximumSize(QtCore.QSize(1250, 800))
        self.initUI()

    def showEvent(self, event):
        super().showEvent(event)
        self.browse_image()
        
    def initUI(self):
   
        self.label = QLabel()
        self.label.setText("Image Recognition")
        self.label.setGeometry(QtCore.QRect(0, 20, 1000, 41))
        font = QtGui.QFont()
        self.window1 = QWidget()
        self.window_layout = QVBoxLayout()
        global listWidget
        listWidget = QListWidget()
        
        self.window_layout.addWidget(listWidget)
        self.setLayout(self.window_layout)
        self.setWindowTitle('Image Viewer')
        self.label_2 = QtWidgets.QLabel(self)
        self.display_width = 750
        self.display_height = 720
        self.label_2.setGeometry(QtCore.QRect(
        20, 20, self.display_width, self.display_height))
        self.label_2.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.label_2.setLineWidth(0)
        self.label_2.setText("")
        self.label_2.resize(self.display_width, self.display_height)
        font.setFamily("Calibri")
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setAutoFillBackground(False)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        
        #self.exit=QAction("Exit Application",self)
        #self.exit.triggered.connect(self.exit_app)
        self.show()
        
    def exit_app(self):
        print("Shortcut pressed") #verification of shortcut press
        self.close()
        
    def detect(self,frame):
            pixmap = QPixmap(frame)
            qimage = pixmap.toImage()
            class_names = {1:'Abccup',2:'Aqua', 3:'Cimory',
               4:'Cocacola', 5:'Frisianflag', 6:'Indomie', 7:'Leminerale', 8:'Milo', 9:'Nescafe', 10:'Popmie', 11:'Pristine', 12:'Sarimi',
               13:'Sedap', 14:'Sedapcup', 15:'Ultramilk', }
            width = qimage.width()
            height = qimage.height()
            byte_string = qimage.bits().asstring(height * width * 4)
            np_image = np.frombuffer(byte_string, dtype=np.uint8).reshape((height, width, 4))
            np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)

            image_non_detection = np_image.copy()
            global listWidget
            img = Image.open(frame)
            img = img.resize((224, 224), Image.ANTIALIAS)  # resize the image to 224x224
            img = img.convert('RGB')

            # Convert the image to a numpy array and preprocess it
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.

# Use the model to classify the image
            prediction = model.predict(img_array)
            
# The prediction is an array with one element for each class. The index of the max element
# is the predicted class
            predicted_class = np.argmax(prediction) + 1  # Adding 1 because class indices start from 1

            #print(prediction)

            # Get the name of the predicted class
            predicted_class_name = class_names[predicted_class]

            listWidget.clear()
            headline2=  QListWidgetItem("== BARANG YANG DIKENALI ==")
            headline2.setTextAlignment(Qt.AlignRight)
            headline2.setFont(QtGui.QFont('Arial', 20))
            
            listWidget.addItem(headline2)
            item2 = QListWidgetItem(predicted_class_name)
            item2.setTextAlignment(Qt.AlignRight)
            item2.setFont(QtGui.QFont('Arial', 20))
            listWidget.addItem(item2)
            #return image_np_with_detections
            return image_non_detection


    def browse_image(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', '.', "Image files (*.jpg *.gif *.png)")
        if fname[0]:
            detected = self.detect(fname[0])
            #np_image = cv2.cvtColor(detected, cv2.COLOR_BGR2RGB)
            #np_image = cv2.cvtColor(np_image, cv2.COLOR_RGBA2RGB)
            height, width, _ = detected.shape
            qimage = QImage(detected.data, width, height, width * 3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            rotated_pixmap = pixmap.transformed(QTransform().rotate(0)) # Rotate the pixmap
            label_width = self.label_2.width()
            label_height = self.label_2.height()
            scaled_pixmap = rotated_pixmap.scaled(label_width, label_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.label_2.setPixmap(scaled_pixmap) 
            self.label_2.setAlignment(Qt.AlignCenter)
            

class About(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.setWindowTitle('Halaman Tentang')
        self.setGeometry(0, 0, 640, 360)
        self.setMinimumSize(QtCore.QSize(640, 525))
        self.setMaximumSize(QtCore.QSize(640, 640))

        # Label About Page
        self.label = QtWidgets.QLabel(self)
        self.label.setText("Tentang")
        self.label.setGeometry(QtCore.QRect(0, 10, 641, 41))
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setAutoFillBackground(False)
        self.label.setAlignment(QtCore.Qt.AlignCenter)

        # Label Box
        self.label_2 = QtWidgets.QLabel(self)
        self.label_2.setGeometry(QtCore.QRect(20, 70, 270, 261)) #(x,y,sizex,sizey)
        self.label_2.setFrameShape(QtWidgets.QFrame.Panel)
        self.label_2.setText("")
        self.label_2.setAlignment(
            QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

        # Label Box
        self.label_3 = QtWidgets.QLabel(self)
        self.label_3.setGeometry(QtCore.QRect(350, 70, 270, 450)) #(x,y,sizex,sizey)
        self.label_3.setFrameShape(QtWidgets.QFrame.Panel)
        self.label_3.setText("")
        self.label_3.setAlignment(
            QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

        # Label About
        self.label_4 = QtWidgets.QLabel(self)
        self.label_4.setGeometry(QtCore.QRect(20, 10, 261, 261))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_4.setFont(font)
        self.label_4.setWordWrap(True)
        self.label_4.setText(
            "Aplikasi ini adalah program pengenalan untuk produk makanan dan minuman yang dapat membantu dalam pengecekan stok barang yang kosong.")

        # Label Name
        self.label_5 = QtWidgets.QLabel(self)
        self.label_5.setGeometry(QtCore.QRect(20, 150, 261, 261))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_5.setFont(font)
        self.label_5.setText("Jika ada kendala hubungi:")

        self.label_6 = QtWidgets.QLabel(self)
        self.label_6.setGeometry(QtCore.QRect(20, 170, 261, 261))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_6.setFont(font)
        self.label_6.setText("085723228116 / Henry Tanujaya")
        
        # Label About1
        self.label_7 = QtWidgets.QLabel(self)
        self.label_7.setGeometry(QtCore.QRect(350, -50, 261, 261))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_7.setFont(font)
        self.label_7.setWordWrap(True)
        self.label_7.setText(
            "Cara menggunakan program :")
        
        # Label About2
        self.label_8 = QtWidgets.QLabel(self)
        self.label_8.setGeometry(QtCore.QRect(350, 50, 261, 261))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_8.setFont(font)
        self.label_8.setWordWrap(True)
        self.label_8.setText(
            "1. Tekan tombol Browse Image untuk memilih gambar produk yang ingin dideteksi")
        
        # Label About3
        self.label_9 = QtWidgets.QLabel(self)
        self.label_9.setGeometry(QtCore.QRect(350, 170, 261, 261))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_9.setFont(font)
        self.label_9.setWordWrap(True)
        self.label_9.setText(
            "2. Setelah memilih gambar produk maka tunggu sampai tampilan produk yang akan dideteksi oleh program")
        
        # Label About4
        self.label_10 = QtWidgets.QLabel(self)
        self.label_10.setGeometry(QtCore.QRect(350, 300, 261, 261))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_10.setFont(font)
        self.label_10.setWordWrap(True)
        self.label_10.setText(
            "3. Tekan tombol tentang untuk mengetahui penjelasan aplikasi dan cara menggunakan aplikasi, atau memiliki kendala saat menggunakan program")


def main():
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
