from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QWidget, QHeaderView, QVBoxLayout, QPushButton, QHBoxLayout, QGridLayout, QTableWidget, QTableWidgetItem, QMenuBar, QAction, QMenu
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette,  QColor
from PyQt5.QtCore import QTimer, Qt
import cv2
import os
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from tensorflow.keras import  models

class VideoWidget(QWidget):
    def __init__(self, parent=None):
        super(VideoWidget, self).__init__(parent)
        self.video_label = QLabel(self)
        self.video_label.setGeometry(25, 25, 400, 400)

        layout = QGridLayout()
        layout.addWidget(self.video_label, 0, 0, 2, 1)

        self.square_labels = []  # Create a list to store the square labels
        for row in range(2):
            for col in range(2):
                square_label = QLabel(self)
                square_label.setStyleSheet('background-color: gray;')
                square_label.setFixedSize(75, 75)
                layout.addWidget(square_label, row, col+1)
                self.square_labels.append(square_label)

        self.setLayout(layout)

    def set_frame(self, frame):
        height, width, channel = frame.shape
        bytes_per_line = channel * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        q_pixmap = QPixmap.fromImage(q_image)
        q_pixmap = q_pixmap.scaled(self.video_label.size(), aspectRatioMode=True)
        self.video_label.setPixmap(q_pixmap)
        self.video_label.setScaledContents(True)
class ChartWindow(QMainWindow):
    def __init__(self, chart):
        super(ChartWindow, self).__init__()
        self.chart = chart
        self.setWindowTitle('Chart Details')

        self.canvas = FigureCanvas(self.chart)
        self.setCentralWidget(self.canvas)

    def showEvent(self, event):
        super(ChartWindow, self).showEvent(event)
        self.canvas.draw()
        
class ImageWindow(QMainWindow):
    def __init__(self, image_path):
        super(ImageWindow, self).__init__()
        self.setWindowTitle('Image Details')
        
        # Create a QLabel widget to display the image
        label = QLabel(self)
        pixmap = QPixmap(image_path)
        label.setPixmap(pixmap)
        
        # Set the QLabel widget as the central widget of the window
        self.setCentralWidget(label)

class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setGeometry(100, 100, 800, 500)
        self.setWindowTitle('Seed Classifier')
        self.figure = plt.figure()  # Create a figure for the pie chart
        self.pie_chart = self.figure.add_subplot(111)  # Add a subplot for the pie chart


        self.video_widget = VideoWidget(self)
        self.start_button = QPushButton('Start Stream', self)
        self.stop_button = QPushButton('Stop Stream', self)
        self.snap_button = QPushButton('Snap Picture', self)
        self.stop_button.setEnabled(False)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.snap_button)

        main_layout = QHBoxLayout()  # Main layout containing video layout and rectangles layout
        menu_bar = QMenuBar(self)  # Create a menu bar
        menu_bar.setFixedHeight(30)
        # Create main menu items
        project_menu = menu_bar.addMenu('Project')
        window_menu = menu_bar.addMenu('Window')
        option_menu = menu_bar.addMenu('Option')
        help_menu = menu_bar.addMenu('Help')

        # Add actions to Project menu
        project_action = QAction('Open Project', self)
        project_menu.addAction(project_action)

        # Add actions to Window menu
        window_action = QAction('New Window', self)
        window_menu.addAction(window_action)

        # Add actions to Option menu
        option_action = QAction('Settings', self)
        option_menu.addAction(option_action)

        # Add actions to Help menu
        help_action = QAction('About', self)
        help_menu.addAction(help_action)

        # Connect the actions to their respective slots
        project_action.triggered.connect(self.open_project)
        window_action.triggered.connect(self.new_window)
        option_action.triggered.connect(self.show_settings)
        help_action.triggered.connect(self.show_about)
        
        # Video layout
        video_layout = QVBoxLayout()
        video_layout.addWidget(menu_bar)
        video_layout.addWidget(self.video_widget)
        video_layout.addLayout(button_layout)
        # Rectangles layout
        rectangles_layout = QVBoxLayout()
        rectangles_layout.setContentsMargins(10, 0, 10, 0)  # Add left and right margins
        # First rectangle with table
        rect1 = QLabel(self)
        rect1.setStyleSheet("background-color: white; border: 1px solid black;")  # Customize the rectangle
        rect1.setFixedSize(275, 200)

        # Create a table widget
        table_widget = QTableWidget(4, 2)
        table_widget.setHorizontalHeaderLabels(['Category', 'Found'])
        values = []  # Store the values from the table
        labels = []  # Store the labels for the pie chart
        self.germinated = 0
        self.abnormal = 0
        self.ungerminated = 0
        self.total = 0
        text1 = "Germinated = 0\nAbnormal = 0\nUngerminated = 0\nTotal = 0"  # Text content
        variables = text1.split('\n')
        rows = text1.split("\n")
        for index, variable in enumerate(variables):
            var_name, var_value = variable.split('=')
            item_var_name = QTableWidgetItem(var_name.strip())
            item_var_value = QTableWidgetItem(var_value.strip())
            table_widget.setItem(index, 0, item_var_name)
            table_widget.setItem(index, 1, item_var_value)
            item_var_value.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            if var_name.strip() != 'Total':
                values.append(float(var_value.strip()))
                labels.append(var_name.strip())
        # Configure the table widget
        table_widget.setEditTriggers(QTableWidget.NoEditTriggers)  # Disable editing

        # Resize the font size to fit the content
        font = table_widget.font()
        font.setPointSize(8)  # Initial font size
        table_widget.setFont(font)

        table_widget.resizeColumnsToContents()  # Resize columns to fit the content
        # Increase column width by 30 pixels
        variable_column = 0
        current_width = table_widget.columnWidth(variable_column)
        table_widget.setColumnWidth(variable_column, current_width + 30)

        # Remove the third blank column
        blank_column = 2
        table_widget.removeColumn(blank_column)
       # Reduce row height by 30 pixels
        row_height = table_widget.rowHeight(0)
        table_widget.setRowHeight(0, row_height - 10)
        table_widget.setRowHeight(1, row_height - 10)
        table_widget.setRowHeight(2, row_height - 10)
        table_widget.setRowHeight(3, row_height - 10)
        table_widget.setRowHeight(4, row_height - 10)
        # Change background color of header to light gray
        palette = table_widget.horizontalHeader().palette()
        palette.setColor(QPalette.Background, QColor(220, 220, 200))
        table_widget.horizontalHeader().setPalette(palette)
        table_widget.horizontalHeader().setAutoFillBackground(True)

        # Make the last row bold
        font = table_widget.item(table_widget.rowCount() - 1, 0).font()
        font.setBold(True)
        for col in range(table_widget.columnCount()):
            item = table_widget.item(table_widget.rowCount() - 1, col)
            item.setFont(font)
        
        # Adjust the font size to fit the content within the table widget
        while table_widget.horizontalHeader().sectionSize(0) > table_widget.columnWidth(0) or \
                table_widget.horizontalHeader().sectionSize(1) > table_widget.columnWidth(1):
            font.setPointSize(font.pointSize() - 1)
            table_widget.setFont(font)
            table_widget.resizeColumnsToContents()

        # Adjust the header size to fit the content
        #table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

        # Add the table widget to the rectangle
        rect1_layout = QVBoxLayout()
        rect1_layout.addWidget(table_widget)
        rect1.setLayout(rect1_layout)

  
        # Second rectangle with pie chart
        self.rect2 = QLabel(self)
        self.rect2.setStyleSheet("background-color: white; border: 1px solid black;")  # Customize the rectangle
        self.rect2.setFixedSize(275, 200)
        try:
            # Add pie chart plotting here
            # Plot the values column as a pie chart
            self.pie_chart.clear()  # Clear the previous chart
            self.pie_chart.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
            self.pie_chart.axis('equal')  # Ensure the chart is circular
            # Increase the font size of the chart labels
            self.pie_chart.legend(fontsize='large')
            # Convert the plot to an image and display it in rect2
            self.figure.savefig('pie_chart.png')
            pixmap = QPixmap('pie_chart.png')
            self.rect2.setPixmap(pixmap)
            self.rect2.setScaledContents(True)
            self.rect2.mousePressEvent = self.show_chart_window
        except:
            pass
        # Add the rectangles to the rectangles layout
        rectangles_layout.addWidget(rect1)
        rectangles_layout.addWidget(self.rect2)

        
        # Add the video layout and rectangles layout to the main layout
        main_layout.addLayout(video_layout)
        main_layout.addLayout(rectangles_layout)
        self.setLayout(main_layout)


        #self.setLayout(main_layout)
        self.xx = 0
        self.video_capture = cv2.VideoCapture(self.xx)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.start_button.clicked.connect(self.start_stream)
        self.stop_button.clicked.connect(self.stop_stream)
        self.snap_button.clicked.connect(self.snap_picture)
        self.snap_button.setEnabled(False)  # Disable Snap Picture button initially
        self.snap_counter = 1  # Counter for image filenames

        # Connect the square label click event to the slot
        for square_label in self.video_widget.square_labels:
            square_label.mousePressEvent = lambda event, label=square_label: self.show_full_size_image(label)
            
        self.image_filenames = []  # List to store the filenames

        

    def open_project(self):
        print('Open Project')

    def new_window(self):
        print('New Window')

    def show_settings(self):
        print('Settings')

    def show_about(self):
        print('About')
        
    def show_chart_window(self, event):
        if event.button() == 1:  # Left mouse button click
            # Create a new window to display the chart
            self.chart_window = ChartWindow(self.figure)
            self.chart_window.show()   
        
    def start_stream(self):
 
            
        self.video_capture.open(0)
        self.timer.start(30)
        self.xx += 1
        self.start_button.setEnabled(False)
        self.snap_button.setEnabled(True)  # Endable Snap Picture button 
        self.stop_button.setEnabled(True)

    def stop_stream(self):
        self.germinated = 0
        self.abnormal = 0
        self.ungerminated = 0
        self.total = 0        
        self.timer.stop()
        if self.video_capture is not None:
            self.video_capture.release()
        self.video_capture = None
        self.video_capture = cv2.VideoCapture(self.xx)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.video_widget.set_frame(np.zeros((400, 400, 3), dtype=np.uint8))  # Set a blank image on the video widget
        # Delete the files
        for filename in self.image_filenames:
            os.remove(filename)
        # Clear the list of filenames
        self.image_filenames.clear()
        self.snap_counter = 1

        # Reset the small squares to display gray color
        for square_label in self.video_widget.square_labels:
            square_label.setPixmap(QPixmap())  # Set a blank pixmap image
            square_label.update()
            
        # Update the values and labels for the pie chart and table widget
        values = [0, 0, 0, 0]  # Updated values for variables a, b, c, total
        labels = ['Germinated', 'Abnormal', 'Ungerminated', 'Total']  # Updated labels for the variables
    
        # Update the table widget
        table_widget = self.findChild(QTableWidget)  # Find the existing table widget
        for index, value in enumerate(values):
            item_var_value = QTableWidgetItem(str(value))
            table_widget.setItem(index, 1, item_var_value)
            item_var_value.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            if labels[index] != 'Total':
                item_var_value = QTableWidgetItem(labels[index])
                table_widget.setItem(index, 0, item_var_value)
        # Remove the displayed image in rect2
        self.rect2.clear()
        
        # Set the background color of rect2 to white
        self.rect2.setStyleSheet("background-color: white;")
        
    def show_full_size_image(self, label):
        if len(self.image_filenames) > 0:
            # Get the index of the square label
            index = self.video_widget.square_labels.index(label)
            # Get the filename of the corresponding image
            filename = self.image_filenames[index]
            # Open a new window to display the image
            self.image_window = ImageWindow(filename)
            self.image_window.show()
        
    def snap_picture(self):
        ret, frame = self.video_capture.read()
        if ret:
            # Get the camera's maximum resolution
            width = 640#int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = 480#int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Resize the frame to the camera's maximum resolution
            frame = cv2.resize(frame, (width, height))
            filename = 'image' + datetime.datetime.now().strftime('%H%M%S') + str(self.snap_counter) + '.jpg'
            self.image_filenames.append(filename)
            cv2.imwrite(filename, frame)
            self.snap_counter += 1
            # Load the saved image and resize it to 75x75
            saved_image = cv2.imread(filename)
            saved_image = cv2.resize(saved_image, (75, 75))

            # Get the row and column of the next available square
            row = (self.snap_counter - 2) // 2
            col = (self.snap_counter - 2) % 2

            # Find the square label and display the saved image
            square_label = self.video_widget.layout().itemAtPosition(row, col+1).widget()
            q_image = QImage(saved_image.data, saved_image.shape[1], saved_image.shape[0], saved_image.strides[0], QImage.Format_BGR888)
            q_pixmap = QPixmap.fromImage(q_image)
            square_label.setPixmap(q_pixmap)
            if self.snap_counter == 5:
                self.germinated = 0
                self.abnormal = 0
                self.ungerminated = 0
                self.total = 0             
                self.model_palay() 
                self.snap_button.setEnabled(False)  # Disable Snap Picture button
                
                # Update the values and labels for the pie chart and table widget
                #values = [100, 10, 20, 130]  # Updated values for variables a, b, c, total
                values=[]
                #print(self.germinated)
                #print(self.abnormal)
                #print(self.ungerminated)
                #print(self.total)
                values.append(int(self.germinated))
                values.append(int(self.abnormal))
                values.append(int(self.ungerminated))
                values.append(int(self.total))
                print(values)
                #print(labels)
                labels = ['Germinated', 'Abnormal', 'Ungerminated', 'Total']  # Updated labels for the variables
                values_without_total = values[:-1]
                labels_without_total = labels[:-1]

                # Update the table widget
                table_widget = self.findChild(QTableWidget)  # Find the existing table widget
                for index, value in enumerate(values):
                    item_var_value = QTableWidgetItem(str(value))
                    table_widget.setItem(index, 1, item_var_value)
                    item_var_value.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    if labels[index] != 'total':
                        item_var_value = QTableWidgetItem(labels[index])
                        table_widget.setItem(index, 0, item_var_value)
                try:
                    self.pie_chart.clear()  # Clear the previous chart
                    self.pie_chart.pie(values_without_total, labels=labels_without_total, autopct='%1.1f%%', startangle=90)
                    self.pie_chart.axis('equal')  # Ensure the chart is circular
                    self.pie_chart.legend(fontsize='large')
                    self.figure.savefig('pie_chart.png')
                    pixmap = QPixmap('pie_chart.png')
                
                    self.rect2.setPixmap(pixmap)
                    self.rect2.setScaledContents(True)
                    self.rect2.mousePressEvent = self.show_chart_window
                except Exception as e:
                    print(str(e))
                    pass
    def model_palay(self):
      
      for file in self.image_filenames:  
        labels = ['good', 'bad', 'nochange', 'invalid']
        
        img_size = 224
        #loading the model
        model = models.load_model('compile_rice_class_new03.model')
        
        # Read the original image
        img = cv2.imread(file)

        # Convert to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Blur the image for better edge detection
        img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)

        # Canny Edge Detection
        edges = cv2.Canny(image=img_blur, threshold1=10, threshold2=255) # Canny Edge Detection
        #cv2.imshow('Canny Edge Detection', edges)
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Group contours that are close to each other
        grouped_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            found_group = False
            for idx, group in enumerate(grouped_contours):
                for c in group:
            
                    if abs(x - c[0]) <= 70 and abs(y - c[1]) <= 80:
                        grouped_contours[idx].append((x, y, w, h))
                        found_group = True
                
                        break
                if found_group:
                    break
            if not found_group:
                grouped_contours.append([(x, y, w, h)])

        # Combine rectangles that are very close to each other
        combined_contours = []
        for group in grouped_contours:
            x_min, y_min, x_max, y_max = float('inf'), float('inf'), float('-inf'), float('-inf')
            for contour in group:
                x, y, w, h = contour
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x + w)
                y_max = max(y_max, y + h)
            combined_contours.append((x_min, y_min, x_max, y_max))

        # Remove rectangles that are fully contained within other rectangles
        final_contours = []
        for contour in combined_contours:
            x1, y1, x2, y2 = contour
            is_inside = False
            for c in combined_contours:
                if c is contour:
                    continue
                x1_c, y1_c, x2_c, y2_c = c
                #if x1_c <= x1 and y1_c <= y1 and x2_c >= x2 and y2_c >= y2:
                if (x1_c <= x1 <= x2_c and y1_c <= y1 <= y2_c) or (x1_c <= x2 <= x2_c and y1_c <= y2 <= y2_c):
                    is_inside = True
                    break
            if not is_inside:
                final_contours.append(contour)
                #print(contour)

        # Draw rectangles around final contours and save as separate jpg files
        for i, contour in enumerate(final_contours):
            x1, y1, x2, y2 = contour
            # Add 25 pixels margin to the rectangle
            x1 -= 25
            y1 -= 25
            x2 += 25
            y2 += 25
            # Make sure the coordinates are within the image bounds
            x1 = max(x1, 0)
            y1 = max(y1, 0)
            x2 = min(x2, img.shape[1])
            y2 = min(y2, img.shape[0])
            # Extract the region of interest
            roi = img[y1:y2, x1:x2]
            # Save the region of interest as a jpg file
            #i=145
            v_img = 'xx.jpg'
            cv2.imwrite(f'xx.jpg', roi)
            img_1 = cv2.imread(v_img)#open the image
            img_1= cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)#convert the color
            img_1 = cv2.resize(img_1, (img_size, img_size))#resize the image
            xx = np.array([img_1])/255 #normalize the data
            predictions = np.argmax(model.predict(xx), axis=-1)#classify the image
            predictions = predictions.reshape(1,-1)[0]
            if labels[predictions[0]] != 'invalid':            
                if labels[predictions[0]] == 'good':
                    self.germinated += 1
                elif labels[predictions[0]] == 'bad':
                    self.abnormal += 1
                elif labels[predictions[0]] == 'nochange':
                    self.ungerminated += 1
                    
                self.total += 1
                
                #cv2.imwrite(f'{labels[predictions[0]]}_{i+1}.jpg', roi)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                caption = f'{labels[predictions[0]]}_{i+1}'
                (text_width, text_height) = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0, 0)[0]
                text_offset_x = 0
                text_offset_y = int(text_height-10)   
                cv2.rectangle(img, (x1, y1 ), (x2, y1), (255, 255, 255), -1)
                cv2.putText(img, caption, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0, 0), 0, cv2.LINE_AA)
        


        # Display image with rectangles around clusters of detected edges
        #cv2.imshow('Edges with Rectangles', img)
        cv2.imwrite(file, img)

    def update_frame(self):
        ret, frame = self.video_capture.read()
        if ret:
            frame = cv2.resize(frame, (400, 400))
            self.video_widget.set_frame(frame)
            
    def closeEvent(self, event):
        self.stop_stream()
        event.accept()
        
if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
