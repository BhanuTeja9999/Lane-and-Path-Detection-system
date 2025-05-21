from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from PIL import Image, ImageTk

# Global filename
global filename

# YOLO model setup
def load_yolo_model():
    net = cv2.dnn.readNet("yolov4.weights")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers


# Function to predict path using LSTM
def predict_path_lstm(data):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(data.shape[1], data.shape[2])))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Assuming 'data' is a placeholder for preprocessed input (e.g., speed, angle, etc.)
    predictions = model.predict(data)
    return predictions

# Calculate average line

def calculate_average_line(lines):
    if len(lines) == 0:
        return None
    x1_sum, y1_sum, x2_sum, y2_sum = 0, 0, 0, 0
    for line in lines:
        x1_sum += line[0]
        y1_sum += line[1]
        x2_sum += line[2]
        y2_sum += line[3]

    return [int(x1_sum / len(lines)), int(y1_sum / len(lines)), int(x2_sum / len(lines)), int(y2_sum / len(lines))]

# Lane detection with YOLO integration
def lane_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # Blur to reduce noise
    edges = cv2.Canny(blur, 50, 150)  # Edge detection

    height, width = image.shape[:2]
    vertices = np.array([[(0, height), (width * 0.5, height * 0.6), (width, height * 0.6), (width, height)]], dtype=np.int32)
    masked_edges = cv2.fillPoly(edges, [vertices], 255)

    # Integrating YOLO for lane detection
    net, output_layers = load_yolo_model()
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Detection threshold
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Hough Transform
    lines = cv2.HoughLinesP(masked_edges, rho=2, theta=np.pi/180, threshold=100, minLineLength=40, maxLineGap=5)

    left_lane_lines, right_lane_lines = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1)
        if slope < 0:  # Negative slope indicates left lane
            left_lane_lines.append([x1, y1, x2, y2])
        else:  # Positive slope indicates right lane
            right_lane_lines.append([x1, y1, x2, y2])

    left_lane = calculate_average_line(left_lane_lines)
    right_lane = calculate_average_line(right_lane_lines)

    if left_lane is None or right_lane is None:
        return "Straight"

    mid_point = width // 2
    left_lane_x = (left_lane[0] + left_lane[2]) / 2
    right_lane_x = (right_lane[0] + right_lane[2]) / 2

    if left_lane_x < mid_point - 50:  # Turn left
        return "Left"
    elif right_lane_x > mid_point + 50:  # Turn right
        return "Right"
    else:
        return "Straight"


def lane_Detection(image):
    """
    Function to detect lane lines in an image and determine if the car should go left, right, or straight. 
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # Blur to reduce noise
    edges = cv2.Canny(blur, 50, 150)  # Edge detection
    
    # Define region of interest (ROI) for lane lines
    height, width = image.shape[:2]
    vertices = np.array([[(0, height), (width * 0.5, height * 0.6), (width, height * 0.6), (width, height)]], dtype=np.int32)
    masked_edges = cv2.fillPoly(edges, [vertices], 255)
    
    # Hough Transform to detect lines
    lines = cv2.HoughLinesP(masked_edges, rho=2, theta=np.pi/180, threshold=100, minLineLength=40, maxLineGap=5)
    
    left_lane_lines = []
    right_lane_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1)
        if slope < 0:  # Negative slope indicates left lane
            left_lane_lines.append([x1, y1, x2, y2])
        else:  # Positive slope indicates right lane
            right_lane_lines.append([x1, y1, x2, y2])
    
    # Calculate average lane lines for left and right
    left_lane = calculate_average_line(left_lane_lines)
    right_lane = calculate_average_line(right_lane_lines)
    # Determine steering direction
    if left_lane is None or right_lane is None:
        return "Straight"  # If no lane lines detected, go straight
    
    mid_point = width // 2
    left_lane_x = (left_lane[0] + left_lane[2]) / 2
    right_lane_x = (right_lane[0] + right_lane[2]) / 2
    print(left_lane_x)
    print(right_lane_x)
    print(str(mid_point)+" "+str(width))
    
    if left_lane_x < mid_point - 50:  # Turn left
        return "Left"
    elif right_lane_x > mid_point + 50:  # Turn right
        return "Right"
    else:
        return "Straight"

def calculateLines(frame, lines):
    left = []
    right = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        y_intercept = parameters[1]
        if slope < 0:
            left.append((slope, y_intercept))
        else:
            right.append((slope, y_intercept))
    left_avg = np.average(left, axis = 0)
    right_avg = np.average(right, axis = 0)
    left_line = calculateCoordinates(frame, left_avg)
    right_line = calculateCoordinates(frame, right_avg)
    return np.array([left_line, right_line])

# Upload video
def uploadVideo():
    global filename
    filename = filedialog.askopenfilename(initialdir="Video")
    text.delete('1.0', END)
    text.insert(END, filename + " loaded\n\n")

# Path detection with LSTM integration
def pathdetection():
    global filename
    text.delete('1.0', END)
    camera = cv2.VideoCapture(filename)
    while True:
        grabbed, frame = camera.read()
        if frame is not None:
            frame = cv2.resize(frame, (400, 400))
            direction = lane_detction(frame)
            # Predict path using LSTM placeholder
            cv2.putText(frame, direction, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.9, (0, 0, 255), 3)
            cv2.imshow("Sign Langauge Prediction", frame)
            keypress = cv2.waitKey(1) & 0xFF
            if keypress == ord("q"):
                break
        else:
            break
    camera.release()
    cv2.destroyAllWindows()

def cannyDetection(img):
    grayImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurImg = cv2.GaussianBlur(grayImg, (5, 5), 0)
    cannyImg = cv2.Canny(blurImg, 50, 150)
    return cannyImg

def segmentDetection(img):
    height = img.shape[0]
    polygons = np.array([[(0, height), (800, height), (380, 290)]])
    maskImg = np.zeros_like(img)
    cv2.fillPoly(maskImg, polygons, 255)
    segmentImg = cv2.bitwise_and(img, maskImg)
    return segmentImg

def calculateCoordinates(frame, parameters):
    if type(parameters) == np.ndarray:
        slope, intercept = parameters
        y1 = frame.shape[0]
        y2 = int(y1 - 150)
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
    else:
        x1 = 2
        y1 = 2
        x2 = 2
        y2 = 2
    return np.array([x1, y1, x2, y2])

def visualizeLines(frame, lines):
    lines_visualize = np.zeros_like(frame)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            try:
                cv2.line(lines_visualize, (x1, y1), (x2, y2), (0, 255, 0), 5)
            except:
                pass
    return lines_visualize

def pathDetection():
    global filename
    text.delete('1.0', END)
    camera = cv2.VideoCapture(filename)
    while(True):
        (grabbed, frame) = camera.read()
        if frame is not None:
            img = cv2.resize(frame, (400, 400))
            direction = lane_Detection(img)
            frame = cv2.resize(frame, (600, 600))
            canny = cannyDetection(frame)
            segment = segmentDetection(canny)
            hough = cv2.HoughLinesP(segment, 2, np.pi / 180, 100, np.array([]), minLineLength = 100, maxLineGap = 50)
            if hough is not None:
                lines = calculateLines(frame, hough)
                linesVisualize = visualizeLines(frame, lines)
                output = cv2.addWeighted(frame, 0.9, linesVisualize, 1, 1)
            else:
                output = frame
            cv2.putText(output, direction, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.9, (0, 0, 255), 3)
            cv2.imshow("Sign Langauge Prediction", output)
            keypress = cv2.waitKey(1) & 0xFF
            if keypress == ord("q"):
                break
        else:
            break
    camera.release()
    cv2.destroyAllWindows()

# Main GUI setup
main = tkinter.Tk()
main.title("Lane & Path Detection")
main.geometry("1300x1200")

# Load the image and resize it to fit the window
image = Image.open("Design.png")
image = image.resize((1300, 1200), Image.ANTIALIAS)
bg_image = ImageTk.PhotoImage(image)


# Create a Label to display the background image
bg_label = Label(main, image=bg_image)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

'''
font = ('times', 16, 'bold')
title = Label(main, text='Lane & Path Detection')
title.config(bg='chocolate', fg='white')
title.config(font=font)
title.config(height=3, width=120)
title.place(x=0, y=5)'''

font1 = ('times', 13, 'bold')
upload = Button(main, text="Upload Video", command=uploadVideo)
upload.place(x=700, y=350)
upload.config(font=font1)

pathButton = Button(main, text="Lane & Path Detection", command=pathDetection)
pathButton.place(x=700, y=500)
pathButton.config(font=font1)

exitButton = Button(main, text="Exit", command=main.destroy)
exitButton.place(x=700, y=600)
exitButton.config(font=font1)

font1 = ('times', 12, 'bold')
text = Text(main, height=20, width=80)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10, y=350)
text.config(font=font1)

main.config(bg='light salmon')
main.mainloop()
