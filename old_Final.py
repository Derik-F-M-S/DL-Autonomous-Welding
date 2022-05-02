'''
Final Demo Scipt for joint and seam detection for autonomous welding 
Using OpenCV for inference
'''
#------------------------------------------------------------------
#               Import required modules
import cv2
import time
import sys
import numpy as np
import cv2                                # state of the art computer vision algorithms library
import numpy as np                        # fundamental package for scientific computing
import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures
import pyrealsense2 as rs                 # Intel RealSense cross-platform open-source API
import paho.mqtt.client as mqtt 
import sys
import argparse
import time
import math
import scipy.interpolate
from scipy.spatial import distance
#import open3d as o3d
#------------------------------------------------------------------
#               Set some constants

#   HSV threshold values
dark_hsv= np.array([0.302*180,0.164*255,.1*255])
light_hsv= np.array([.487*180,1*255,1.00*255])
#   colors for showing the bounding boxes
colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]
#   Number of points for interpolation
NUM_POINTS= 40
#   Model input images need to be 640 x 640
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
#   NMS and thersholds for Yolo output (OpenCV version) 
SCORE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
CONFIDENCE_THRESHOLD = 0.8 #only detections with confidence > this are valid 
#------------------------------------------------------------------
#               Set some flags

calib_depth = False # set true if testing with depth stuff
calib_pointCloud = False # set true if testing with point clouds
# set Cuda flag and get model
is_cuda = True
# True for camera feed, false for bag file
LIVE_FEED=True 
#------------------------------------------------------------------
#               MQTT Setup
# replace localhost with ip of Arm nano for demo
mqttBroker ="10.155.61.199" 
client = mqtt.Client("Label Detection")
client.connect(mqttBroker) 
#------------------------------------------------------------------
# if running on a pre-recorded rosbag:
if not LIVE_FEED:
    pipeline = rs.pipeline()
    cfg = rs.config()
    #change to ROS bag to use as needed
    cfg.enable_device_from_file("/home/learning/librealsense-jupyter/object_detect.bag")
    profile = pipeline.start(cfg)

# Live feed from camera set-up
elif LIVE_FEED:
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            s.set_option(rs.option.exposure, 150)
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

else :
    print("Unknown video source!")
    sys.exit(-1)

print("Environment Ready")
#------------------------------------------------------------------
#               Function Defs
#   Defining the functions used by the main driver loop

#From:https://github.com/doleron/yolov5-opencv-cpp-python
def build_model(is_cuda):
    # select onnx file to read weights/model from
    net = cv2.dnn.readNet("best_from_scratch.onnx")
    if is_cuda:
        print("Attempting to use CUDA")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    else:
        print("Running on CPU")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

#From:https://github.com/doleron/yolov5-opencv-cpp-python
def detect(image, net):
    blob = cv2.dnn.blobFromImage(image, 1/255.0, size = (INPUT_WIDTH, INPUT_HEIGHT), swapRB=False, crop=False)
    net.setInput(blob)
    preds = net.forward()
    return preds
#From:https://github.com/doleron/yolov5-opencv-cpp-python
def load_capture():
    capture = cv2.VideoCapture("tjoint.mp4")
    return capture

#From:https://github.com/doleron/yolov5-opencv-cpp-python
def load_classes():
    class_list = []
    with open("classes.txt", "r") as f:
        class_list = [cname.strip() for cname in f.readlines()]
    return class_list

class_list = load_classes()

#From:https://github.com/doleron/yolov5-opencv-cpp-python
# returns our bounding boxes:
def wrap_detection(input_image, output_data):
    class_ids = []
    confidences = []
    boxes = []

    rows = output_data.shape[0]

    image_width, image_height, _ = input_image.shape

    x_factor = image_width / INPUT_WIDTH
    y_factor =  image_height / INPUT_HEIGHT

    for r in range(rows):
        row = output_data[r]
        confidence = row[4]
        if confidence >= CONFIDENCE_THRESHOLD:

            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > .25):

                confidences.append(confidence)

                class_ids.append(class_id)

                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45) 

    result_class_ids = []
    result_confidences = []
    result_boxes = []

    for i in indexes:
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])

    return result_class_ids, result_confidences, result_boxes

#From:https://github.com/doleron/yolov5-opencv-cpp-python
def format_yolov5(frame):
    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame
    return result





net = build_model(is_cuda)

#start some counters
start = time.time()*10**9
frame_count = 0
total_frames = 0
fps = -1
jointCount = 0  # how many consecutive frames have we seen the joint


def transform_coordinates(original_x, original_y):
	# Shilpa's function, not currently in use
	new_x, new_y = None, None
	# If original x is less than or equal to 320 and
	# y is less than or equal 240, new coordinate is (-x, +y)
	if (original_x <= 320) and (original_y <= 240):
		new_x = (original_x - 320) / 320
		new_y = (original_y + 240) / 240
	# If original x is less than or equal to 320 and
	# y is greater than 240, new coordinate is (-x, -y)
	elif (original_x <= 320) and (original_y > 240):
		new_x = (original_x - 320) / 320
		new_y = (original_y - 240) / 240
	# If original x is greater than  to 320 and
	# y is less than or equal 240, new coordinate is (+x, +y)
	elif (original_x > 320) and (original_y <= 240):
		new_x = (original_x + 320) / 320
		new_y = (original_y + 240) / 240
	# If original x is greater than  to 320 and
	# y is greater than 240, new coordinate is (+x, -y)
	elif (original_x > 320) and (original_y > 240):
		new_x = (original_x + 320) / 320
		new_y = (original_y - 240) / 240

	return new_x, new_y

while True:
    pipeline.wait_for_frames()

    # store frame
    frameset = pipeline.wait_for_frames()
    color_frame = frameset.get_color_frame()
    depth_frame = frameset.get_depth_frame()
    frame = np.asanyarray(color_frame.get_data())

    # Create alignment primitive with color as its target stream:
    align = rs.align(rs.stream.color)
    frameset = align.process(frameset)

    # Update color and depth frames:
    colorizer = rs.colorizer()
    aligned_depth_frame = frameset.get_depth_frame()
    colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
    
    # Standard OpenCV boilerplate for running the net:
    inputImage = format_yolov5(frame)
    outs = detect(inputImage, net)

    class_ids, confidences, boxes = wrap_detection(inputImage, outs[0])

    frame_count += 1
    total_frames += 1

    # MQTT/INTEGRATION STUFF:
    # if no detection just publish 0 and reset the joint counter 
    # otherwise grab bbox values
    if len(class_ids) == 0:
        jointCount = 0
        client.publish("joint_found",str(0))
    else:
        jointCount +=  1
        boxLeft = boxes[0][0]
        boxTop = boxes[0][1]
        boxWidth = boxes[0][2]
        boxHeight = boxes[0][3]

    
    print('saw a joint for ',jointCount,' frames')

    # iterate through the bounding boxes/detections
    for (classid, confidence, box) in zip(class_ids, confidences, boxes):
        color = colors[int(classid) % len(colors)]
        print(confidence)

        # extract cropped camera frame to bbox coords
        img= frame[box[1]:box[1]+box[3],box[0]:box[0]+box[2]]
        lines = None
        if all(dim >0 for dim in [ dim for dim in img.shape]):
            # create HSV image -> threshold mask -> canny edge detect:
            boxh, boxw , _ = img.shape
            min_size= max(boxh,boxw) 
            hsvIMG= cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsvIMG, dark_hsv, light_hsv)
            edges= cv2.Canny(mask,50,150,apertureSize=3)

            # grab lines
            lines = cv2.HoughLinesP(
                        edges, # Input edge image
                        1, # Distance resolution in pixels
                        np.pi/180, # Angle resolution in radians
                        threshold=50, # Min number of votes for valid line
                        minLineLength=int(.88*min_size), # Min allowed length of line
                        maxLineGap=int(.25*min_size) # Max allowed gap between line for joining them
                        )
        
        # Iterate over points
        lines_list=[]
        lengths_list = []
        if lines is not None:
            for points in lines[0:4]:
                # Extracted points nested in the list
                x1,y1,x2,y2=points[0]
                lineLength = distance.euclidean((x1,y1),(x2,y2))
                lines_list.append([(x1,y1),(x2,y2)])
                lengths_list.append(lineLength)

            # grab the longest detected line
            # and grab (x,y) for its two points
            lengths_list = np.array(lengths_list)
            finalLine = lines_list[np.argmax(lengths_list)]
            x1 = finalLine[0][0] + boxLeft
            y1 = finalLine[0][1] + boxTop
            x2 = finalLine[1][0] + boxLeft
            y2 = finalLine[1][1] + boxTop         
  
            cv2.line(frame,(x1,y1),(x2,y2),(255,0,0),10) # adds line to frame 

            # get depth values at line points:
            depth1 = depth_frame.get_distance(x1,y1)
            depth2 = depth_frame.get_distance(x2,y2)

            # normalize detection coords for arm team
            #x1final, y1final = transform_coordinates(x1, y1)
            #x2final, y2final = transform_coordinates(x2, y2)
            x1final = (x1-320) / 320   
            x2final = (x2-320) / 320   # normalizing coordinates
            y1final = (y1-240) / -240   # so that center of image is 0,0
            y2final = (y2-240) / -240


            # Just for calibration can delete if you want:
            print("ORIG X:",x1,",",x2)
            print("ORIG Y:",y1,",",y2)
            print("TOP:",x1final,y1final)
            print("BOT:",x2final,y2final)

            if calib_depth:
                #print("DEPTHS:",depth1,depth2)
                # adds circles to frame if plotting
                circle = cv2.circle(colorized_depth,(x1,y1),radius=10,color=(0,0,0),thickness=-1)
                circle = cv2.circle(colorized_depth,(x2,y2),radius=10,color=(0,255,0),thickness=-1)
                cv2.namedWindow("Realsense_dist", cv2.WINDOW_AUTOSIZE)
                cv2.imshow('Realsense_dist',colorized_depth)

            #interpolate 10 points to send to arm team and send as a string list
            xlist = np.linspace(x2final,x1final,NUM_POINTS)
            ylist = np.linspace(y2final,y1final,NUM_POINTS) 
            dlist = np.linspace(depth2,depth1,NUM_POINTS)
            finalLine = [[round(x,2),round(y,2),round(z,2)] for x,y,z in zip(xlist,ylist,dlist)]
            finalLine = ','.join(str(i) for i in finalLine)

            # publish if seam detected and joint has been in view for at least 5 consecutive frames
            if jointCount >= 5:
                client.publish("joint_found",str(1))
                client.publish("coord Start",str(x1final)+','+str(y1final))  # MQTT pub start & end
                client.publish("coord_list",finalLine)
                print("Just Published Coords: ",finalLine) 
            else:

                client.publish("joint_found",str(0))

            if calib_pointCloud:
                pc = rs.pointcloud();
                pc.map_to(color_frame);
                pointcloud = pc.calculate(depth_frame)
                h = rs.video_frame(depth_frame).width
                w = rs.video_frame(depth_frame).height
                verts = np.asanyarray(pointcloud.get_vertices(dims=2)).view(np.float32).reshape(h,w,3)
                roi = verts[200:600, 100:400,:].reshape(-1,3)
                #pointcloud.export_to_ply("pointcloud.ply", color_frame);
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(roi)
                o3d.io.write_point_cloud("croppedCloud.ply", pcd)
        else:
            
            client.publish("joint_found",str(0))
        
        cv2.rectangle(frame, box, color, 2)
        cv2.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
        cv2.putText(frame, class_list[classid], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0))


    if frame_count >= 30:
        end = time.time()*10**9
        fps = 1000000000 * frame_count / (end - start)
        frame_count = 0
        start = time.time()*10**9
    
    if fps > 0:
        fps_label = "FPS: %.2f" % fps
        cv2.putText(frame, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    #cv2.resizeWindow("Resized_window",480,640)
    cv2.imshow("output",frame)


    if cv2.waitKey(1) > -1:
        print("finished by user")
        break

print("Total frames: " + str(total_frames))
