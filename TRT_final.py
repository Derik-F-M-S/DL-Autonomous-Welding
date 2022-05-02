'''
Final Demo Scipt for joint and seam detection for autonomous welding 
Using TensorRT for inference
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
from cvu.detector.yolov5 import Yolov5 as Yolov5Trt
from torch import int32
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
CONFIDENCE_THRESHOLD = 0.8 #only detections with confidence > this are valid 
#------------------------------------------------------------------
#               Set some flags

calib_depth = True # set true if testing with depth stuff
calib_pointCloud = False # set true if testing with point clouds
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
    #s = pipeline_profile.get_device().query_sensors()[1]
    #s.set_option(rs.option.exposure, 100)
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

def build_model(classes):
    '''
    Builds the trt engine model using the Onnx model file or uses it directly if it already 
    exist and is provided as the weights
    This function uses the functionality of the YoloV5 from cvu.detector.yolov5 to build the model
    see https://github.com/BlueMirrors/cvu for additional details
    
    Args:
        classes (list):
            A list of class names to map the detections back to
    Returns:
        model:
            A tensorRT model for inference
    
    '''
    # select onnx file to read weights/model from
    weight="./models/best_from_scratch.onnx"
    model = Yolov5Trt(classes=classes,
                      backend="tensorrt",
                      weight=weight,
                      auto_install=False)
    return model

def detect(image, net):
    '''
    Returns the detections by passing the image into the model
    
    Args: 
        image (np.ndarray):
            an image or image batch as an np ndarray in the format
            bs,c,w,h
        net:
            The network as tensorRT engine 
    Returns:
        preds:
            a prediction object (see https://github.com/BlueMirrors/cvu for more details)
    
    '''
    preds = net(image)
    return preds

def load_classes():
    '''
    Loads the classes from a text file containing each class in a new line ans converts it 
    to a list of string with an element in the list for each class
    
    Returns:
        class_list (list): 
            a list of classes
    '''
    
    class_list = []
    with open("classes.txt", "r") as f:
        class_list = [cname.strip() for cname in f.readlines()]
    return class_list

class_list = load_classes()

def unwrap_detection(input_image, output_data):
    '''
    Unwraps the detections object and reformats it so that it has the correct format for the rest of the code 
    that is in the ((list)class_ids,(list)confidences,(list)boxes) format
    
    Args:
        input_image (ndarray):
            The input image (the image before resizing ie the image coming from the camera)
        output_data(list):
            The output from the TRT model (detections object from cvu)
    Returns:
            class_ids(list): list of class ids 
            confidences(list): list of confidences
            boxes(list): list of bounding boxes
    
    '''
    
    class_ids = []
    confidences = []
    boxes = []
    image_h, image_w, _ = input_image.shape
    scale = np.flipud(np.divide([image_h,image_w],[INPUT_HEIGHT,INPUT_WIDTH]))  # you have to flip because the image.shape is (y,x) but your corner points are (x,y)
    for row in output_data:
        confidence = row.confidence
        class_id=row.class_id
        if confidence >= CONFIDENCE_THRESHOLD:
            confidences.append(confidence)
            class_ids.append(class_id)
            x1, y1, x2, y2 = row.bbox
            #Reformatting the bbox from x1, y1, x2, y2 to x,y,w,h
            #also mapping the bounding box from the resized frame to the original frame
            top_left_corner= [int(x1),int(y1)]
            bottom_right_corner= [int(x2),int(y2)]
            new_top_left_corner = np.multiply(top_left_corner, scale )
            new_bottom_right_corner = np.multiply(bottom_right_corner, scale )
            w= new_bottom_right_corner[0]-new_top_left_corner[0]
            h= new_bottom_right_corner[1]-new_top_left_corner[1]
            box = np.array([new_top_left_corner[0], new_top_left_corner[1], w ,h],dtype=int)
            boxes.append(box)
    return class_ids, confidences, boxes

def format_yolov5(frame):
    '''
    Formats the frame into the expected size for the model (640,640)
    Args:
        frame(ndarray): input frame from camera
    Returns:
        result(ndarray): resized frame for yolo
    '''
    result= cv2.resize(frame,(INPUT_WIDTH, INPUT_HEIGHT))
    return result

#Get model
net = build_model(class_list)

#start some counters
start = time.time()*10**9
frame_count = 0
total_frames = 0
fps = -1
jointCount = 0  # how many consecutive frames have we seen the joint

while True:
    #
    #Main Driving loop executes until the esc key is pressed 
    #

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

    class_ids, confidences, boxes = unwrap_detection(frame, outs)

    frame_count += 1
    total_frames += 1

    # MQTT/INTEGRATION STUFF:
    # if no detection just publish 0 and reset the joint counter 
    if len(class_ids) == 0:
        jointCount = 0
        client.publish("joint_found",str(0))
    else:
        jointCount +=  1    
    print('saw a joint for ',jointCount,' frames')
    if len(confidences)>0: 
        #get the detection with the largest confidence since we only work with 1 detection at a time this works
        b=np.argmax(confidences)#
        classid= class_ids[b]
        confidence= confidences[b]
        box= boxes[0]
        color = colors[int(classid) % len(colors)]
        print(confidence)
        boxLeft = box[0]
        boxTop = box[1]
        boxWidth = box[2]
        boxHeight = box[3]

        # extract cropped camera frame to bbox coords
        img= frame[box[1]:box[1]+box[3],box[0]:box[0]+box[2]]
        lines = None
        if all(dim >0 for dim in [ dim for dim in img.shape]):
            # create HSV image -> threshold mask -> canny edge detect:
            boxh, boxw , _ = img.shape
            max_size= max(boxh,boxw) 
            hsvIMG= cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsvIMG, dark_hsv, light_hsv)
            cv2.namedWindow("mask")
            cv2.imshow('mask',mask)
            #edges= cv2.Canny(mask,50,150,apertureSize=3)

            # grab lines
            lines = cv2.HoughLinesP(
                        mask, # Input edge image
                        1, # Distance resolution in pixels
                        np.pi/180, # Angle resolution in radians
                        threshold=50, # Min number of votes for valid line
                        minLineLength=int(.80*max_size), # Min allowed length of line
                        maxLineGap=int(.25*max_size) # Max allowed gap between line for joining them
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
            x1 = min(finalLine[0][0] + boxLeft, 640)
            y1 = min(finalLine[0][1] + boxTop, 480)
            x2 = min(finalLine[1][0] + boxLeft, 640)
            y2 = min(finalLine[1][1] + boxTop, 480)        
  
            cv2.line(frame,(x1,y1),(x2,y2),(255,0,0),10) # adds line to frame 
            
            depth1 = depth_frame.get_distance(x1,y1)
            depth2 = depth_frame.get_distance(x2,y2)

            # normalize detection coords for arm team
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

            #interpolate points to send to arm team and send as a string list
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
                #pass
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

    #for final demo we switch from one color seam to another with the c key
    if cv2.waitKey(1) == ord('c'):
        print("Changing color threshold values") 
        print("\n \n \n \n \n")
        dark_hsv= np.array([0.05*180,0.0*255,.36*255])
        light_hsv= np.array([0.27*180,1*255,1*255])
         

    if cv2.waitKey(1) == 27:
        print("finished by user")
        break

print("Total frames: " + str(total_frames))
