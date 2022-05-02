# Learning Team
<p align="center">
  <img src="https://user-images.githubusercontent.com/97491169/165002176-2b2c9a6a-15fa-4bfa-93c3-226ca11ea3d3.gif" />
</p>

## This branch contains the perception and deep learning team code for the sp22Robot. This includes training scripts for the instance segmentation and object detection models as well as other utilities and the main script that ran during robot operation. 
## Hardware Requirements for perception: 
- Realsense camera with RGB and Depth sensors
- Nvidia Jetson Nano
## Software Requirements for perception
- Jetpack 4.4 
- PyRealSense2 built from source for ARM 64 
- OpenCV built with cuda (see https://github.com/mdegans/nano_build_opencv) 
- CVU https://github.com/BlueMirrors/cvu

## Dataset 
The dataset consist of images of metal plates to be welded captured using the RGB sensor from Realsense D455 and D435 cameras from Intel. The object detection dataset can be found at https://app.roboflow.com/stuti-garg-oqsc8/robotics-jkowd/5 
<p align="center">
  <img src="https://user-images.githubusercontent.com/97491169/165000356-b013d502-25ec-4758-9db9-50062e10566f.png" />
</p>

## Model Training
Both the obeject detection and instance segmentation models were trainied on Google Colab Pro using an Nvidia Tesla V100 using the official training script from https://github.com/ultralytics/yolov5 

## Approach
### Object Detection
During the developement process for the welding joint detection we explored using both instance segmentation and object detection and while both models were trained on the custom dataset, ultimately the object detection using YOLO v5 was chosen as the final model to identify the welding joints. Once the joints are identified color thresholding is used to identify the seam and the Realsense D455 camera's depth module is used to augment the detections with the 3d location of the seam so that this information can be passed on to the robotic arm using MQTT. 

![pipeline](https://user-images.githubusercontent.com/97491169/165000523-8cd23a11-e3aa-475a-8e88-5863a220b194.png)
<p align="center">
  <img width="360" height="200" src="https://user-images.githubusercontent.com/97491169/165000559-cc874585-200e-4ba6-9bcd-ac99d5e33f29.PNG" />
</p>

### Deployment to the Nvidia Jetson Nano 
The YOLO v5 model was deployed to the Jetson Nano using two approaches. Before the model could be deployed, it was trainend using the training script provided by https://github.com/ultralytics/yolov5 and the model weights were exported in the ONNX format, then the first approach used OpenCV and the it's DNN module in python to deploy the YOLO v5 model loaded in as an ONNX model based on code from https://github.com/doleron/yolov5-opencv-cpp-python. Using this implementation, the performace averaged around 2 FPS and while this proved to be enough for the detections as long as the robotic arm moved slowly, a more efficient implementation using TensorRT was ultimately used based on the implementation found at https://github.com/BlueMirrors/Yolov5-TensorRT. This implementation saw a performance of about 6-7 FPS on the Nano
