# Deploy a People Counter App at the Edge
Project 1 of the **Intel® Edge AI for IoT Developers** Nanodegree Program.

The backbone of the project is based on the original Udacity/Intel [github repo](https://github.com/udacity/nd131-openvino-fundamentals-project-starter).


| Details            |              |
|-----------------------|---------------|
| Programming Language: |  Python 3.5 to 3.7 |

![people-counter-python](./images/people-counter-image.png=200)

## What it Does

The people counter application detects people in a designated area, provides the number of people in the frame, average duration of people in frame, and total count.

## How it Works

The counter app uses the **Inference Engine**(IE) included in the Intel® Distribution of OpenVINO™ Toolkit to make the inference.  

The detection model used is the pre-trained [person-detection-retail-0013](https://docs.openvinotoolkit.org/2018_R5/_docs_Retail_object_detection_pedestrian_rmnet_ssd_0013_caffe_desc_person_detection_retail_0013.html), based on MobileNetV2-like backbone.
 
 The app counts the number of people in the current frame, the duration that a person is in the frame (time elapsed between entering and exiting a frame) and the total count of people. It then sends the data to a local web server using the Paho MQTT Python package.

The overall app architecture is described below:
![architectural diagram](./images/arch_diagram.png)

## Requirements

### Hardware

* 6th to 10th generation Intel® Core™ processor with Iris® Pro graphics or Intel® HD Graphics.
* OR use of Intel® Neural Compute Stick 2 (NCS2)

### Software

*   Intel® Distribution of OpenVINO™ toolkit 2019 R3 release
*   Node v6.17.1, or higher
*   Npm v3.10.10, or higher
*   CMake
*   MQTT Mosca server
  
        
## Setup

### Install Intel® Distribution of OpenVINO™ toolkit

Refer to the relevant instructions for the appropriate operating system:

- [Linux/Ubuntu](./linux-setup.md)
- [Mac](./mac-setup.md)
- [Windows](./windows-setup.md)
- [Raspberry Pi](./rpi-setup.md)

### Install Nodejs and npm

Refer to the relevant instructions for the appropriate operating system:

- [Linux/Ubuntu](./linux-setup.md)
- [Mac](./mac-setup.md)
- [Windows](./windows-setup.md)
- [Raspberry Pi](./rpi-setup.md)


### Clone the directory and install dependencies
After cloning the directory, some modules dependencies need to be installed.

There are three components that need to be running in separate terminals for this application to work:

-   MQTT Mosca server 
-   Node.js* Web server
-   FFmpeg server
     
From the main directory:

* For MQTT/Mosca server:
   ```
   cd webservice/server
   npm install
   ```

* For Web server:
  ```
  cd ../ui
  npm install
  ```
  **Note:** If any configuration errors occur in mosca server or Web server while using **npm install**, use the below commands:
   ```
   sudo npm install npm -g 
   rm -rf node_modules
   npm cache clean
   npm config set registry "http://registry.npmjs.org"
   npm install
   ```

### Download the model
The model used can be downloaded using the __Model Downloader__ included in the toolkit.

From the main directory

```
mkdir model && cd model
python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name person-detection-retail-0013 --outputdir .
```

**Note1:** The model has been used with precision FP16. Add `--precisions=FP16` to the above command if you do not want to download and/or use the FP32 precision model.

**Note2:** The app can be used with any other detection model, or ad-hoc trained model. If it is not already converted, the model will need  to be converted into an **Intermediate Representation** (.xml and .bin files) in order to be used by the IE.

## Run the application

From the main directory:

### Step 1 - Start the Mosca server

```
cd webservice/server/node-server
node ./server.js
```

You should see the following message, if successful:
```
Mosca server started.
```

### Step 2 - Start the GUI

Open new terminal and run below commands.
```
cd webservice/ui
npm run dev
```

You should see the following message in the terminal.
```
webpack: Compiled successfully
```

### Step 3 - FFmpeg Server

Open new terminal and run the below commands.
```
sudo ffserver -f ./ffmpeg/server.conf
```

**Note:** If `ffserver` is not found, you might need to specify the whole path:
```
sudo <ffmpeg_install_dir>/ffmpeg/ffserver -f .ffmpeg/server.conf
```

### Step 4 - Run the code

Open a new terminal to run the code. 

#### Setup the environment

You must configure the environment to use the Intel® Distribution of OpenVINO™ toolkit one time per session by running the following command:
```
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.<python_version>
export PYTHONPATH="/opt/intel/openvino/python/python3.<python_version>
```


#### Running on the CPU

When running Intel® Distribution of OpenVINO™ toolkit Python applications on the CPU, the CPU extension library is required. This can be found at: 

```
/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/
```

*Depending on whether you are using Linux or Mac, the filename will be either `libcpu_extension_sse4.so` or `libcpu_extension.dylib`, respectively.* (The Linux filename may be different if you are using a AVX architecture)

Though by default application runs on CPU, this can also be explicitly specified by ```-d CPU``` command-line argument:

```
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m <model_path/model.xml> -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```

#### Running on the Intel® Neural Compute Stick

To run on the Intel® Neural Compute Stick, use the ```-d MYRIAD``` command-line argument:

```
python main.py -d MYRIAD -i resources/Pedestrian_Detect_2_1_1.mp4 -m <model_path/model.xml> -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```

To see the output on a web based interface, open the link [http://0.0.0.0:3000](http://0.0.0.0:3004/) in a browser.

**Note:** The Intel® Neural Compute Stick can only run FP16 models at this time. The model that is passed to the application, through the `-m <path_to_model>` command-line argument, must be of data type FP16.

#### Using a camera stream instead of a video file

To get the input video from the camera, use the `-i CAM` command-line argument. Specify the resolution of the camera using the `-video_size` command line argument.

For example:
```
python main.py -i CAM -m <model_path/model.xml> -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```

To see the output on a web based interface, open the link [http://0.0.0.0:3000](http://0.0.0.0:3004/) in a browser.

**Note:**
Give `-video_size` command line argument according to the input as it is used to specify the resolution of the video or image file.

