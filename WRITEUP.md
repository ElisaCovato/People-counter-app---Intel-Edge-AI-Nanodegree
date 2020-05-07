# Project Write-Up
The aim of the Intel®'s project was to deploy a **People Counter App** using the OpenVINO™ Toolkit. 

The app performs inference on an input video, extracts and analyzes the output data, then sends these to a server. The inference models has been deployed on the edge, in this way only data on 1) the number of people in the frame, 2) time those people spent in frame, and 3) the total number of people counted are sent to a MQTT server.

Below we report some OpenVINO™ Toolkit features and their impact on performance, as well as possible use cases and details on the inference models used.

## Explaining Custom Layers
*Layer* is a general term that applies to a collection of 'nodes' operating together at a specific depth within a _neural network_.

When converting a model from its original framework to an Intermediate Representation (**IR**), some of the layers might not be _supported_. Any layer that is not in the list of the supported layers for that framework, it is automatically classified as **custom layer** by the Model Optimizer.

Adding custom layers depends on the original model framework. One common option for all the frameworks, is to add the custom layer as _extension_. The process behind converting custom layers as extension involves:

1. **Generate**: Use the Model Extension Generator to generate the Custom Layer Template Files.

2. **Edit**: Edit the Custom Layer Template Files as necessary to create the specialized Custom Layer Extension Source Code.

3. **Specify**: Specify the custom layer extension locations to be used by the Model Optimizer or Inference Engine.

Other options to convert custom layers depends on the specific framework. For example, in the case of Tensorflow frameworks, one option is to replace the unsupported subgraph/layer with a different subgraph. 

Some of the reasons for handling custom layers is to allow the Model Optimizer to build the model's internal representation, optimize the model, and producing the IR files. In the case of the Inference Engine, implementing custom layers guarantees that the input model IR files are loaded into the specified device plugin.

## Comparing Model Performance

To compare models before and after conversion to Intermediate Representations I've first run the inference with the downloaded TensorFlow models, then I've converted the models to an IR file and run the inference on the same input.  

The difference between model accuracy pre- and post-conversion was almost the same, indeed converting a model to an IR files has minimum effects on loss of accuracy.

Converting the model to an IR has helped reducing the size of the model of about 2MB.

Inference time of model post-conversion was almost twice as fast when compared to inference time pre-conversion.

The performance between the pre-conversion and post-conversion model is also due to a difference between cloud and edge computing, respectively. Indeed:
#### Cloud computing:
- Cloud Computing is more suitable for projects which deal with massive data storage where latency is not a major concern.
- Cloud processing power is nearly limitless. 
- The Cloud can be accessed from anywhere on multiple devices, making ideal for projects that need to be widely distributed and viewed on a variety of platforms. 
- Because Network communications are expensive (bandwidth, power consumption, etc.) and sometimes impossible (like in remote locations or during natural disasters), cloud computing can be very cost expensive and not always adapt for user needs.
#### Edge computing:
- Edge Computing is regarded as ideal for operations with extreme latency concerns, for example when applications need rapid data sampling or must calculate results with a minimum amount of delay.
- When there is not enough or reliable network bandwidth to send the data to the cloud, edge computing is a better option than cloud computing.
- However, Edge Computing requires a robust security plan with advanced authentication methods to avoid external attacks.

## Assess Model Use Cases

Some of the potential use cases of the _people counter app_ are:

- **Track activity in retail or departmental store**: understand how many customers per day and in which areas of the stores customers spend most of the time. It could help to reconfigure the store to avoid crowded areas and make better use of the space.
- **Monitor factory / building work spaces**: similar to the above use case, it would allow understanding where in work space people gather the most and how much time to they spend. Moreover, it could also be used for security reasons, monitoring the sensitivity areas of the building. 


## Assess Effects on End User Needs
When deploying an edge model, it’s very important to consider end user needs. These needs will have an effect on the type of model to use (speed, size, accuracy, etc.), what information to send to servers, security of information, etc. Also the environment (lighting, position of the camera, etc) and the hardware used (camera quality, focal length, image size, etc) have an impact: in some cases, higher accuracy might be needed resulting in a more resource-intensive app; in other cases lower-power devices will need less accuracy in order to get a lighter, faster app. 
## Model Research

In investigating potential people counter models, I tried 3 [pre-trained Tensorflow models](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md):

- Model 1: **SSD MobileNet V2 COCO** (Download it [here](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz))
  
- Model 2: **SSD Lite MobileNet V2 COCO** (Download it [here](http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz))

- Model 3: **SSD Inception V2 COCO** (Download it [here](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz))

The three models have been converted into an Intermediate Representation with the following command (run from inside the model directory):

    python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py \
    --input_model frozen_inference_graph.pb \
    --tensorflow_object_detection_api_pipeline_config pipeline.config \
    --reverse_input_channels \
    --transformations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json


All three models were insufficient for the app because of missing detecting boxes: for some frames the model was not detecting any person. Due to this, the _total count_ was not exact: the count was increasing everytime a box was missing in a frame and reappearing in the next one. I have tried to improve the model detection by lowering the probability thereshold, and then by skipping the undetected frames. However this has not helped increasing the model accuracy.

The final inference has been run using the [person-detection-retail-0013](https://docs.openvinotoolkit.org/latest/_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html) model from openvino model zoo. This pre-trained model, already in IR format, performed faster and with a better accuracy than the three models reported above.
