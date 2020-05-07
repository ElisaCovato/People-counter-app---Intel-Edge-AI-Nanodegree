"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
# HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname("localhost")
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser

def draw_boxes(frame, result):
    '''
        Draw bounding boxes around object when its probability
        is more than the specified one

        @param: frame The original input frame where to draw the boxes.
        @param: result The inferencing result.
    '''
    current_count = 0
    for obj in result[0][0]:
        if obj[2] > prob_threshold:
            xmin = int(obj[3] * width)
            ymin = int(obj[4] * height)
            xmax = int(obj[5] * width)
            ymax = int(obj[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 55, 255), 1)
            current_count += 1

    # Return original frame with box on it and
    # the number of object (people) on the current frame
    return frame, current_count


def connect_mqtt():
    '''
        Connect to the MQTT client.
    '''

    # Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client



def infer_on_stream(args, client):
    '''
        Initialize the inference network, stream video to network,
        and output stats and video.

        :param args: Command line arguments parsed by `build_argparser()`
        :param client: MQTT client
        :return: None
    '''

    # Initialise the Inference Network
    log.info("Initialize the inference Network:")
    infer_network = Network()



    # Load the model through `infer_network`
    # print("Loading the model through the Inference Network.")
    infer_network.load_model(args.model, args.device, args.cpu_extension)
    net_input_shape = infer_network.get_input_shape()
    # Handle the input stream
    # (cv2.VideoCapture  accepts either 0 for webcam streams,
    # or a path for an input  image or video file)
    # Flag for the input image
    single_image_mode = False
    # Check if input image is a webcam.
    if args.input == 'CAM':
        input_stream = 0
    # Checks for input image
    elif (args.input.endswith('.jpg') or args.input.endswith('.bmp') ) and os.path.isfile(args.input):
        single_image_mode = True
        input_stream = args.input
    # Check for video file
    elif (args.input.endswith('.avi') or args.input.endswith('mp4') ) and os.path.isfile(args.input) :
        input_stream = args.input
    # Return message error if input file does not exist
    else:
        log.error("Specified input file does not exist.")
        sys.exit(1)

    # Get and open video capture
    cap = cv2.VideoCapture(input_stream)
    cap.open(input_stream)

    # Grab the shape of the input
    global width, height
    width = int(cap.get(3))
    height = int(cap.get(4))

    # Set Probability threshold for detections
    global prob_threshold
    prob_threshold = args.prob_threshold

    # Initialise people count and duration variables
    # Last confirmed count of object on frame
    last_count = 0
    # Count holder
    count = 0
    # Total number of people detect on input video
    total_count = 0


    # Number of consecutive frames with same number of people on it
    consec_frames_counter = 0
    # Number of frames a person was on screen
    frames_counter = 0
    # Number of frames per seconds
    fps = int(cap.get(cv2.CAP_PROP_FPS))


    # Loop until stream is over
    while cap.isOpened():
        # Read from the video capture
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        # Pre-process the image as needed
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)


        # Start asynchronous inference for specified request
        inf_start = time.time()
        infer_network.exec_net(p_frame)

        # Wait for the result
        if infer_network.wait() == 0:
            # Get the results of the inference request
            det_time = time.time() - inf_start
            result = infer_network.get_output()
            # Extract any desired stats from the results
            frame, current_count = draw_boxes(frame, result)

            # Calculate and send relevant information on:
            # current_count, total_count and duration to the MQTT server.
            # Topic "person": keys of "count" and "total"
            # Topic "person/duration": key of "duration"
            inf_time_message = "Inference time: {:.3f}ms".format(det_time * 1000)
            cv2.putText(frame, inf_time_message, (15, 25),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (200, 10, 10), 1)

            # Increase frame counter to count duration
            frames_counter += 1

            # Number of people in the frame has changed
            # the consecutive frames counter is set to 0
            if current_count != count:
                count = current_count
                consec_frames_counter = 0

            # Number of people in frame did not changed
            else:
                consec_frames_counter += 1
                # if the number of people on frames did not change for
                # more than 5 frames (roughly 1/2 sec)
                if consec_frames_counter >= 5:
                    # if the number of people increased since
                    # last confirmed count we increase the total
                    # and reset the frame counter
                    if current_count > last_count:
                        total_count += current_count - last_count
                        # Reset count of frames / duration for the new
                        # people in frame
                        frames_counter = 0
                        client.publish('person',
                                       payload=json.dumps({
                                           'count': current_count,
                                           'total': total_count}))
                    # if some people left, we compute the average duration
                    elif current_count < last_count:
                        duration = int(frames_counter/fps)
                        client.publish('person',
                                       payload=json.dumps({
                                           'count': current_count}))
                        if duration is not None:
                            client.publish('person/duration',
                                           payload=json.dumps({'duration': duration}))
                    # Set the last confirmed count to the current one
                    last_count = current_count


        # Send the frame to the FFMPEG server
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

        if key_pressed == 27:
            break

        # Write an output image if `single_image_mode`
        if single_image_mode:
            cv2.imwrite('output_image.jpg', frame)

    # Release the capture and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    # Disconnect from MQTT
    client.disconnect()

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
