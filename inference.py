#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        self.plugin = None
        self.network = None
        self.net_plugin = None
        self.input_blob = None
        self.output_blob= None
        self.infer_request_handle = None


    def load_model(self, model, device="CPU", cpu_extension=None):
        '''
            Load the model given IR files.
            Defaults to CPU as device to use.
        '''

        # Load the model
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        # Initialize the plugin - load the inference engine API
        # print("Initializing plugin.")
        self.plugin = IECore()

        # Add a CPU extension, if applicable
        if cpu_extension and "CPU" in device:
            # print("Adding CPU extension.")
            self.plugin.add_extension(cpu_extension, device)

        # Read the IR as a IENetwork
        # print("Reading the IR as a IENetwork")
        self.network = IENetwork(model=model_xml, weights=model_bin)

        # Check for supported layers
        supported_layers = self.plugin.query_network(network=self.network, device_name="CPU")

        # Check for any unsupported layers, and let the user
        # know if anything is missing. Exit the program, if so.
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            log.error("Unsupported layers found: {}".format(unsupported_layers))
            if cpu_extension:
                log.error("The CPU extension specified does not support some layers. Please specify a new CPU extension.")
            else:
                log.error("Please try to specify a CPU extensions library path by using the --cpu_extension command line argument.")
            sys.exit(1)

        # Load the IENetwork into the plugins
        self.net_plugin = self.plugin.load_network(self.network, device)
        # print("IR successfully loaded into Inference Engine.")

        # Grab output and input blobs
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))

        # Return the loaded inference plugin
        return

    def get_input_shape(self):
        '''
            Get the input shape of the network
        '''

        # Return the shape of the input layer ###
        return self.network.inputs[self.input_blob].shape

    def exec_net(self, image):
        '''
            Make an asynchronous inference request given an input image/frame
        '''
        # Start an asynchronous request
        self.infer_request_handle = self.net_plugin.start_async(request_id=0,
                                                                inputs={self.input_blob: image})
        return

    def wait(self):
        '''
            Check the status of the inference request.
        '''

        # Wait for the request to be complete
        status = self.net_plugin.requests[0].wait(-1)
        return status

    def get_output(self, output=None):
        '''
            Return a list of the results for the output layer of the network.
        '''

        # Extract and return the output results
        if output:
            out = self.infer_request_handle.outputs[output]
        else:
            out = self.infer_request_handle.outputs[self.output_blob]
        return out
