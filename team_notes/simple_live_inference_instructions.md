# Simple Live Inference Instructions

## Overview

Simple live inference is achieved by running BytePS to use distributed training to learn model weights while allowing the user to locally request inferences by providing input as that is predicted immediately using the best model achieved so far (according to the development dataset accuracy).  Simple inference retrieves user requests via the terminal of the same device initiating BytePS for training as a proof-of-concept and to avoid network protocol difficulties, however, note that this design can easily support the abstraction of receiving a client message as input instead.

## Setup

1. Follow the Development Setup instructions outlined in [byteps-setup-instructions.md](https://github.com/liangw6/byteps/blob/master/team_notes/byteps-setup-instructions.md)

2. After completing all the configuration steps, run the command ``` export EVAL_TYPE=mnist ```

3. Install OpenCV2 on to the machine to support input image reading with the command ``` pip install opencv-python ```

4. Then run the modified start_pytorch_byteps.sh (only configured for the mnist dataset currently).

This script silences training information and allows live inferences to be made by supplying a file path to the location of an image file stored on the host.  The program currently assumes that the input data is configured the same way as the data the model is using during training.  The best model trained so far by BytePS is then used to provide the prediction on the input data and reported to the user.  This process can repeat indefinitely throughout training and until the user specifies explicitly that they are done requesting inferences (by issuing the 'q' command)

## Example Simple Live Inference Output

The following is an example of how a live inference session would operate after launching BytePS and training a model on the MNIST dataset.

Example image provided:
    
<img src="mnist_example_image_7.png" width="200" height="200" />

Example simple live inference execution for MNIST dataset:

<img src="example_simple_live_inference_execution.png" />




