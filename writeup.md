# **Behavioral Cloning** 


---

[image1]: ./examples/drive_left.jpg "Left Camera"
[image2]: ./examples/drive_right.jpg  "Right Camera"
[image3]: ./examples/hard_track.jpg "Challenge Road"

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* video.mp4 records one lap of autonomous driving on track 1.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
The model is built with reference to the [NVIDIA end to end framework](https://arxiv.org/abs/1604.07316). 
Image preprocessing is processed within the model framework so that drive.py can take raw images from the simulator directly to generate steering signals.

The first layer normalizes the images using the Lambda layer in keras and the second layer crop the image vertically so that the model focuses on the road section from input images.

The following thress layers are convulational layers with kernel size of $5\times 5$ and stride of $(2,2)$. The number of output filters are 24,36, and 48 respectively.

Then, two convulutional layers with kernel size of $5\times 5$ and stride of $(1,1)$ are added and the number of output filters are 48 for both layers.
3 fully connected layers are added to make the model so that the model output the control signal which is the steering angle in our case.

To avoid overfitting, a dropout layer is applied after the hidden layers are flattened in 1D.
#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 74). 

The model was trained and validated on different data sets with a split ratio of 8:2 model.py line 53) to ensure that the model was not overfitting (model.py line 86-90). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 79).
The dataset included side carmera images and a correction value needs to be applied to the steering angles to make the training data valid. Values ranging from 0.10 to 0.3 with increments of 0.005 are tested and the value is set to 0.25 (model.py line 17).
I tested various number of learning rate for Adam and used a learning rate of 0.0001. 
#### 4. Appropriate training data
I used the sample dataset as the initial training dataset. Side-camera images are also added after correcting for steering angles to expand the training dataset. Then, I included the horizontally reversed images and steering angles. 
While testing the performance of the model in autonomous mode, I kept track of the situations where the model failed to driving properly. Then, I collected more training data by recording demonstrative runs around tough corners. 

Here are the two examples of challenging corners.
In both cases, the model initally failed to apply enough steering to keep the vihecle on the track. I started recordinng the action when the car is on the outer track. I drove the car to the middle of the track and then exist the turn as smoth as possible.

![image1]
![image2]

I also collected some driving data from the challenging track which also helped the model to handle tough corners.

![image3]
### Model Architecture and Training Strategy

#### 1. Solution Design Approach


My first step was to use a convolution neural network model similar to the NVIDIA end-to-end self-driving car framework. I thought this model might be appropriate because it is a well-known application of behavior cloning and has shown success in a physical application.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model by adding a dropout layer. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track especially near the wide turns close to water area. To improve the driving behavior in these cases, I added demonstrations by driving properly and driving to save from failure.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture
My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 image BGR  							| 
|Lambda                 |                                               |
|Crop2D                 | Outputs 90x320x3                              |
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 43x158x24 	|
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 20x77x36 	|
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 8x37x48	|
| RELU					|												|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 6x35x64	|
| RELU					|												|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 4x33x64 	|
| RELU					|												|
| Flattening		    |        									    |
|Dropout                | rate: 0.2                                     |
|Fully connected		| outputs 100        							|
|Fully connected		| outputs 50									|
|Fully connected        | outputs 10                                    |
|Fully connected        | outputs 1                                     |

 

#### 3. Creation of the Training Set & Training Process



