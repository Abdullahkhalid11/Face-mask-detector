# Face-mask-detector
Python script and trained model for face detection with and without a mask
The following are the steps to set up a system to run this script. Feel free to skip any step to jump right into the fun part.
1.	Installing python
2.	Setting up python variable.
3.	Installing required libraries.
4.	Running the script.
Installing the Python:
The program is written in python so python must be installed. Download and install python from the link Python Download.
Note: My version of python is 3.7.6
Setting up python variable:
Setting up environment variable is necessary so that the system knows that python is installed and can be accessed through a command prompt. Use the following link to set up environment variable Set up Path Variable.
Installing required libraries:
Required libraries for this project are open-CV, numpy, tensor flow.
My system is set up with the following versions of these libraries.
open-cv == 4.1.2
numpy == 1.16.0
tensorflow == 1.15.0-rc3
To install a library, use the following command:
pip install <library_name>



For example
pip install opencv-python
pip install numpy
pip install tensorflow

To install a specific version, use the following command:
pip install opencv-python==
The following is the output of the command.
 
This above command will give an error and will list all the available version and choose the version that best matches the above listed version. For example, your opencv list does not have 4.1.2 but has 4.1.3 so write 
pip install opencv-python==4.1.3
To installing MTCNN use the following command:
pip install mtcnn

Running the script:
Open the folder that has the model and the video file you want to test on and type ‘cmd’ here
 
On cmd write the following command.
face mask detector>python face_mask_detector.py --video cctv_2.mp4 --model mask_detector.model --confidence 78

where cctv_2.mp4 is the path of the video file, mask_detector.model is the path to the model file and confidence[optional] is the minimum probability to filter weak detections.

