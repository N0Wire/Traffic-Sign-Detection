# Traffic-Sign-Detection

A small tool to detect traffic signs (not all) in images.
Selective Search [1] is used for localization, a Support Vector Machine (SVM) for 
determination of sign or no sign and a Spatial Transformer Network [2] combined 
with a classification Convolutional Neural Networ (CNN) gives classification results.
For training and testing purposes the GTSDB [3] and GTSRB [4] database are used.

#   PARTS OF THE SYSTEM      #

# main.py (OD+KS)
This file contains the main file. You can import any image and run the whole algorithm on this image.
The function in this file will output a plot with the found bounding boxes and the name of the traffic sign.
It takes some time, but try it out! We have included pictures from Heidelberg. :)

# final_eval.py (OD+KS)
This file does the final evaluation of the combined system. For the trainset of the GTSDB the bounding boxes
of the Selective Search + SVM are loaded and fed into the CNN+STN classifier. The accuracy is calculated.
Additionally the result of the STN on some of the bounding boxes is plotted.

# preprocessor_plot.py (OD)
The preprocessor and dataloader is tested. The plot from the project report of the preprocessor result is created.

# cnn_stn_plot.py (OD)
This file can be used to train and save a CNN+STN classifier (which can be used in final_eval and main if flag is changed)
This trained classifier of our pretrained classifier can be loaded and all the evaluation plots are created
that can also be found in the  project report.


# Classifier (OD) #

# dataloader.py
This file contains the image class, which is used internally for saving preprocessed images, the preprocessor, which performs
histogram stretching, resizing and calculates the Canny edge picture. Also the datasets are written that can be used for
the pytorch neutral network input.

# network.py
This file contains our neural network for the CNN+STN classifier with initilization functions.

# network_utils.py
This file contains everything which is needed to make the classifer work. It contains the train function, which trains the system,
it contains an evaluate function to evaluate datasets, it contains a predict function to predict the classid for an image etc.

# load_save.py
This file contains stuff to save and load the datastructures we use (especially the model) and scalar values for later evaluation.

# signname.py
This file contains a function that converts the class ids to the corresponding class name (name of traffic sign).

# trainer.py
This file contains the import_classifier function, which sets up the neural network and all other objects and loads all data.
It returns the final model which can be used on images without much ugly code.

# test.py
Not important, was just for testing.

# /Saved
Contains the saved models.

# /Plots
Contains all plots. All created plots for the classifier are saved there.

# /Temp
The saving checkpoints of the model are saved there.

# /Checkpoints
Not used anymore.


# Detection (KS) #


# Localization (KS) #


################

[1] Selective Search for Object Recognition
    J. R. R. Uijlings, K. E. A. van de Sande, T. Gevers, A. W. M. Smeulders
    International Journal of Computer Vision 2013.
    (https://ivi.fnwi.uva.nl/isis/publications/2013/UijlingsIJCV2013)

[2] Spatial Transformer Networks
    Max Jaderberg, Karen Simonyan, Andrew Zisserman, and Koray Kavukcuoglu
    (https://arxiv.org/pdf/1506.02025v3.pdf)

[3] Detection of Traffic Signs in Real-World Images: The German Traffic Sign Detection Benchmark 
	   Sebastian Houben and Johannes Stallkamp and Jan Salmen and Marc Schlipsing and Christian Igel,
   	International Joint Conference on Neural Networks 2013.
   	(http://benchmark.ini.rub.de/?section=gtsdb&subsection=dataset)

[4] The German Traffic Sign Recognition Benchmark: A multi-class classification competition
    Johannes Stallkamp and Marc Schlipsing and Jan Salmen and Christian Igel,
    IEEE International Joint Conference on Neural Networks 2011.
    (http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)
