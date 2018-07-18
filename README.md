# Traffic-Sign-Detection

A small tool to detect traffic signs (not all) in images.
Selective Search [1] is used for localization and
a Spatial Transformer Network compared with a CNN gives classification results.
For training and testing purposes the GTSDB [2] and GTSRB [3] database are used.


[1] Selective Search for Object Recognition
    J. R. R. Uijlings, K. E. A. van de Sande, T. Gevers, A. W. M. Smeulders
    In International Journal of Computer Vision 2013.
    (https://ivi.fnwi.uva.nl/isis/publications/2013/UijlingsIJCV2013)

[2] Detection of Traffic Signs in Real-World Images: The German Traffic Sign Detection Benchmark, 
	Sebastian Houben and Johannes Stallkamp and Jan Salmen and Marc Schlipsing and Christian Igel,
   	International Joint Conference on Neural Networks 2013.
   	(http://benchmark.ini.rub.de/?section=gtsdb&subsection=dataset)

[3] The German Traffic Sign Recognition Benchmark: A multi-class classification competition, 
    Johannes Stallkamp and Marc Schlipsing and Jan Salmen and Christian Igel,
    IEEE International Joint Conference on Neural Networks 2011.
    (http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)
