# Overview

The objective of this project is to adapt, train and test a Mask R-CNN model to identify occupations in Remote Sensing imagery, applying the developed model into a plug-in for QGis. The state-of-the-art architecture in question was chosen because each occupation, even if reasonably clustered, must be independently identified and accurately delimited, and region based approaches showed good results for detecting close targets.

The project's dataset of training, validation and test samples is limited, containing few km² of manually labelled occupations in the region of Distrito Federal, so for that reason the model will be first trained on a publicly available dataset from <a href="https://www.aicrowd.com/challenges/mapping-challenge/">AIcrowd mapping challenge</a> and then finetuned on Brasília's dataset for better generalization.

The implementation was made through the framework Pytorch and PyQgis, along with other well known modules.

This project was a result of a monograph concerning Machine Learning, Remote Sensing and Object Detection applied to the field of Environmental Engineering. For more information regarding its development refer to <a href="https://1drv.ms/b/s!AqyY2q12MmSlj2LgqJD1Maoq6P9Z?e=MffE5W/">monograph</a> (in portuguese).

# How to

For QGis plug-in usage you must have the modules described in "requirements.txt" installed through OSGeo4W shell, if they aren't already in QGis's PYTHONPATH. Detailed "how to" will be soon added. After that you may copy this repository to QGis's plugins directory, along with one of <a href="https://drive.google.com/drive/folders/1r2LnczsIW_MIuYGiyvUS_joonwyOX4X?usp=sharing/">these weights</a>. Complete tutorial will be added soon.
