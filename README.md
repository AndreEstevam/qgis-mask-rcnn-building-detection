# occupation-detection-on-satellite-imagery

The objective of this project is to adapt, train and test a Mask R-CNN model to identify occupations in satellite imagery, applying the developed model into a plug-in for QGis. The state-of-the-art architecture in question was chosen because each occupation, even if reasonably clustered, must be independently identified and accurately delimitated, and region based approaches showed good results for detecting small targets.

The project's dataset of training, validation and test samples is limited, containing few km² of manually labelled occupations in the region of Distrito Federal, so for that reason the model will be first trained on a publicly available dataset from <a href="https://www.aicrowd.com/challenges/mapping-challenge/">AIcrowd mapping challenge</a> and then finetuned on Brasília's dataset for better generalization.

The implementation was made through the framework Pytorch and PyQgis, along with  other well known modules.