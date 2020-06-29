# object-detection-using-ssd-on-construction-sites
Using SSD for object detection on construction sites.
Many use techniques and approches and technique has been introduced since the ssd have been introduced which can improve the traning and prediction. 
In this repository I am improving the SSD approch using better approches like using data agumentation, cyclyic learning rate, building better model architecture and other methods and techniques.


# This repository is using sgrvinod a-PyTorch-Tutorial-to-Object-Detection repository as base for future development. 
Link to the repo- https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection



I have implepented traning of the for three diffrent kind of learning rate. 
1. constant learning rate.
2. learning rate annelling.
3. cyclic learning rate.

The dataset should be in Pascal VOC format.

Pipeline for file running
1. Run create_data_lists.py
2. Run the train file i.e. constlr.py/annelling.py/cyclelr.py
3. Run eval.py to get the map on test set.
4. Run detect.py to get real time object detection inference.


The code is intregated with Weights & Biases AIP so that you can experiment with your model.


# Enjoy
