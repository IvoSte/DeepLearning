# DeepLearning
Deep Learning Course

# Requirements
Requires a number of packages. To install them all in one go, run
`pip install -r requirements.txt`

# Folders
- ```originals```: forked code scripts.
- ```output```: folder in which to store output plots of accuracies+losses.
- ```tutorial``` and ```tutorial_piek```: tutorials and scratch files. 
- ```development```: files used for development. Largely don't work. 

# Files
- ```lenet_new.py```: functional lenet implementation. Current version uses SGD as optimizer (original). 
- ```inc_v3.py```: inception-v3 implementation. Not working yet. 
- ```data_loading.py```: provides package ```data_loading```, one (soon to be two) way of loading data. This version uses cv2, and thus takes a while.
- ```train_and_test.py```: provides package ```train_and_test``` for training and testing. Has one function: ```learning()```.

# Past files (now archived into folders)
- ```data_prep.py```: helper functions for data preparation, including three ways of retrieving the data from the directory.
- ```model.py```: currently used to train inception model
- ```resnet.py```: probably redundant and not-tested implementation of resnet
- ```lenet_old.py```: implementation of lenet that works perfectly
- ```lenet.py```: so-far-failed attempts at providing lenet with input that hasn't been specified as x and y.
