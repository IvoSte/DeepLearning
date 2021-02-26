# DeepLearning
Deep Learning Course

# Folders
- ```originals```: forked code scripts.
- ```output```: folder in which to store output plots of accuracies+losses.
- ```tutorial``` and ```tutorial_piek```: tutorials and scratch files. 
- ```development```: files used for development. Largely don't work. 

# Files
- ```lenet\_new.py```: functional lenet implementation. Current version uses SGD as optimizer (original). 
- ```inc\_v3.py```: inception-v3 implementation. Not working yet. 
- ```data\_loading.py```: provides package ```data\_loading```, one (soon to be two) way of loading data. This version uses cv2, and thus takes a while.
- ```train\_and\_test.py```: provides package ```train\_and\_test``` for training and testing. Has one function: ```learning()```.

# Past files (now archived into folders)
- ```data\_prep.py```: helper functions for data preparation, including three ways of retrieving the data from the directory.
- ```model.py```: currently used to train inception model
- ```resnet.py```: probably redundant and not-tested implementation of resnet
- ```lenet\_old.py```: implementation of lenet that works perfectly
- ```lenet.py```: so-far-failed attempts at providing lenet with input that hasn't been specified as x and y.
