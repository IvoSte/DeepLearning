# DeepLearning
Deep Learning Course

# Requirements and running
Requires a number of packages. To install them all in one go, run
`pip install -r requirements.txt`

To recreate the project, run
`python3 main.py`
You will be greeted with an option menu for the different models and experiments. 

# Folders
- `output`: folder in which to store output plots of accuracies+losses.
- `development`: files used for development. Original forked codes, tutorials, scratch files, unfinished models, etc. Largely don't work. 

# Files
- `main.py`: main program to run. 
- `lenet_new.py`: functional lenet models. 
- `data_loading.py`: provides package `data_loading`. This version uses cv2, and thus takes a while.
- `train_and_test.py`: provides package `train_and_test` for training and testing. Has one function: `learning()`.