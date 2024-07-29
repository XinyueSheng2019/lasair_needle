# Lasair version NEEDLE-TH

### Developer
Xinyue Sheng 

Publish date: 2024/01/11

### Environment setup
Build a virtual environment using anaconda/conda, then follow the *requirements.txt* to download all packages.

### Dataset
Available from [Kaggle API](https://kaggle.com/datasets/2135ac690e420c1129df77fc059cbddedc32c684ece3e2091bd4d03a23eb2470)

Download four files and put them into the directory *lasair_20240105/*

### Training 
Run the *Makefile* from the *source* folder.
For each run, it will train 10 models, with half-shuffled test sets.

You could change the architecture parameters in file *config.py*. 
Note in this file, IMAGE_PATH, HOST_PATH, MAG_PATH are set as they are for generating new training and test sets.
If you would like to test the codes, you could modify the MODEL_NAME.

### Get annotation working 
Go to the annotator folder, and run:

    python generate_annotator.py

If you would like to change the models, go to the *generate_annotator.py* and modify the global variables on the top lines.













