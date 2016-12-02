# CIFAR ISIC
 - edit the train_dir flag in isic_train.py
 - edit the data_dir flag in cifar10.py
 - edit the dir flags in isic_eval.py to reflect their locations
 - Place the binaries generated with isic_cnn into the data_dir
 - ensure the filenames within the functions inputs and distorted inputs are the same as the names of the binaries generated with isic_cnn
 - ensure that the xrange functions reflect the numbers of binaries generated
 - run isic_train.py (this will begin the training process.  It will save every 1000 steps within your indicated train_dir)
 - if you want to visualize training run tensorboard pointing at the train_dir
 - When sufficient steps are complete (24k to recreate the experiment) terminate the process of training
 - before evaluation change batch_size to 10 in cifar10.py
 - To test run isic_eval.py
 - this will output two csv files for analysis.  (labels and success)


