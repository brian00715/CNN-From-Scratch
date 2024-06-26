
# CNN-From-Scratch

This repository contains the source code for the ME5411 continuous assessment project.

Key Features:
- [x] Implemented some fundamental image processing functions from scratch, located in `src/image_core`.
- [x] Developed a custom CNN framework without using off-the-shelf libraries, including customizable model building, forward and backward propagation, parameter optimization using Stochastic Gradient Descent (SGD), etc., see `src/cnn_core`. Also implemented a digit recognition application.
- [x] Created an automatic hyperparameter tuning pipeline for conducting comparison experiments.

Team members (sort by first name):
| Name          | ID        |
| ------------- | --------- | 
| LI HAOWEI     | A0285319W | 
| LIN YI        | A0285080E | 
| SIMA KUANKUAN | A0284990M | 

## Code structure
``` shell
.
├── data                   # dataset
│   ├── p_dataset_26       # default dataset
│   ├── train              # training set
│   ├── test               # testing set
│   ├── train.mat          # training set in .mat format
│   └── test.mat           # testing set in .mat format
├── docs
├── logs                   # training logs. Each log is a folder named by timestamp, containing hyperparameters and training result.
│   └── clean_empty.py     # clean empty folders
├── src
│   ├── cnn_core           # core functions of CNN
│   ├── image_core         # core functions of image processing
│   ├── utils              # utility functions
│   ├── viz_error.m        # visualize mis-classified samples
│   ├── viz_featmap.m      # visualize feature maps
│   ├── viz_result.m       # visualize training result
│   ├── task1_6.m          # main entrance for task 1-6
│   ├── task7_1.m          # main entrance for task 7.1
│   ├── task7_2.m          # main entrance for task 7.2
│   └── task8.m            # main entrance for task 8
└── README.md

```

> Note: task7_2.m should be put at the same path as p_dataset_26 folder
Hyperparameters can be changed at very beginning. The network uses sigmoid as the activation of every layer except for the last layer where softmax was applied.
Currently there is no way to specify activation for each layer, which can be added if needed.
