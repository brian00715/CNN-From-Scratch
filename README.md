
# NUS-ME5411-Group1

This repo contains all codes about Continuous Assignment (CA) of NUS-ME5411.

Team members (sort by first name):
| Name          | ID       | Assignment                                                               |
| ------------- | -------- | ------------------------------------------------------------------------ |
| LI HAOWEI     |          | Implement principle function of task 2,3.<br />Try GUI (optional).       |
| LIN YI        |          | Implement principle function of task 5,6.                                |
| SIMA KUANKUAN | E1192548 | Implement task 1-6 based on MATLAB API and principle function of task 4. |

Group meeting time: 3pm weekly.

mlp1.m should be put at the same path as p_dataset_26 folder
Hyperparameters can be changed at very beginning. The network uses sigmoid as the activation of every layer except for the last layer where softmax was applied.
Currently there is no way to specify activation for each layer, which can be added if needed.

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
│   ├── logs               # training logs. Each log is a folder named by timestamp, containing hyperparameters and training result.
│   └── clean_empty.py     # clean empty folders
├── src
│   ├── cnn_core           # core functions of CNN
│   ├── utils              # utility functions
│   ├── ME5411_Project.m   # main entrance
│   ├── mlp1.m             # letter recognition by MLP
│   ├── OtsuThres.m        # Otsu thresholding
│   ├── train.m            # train entrance for CNN
│   ├── train_toolbox.m    # train by MATLAB toolbox
│   ├── viz_error.m        # visualize mis-classified samples
│   ├── viz_featmap.m      # visualize feature maps
│   └── viz_result.m       # visualize training result
└── README.md

```