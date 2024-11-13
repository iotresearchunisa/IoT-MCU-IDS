# Structure of Results Folders
This section describes the contents of the folders used to organise the results of experiments with the Siamese network.

### `bioinf_code` folder
Contains the results of the Siamese network trained and tested on my dataset, in both the 4-class and 5-class versions.

### `paper_code` folder
This folder contains the results of testing with the paper-based approach, and includes the following subfolders:

- `train_test`: contains the results of the Siamese network trained and tested on both my dataset (4 and 5 classes) and the TON_IoT dataset (4 classes).
- `test`: 
  - `mio - TON_IoT`:
    - `only_test`: contains the results of the Siamese network pre-tested on my dataset (4 and 5 classes) and then tested on the TON_IoT dataset.
    - `transfer_learning`: contains the results of the Siamese network pre-trained on my dataset (4 and 5 classes) and updated using transfer learning on the TON_IoT dataset. The training parameters are set as follows: learning rate (lr) = 0.000001, epochs = 40, patience = 5.
  - `TON_IoT - mio`: contains the results of the Siamese network pre-trained on TON_IoT and tested on the mio dataset with 4 classes.