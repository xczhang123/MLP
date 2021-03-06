## What each file do:
* MultiLayerPerceptron.ipynb: main notebook
* dataset: The folder contains both the training data and test data for the experiments
* data_utils.py: The file contains a data utility function to read in the dataset required for training and testing
* fc_net.py: The file contains the implementation for the MLP model         
* gradient_check: The file contains the codes for numerical gradient checking (against analytical from back propagation)
* layers.py: The file contains all layers implemented in the network
* layer_utils.py: The file contains multiple composite of layers (from layers.py) that are commonly used for ease of access
* optim.py: This file implements various first-order update rules that are commonly used for training neural networks.

## To run the notebook:
* if on Google Colab: 
    Change Line 5 to match your file path
    Navigate to the 'Runtime' bar and click 'Run all' in the dropdown menu
* if on local jupyter notebook:
    Please comment out Line 1 through 8 of the notebook and execute as above

It takes approximately 20-30 minutes to finish running the whole notebook, depending on the hardware.
