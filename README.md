# PyTorch Deep Learning Utility
This framework runs hyperparameter tuning, training, and prediction on a provided PyTorch model and data. It logs all model performance metrics to a training file, plots performances in real-time with Visdom, save best models while training, and has options for early stop based off your defined criteria.

![alt text](https://github.com/gifford-lab/pytorch-dl-utility/blob/master/resources/visdom.png "Visdom Performance Plotting")

## Overview
In general, the user needs to define three components in a python file (without loss of generality call this file `model.py`):
* Loaders for datasets (there are predefined loaders that handle certain data formats)
* Model network (layers, loss), optimizer, regularizers, initializers, constraints. There are some predefined classes for all of these.
* Hyperparameter search space (optional if hyperparameters are known)

It's also recommended that the user create a data directory `data_dir` where all the data resides.

## Install
```bash
pip install -r requirements.txt
```

## Run
WLOG let the desired result directory (empty at first) be `result_dir`. The first argument is usually `result_dir`

### Training
```
python3 -m visdom.server
```
In a separate shell:
```
cd result_dir && python3 train.py . -f model.py -d data_dir -te train_epoch [-tb train_batch] [-es early_stop] [--debug] [--cpu]
```
Verbose version of the same command:
```
cd result_dir && python3 train.py . --model model.py --data data_dir --train-epoch train_epoch [--train-batch train_batch] [--early-stop early_stop] [--debug] [--cpu]
```

### Evaluation
```
cd result_dir && python3 test.py . [-v] [-pi pred_input_1 pred_input_2 ...] [-po pred_output_1 pred_output_2 ...] [-pe pred_epoch] [-pb pred_batch] [--cpu]
```
Verbose version:
```
cd result_dir && python3 test.py . [--eval] [--pred-in pred_input_1 pred_input_2 ...] [--pred-out pred_output_1 pred_output_2 ...] [--pred-epoch pred_epoch] [--pred-batch pred_batch] [--cpu]
```
The `-v` / `--eval` option evaluates the model on a predefined test set and metrics, while the `-pi` / `--pred-in` and `-po` / `--pred-out` options runs prediction on the list of inputs and save the predictions (no metrics) to the list of outputs

### HyperBand Hyperparameter Search
Given an empty directory `hyperband_dir`, for every hyperparameter set trained, the `result_dir` for this set of hyperparameter will be `hyperband_dir/hyperparameter_set_name`.
The first argument of `hyperband.py` is `hyperband_dir`, otherwise the arguments are the same as `train.py`

```
python3 -m visdom.server
```
In a separate shell:
```
cd hyperband_dir && python3 hyperband.py . -f model.py -d data_dir -te train_epoch [-tb train_batch] [-es early_stop] [--debug] [--cpu]
```

### Additional Arguments
If you need additional arguments for any part of `model.py`, you may use one of the following methods

#### config.json
In `result_dir`, before running `train.py` or `test.py`, create a `config.json` with
```
{
    "variable_1": value_1,
    "variable_2": value_2,
    ...
}
```
A `config.json` is automatically created for you the first time you run `train.py` on `result_dir`. Subsequent runs will always fill in missing variables from this file, so you do not need to specify the arguments for `train.py` that you specified the first run.

#### Environmental Variables
```ENV_VAR_NAME=ENV_VAR_VALUE ... python [train.py, test.py] ...```
Access the env vars by `os.environ['ENV_VAR_NAME']`

## Writing `model.py`
See [examples](https://github.com/gifford-lab/pytorch-dl-utility/tree/master/examples) for examples

### Dataloader
User needs to define several functions for creating loaders for training, validation, test, and prediction data. Training, validation, and test dataloaders iterate over both input and labels, while predict dataloader only iterate over input.

```python
class Model(NNModel):
    ...
    
    def get_train_data(self):
        # don't define get_train_val_data if this is defined
    
    def get_val_data(self):
        # don't define get_train_val_data if this is defined
    
    def get_train_val_data(self):
        # don't define get_train_data and get_val_data if this is defined

    def get_test_data(self):
    
    def get_pred_data(self):
    
    ...
```

Each method creates a data generator for their respective tasks, which is a subclass of [`src.generators.matrix.MatrixGen`](https://github.com/gifford-lab/pytorch-dl-utility/blob/master/src/generators/matrix.py#L8) or a generator over `(x, y)` when `y` is provided and over `(x, None)` when `y` is not provided. The command line arguments can be found as bound variables in `self.config`, for example the value for `--train-batch` is in `self.config.train_batch`. The `--cpu` and `--debug` values can be found in `self.device` and `self.debug`

### Netork
User must define a `Network` class that defines the network to return a loss if a label is provided. In addition, the network should define optimizer and any regularizers, initializers, and constraints as lists in `__init__`.

```python
class Network(nn.Module):

    def __init__(self, config):
        # you can pass in parameters via config or via environmental variables
        super(Network, self).__init__()
        self.layers = # define pytorch layers
        self.loss = lambda y_true_t, y_pred_t:  # anything that takes the tensors y_true_t and y_pred_t and returns a tensor loss scalar. See https://pytorch.org/docs/stable/nn.html#loss-functions for examples
        self.optimizer = # See https://pytorch.org/docs/stable/optim.html
        self.regularizers = [] # See https://github.com/gifford-lab/pytorch-dl-utility/blob/master/src/matchers/regularizers.py
        self.initializers = [] # See https://github.com/gifford-lab/pytorch-dl-utility/blob/master/src/matchers/initializers.py
        self.constraints = [] # See https://github.com/gifford-lab/pytorch-dl-utility/blob/master/src/matchers/constraints.py

    def forward(self, x_t, y_t):
        y_pred_t = self.layers(x_t)

        if y_t is not None:
            loss_t = self.loss(y_pred_t, y_t)
        else:
            loss_t = None
        pred_t = { # this return value is converted to numpy then fed into Model.train_metrics and Model.eval_metrics during training / evaluation
            'loss': loss_t,
            'y': y_pred_t
        }
        return loss_t, pred_t
```

### Model
Model serves as a wrapper for defining the hyperparameter search space, network initialization, calculating metrics

```python
class Model(NNModel):
    @classmethod
    def get_params(cls, config):
        # returns a dictionary of hyperparameters for hyperparameter search (this can be randomly sampled). These will be added to config as bound variables
        return dict(hyperparam1=1, hyperparam2=2)

    def init_model(self):
        network = Network(self.config) # anything that implements nn.Module. See https://pytorch.org/docs/stable/nn.html#containers
        self.set_network(network)
        # any other initialization

    def train_metrics(self, y_true, pred):
        """
        Takes in numpy arrays and return a dictionary of metrics.

        :param y_true: a batch of labels from the train data loader
        :param pred: output (converted to numpy) of the network on a batch of inputs from the train data loader. This is second return value of Network.forward, converted to numpy
        :returns: a dictionary mapping strings to scalar values
        """

    def eval_metrics(self, y_true, pred):
        """
        Same as Model.train_metrics, but for validation and test
        """

    def reward(self, epoch_result):
        """
        If not implemented, the NNModel parent class will choose to return -epoch_result['val_loss']
        :param epoch_result: mapping from outputs of train_metrics (prefixed by 'train_') and eval_metrics (prefixed by 'val_')
        :returns: the scalar value that the Hyperband algorithm wants to maximize
        """
```

User can extend predefined [models](#models) and [networks](#networks) for convenience, and users can override any additional methods of `models.Model` or `models.nn_model.NNModel` for custom functionality.

## Defined Classes
### Generators
See [`src.generators`](https://github.com/gifford-lab/pytorch-dl-utility/blob/master/src/generators) for definition. In general takes in input as `X` and labels as `Y` (optional for when there is no labels), give option for shuffling, and iterate over `batch_size` samples at a time.

#### `src.generators.matrix.MatrixGen`
For when `X` and `Y` are numpy matrices.

#### `src.generators.h5py.H5pyGen`
When data is in the form of several h5py files. Each file contains 'data' key for `X` and 'label' key for `Y`. Concatenates `X` and `Y` for all h5py files together.

### Models
See [`src.models`](https://github.com/gifford-lab/pytorch-dl-utility/tree/master/src/models) for definitions

#### src.models.Model
Contains the barebone model method interface

#### src.models.nn_model.NNModel
Contains most of the training, evaluation, and prediction logic. When subclassing, need to implement methods `init_model`, `train_metrics`, `eval_metrics`, and `reward`. Implement classmethod `get_params` if want to use hyperparameter search.

#### src.models.regression_model.RegressionModel
Model for regression that defines mean squared error and pearson correlation as metrics, and the negative validation loss as reward. Need to implement `init_model` with a network. See [ConvRegressionNetwork](#convregressionnetwork) for sample network implementation with convolution

#### src.models.classification_model.ClassificationModel
Model for classification that defines accuracy and auroc as metrics, and the negative validation loss as reward. Need to implement `init_model` with a network.

### Networks
See [`src.networks`](https://github.com/gifford-lab/pytorch-dl-utility/tree/master/src/networks) for definitions

#### src.networks.conv_regression_network.ConvRegressionNetwork
Define `features` and `regressor` where `features` is a sequential series of operation on batches with shape (num_samples, depth, height, width), the output of `features` is flattened, and `regressor` is a sequential series of operations on tensors of shape (num_samples, num_features) that result in a (num_samples) shaped output. Applies `nn.MSELoss`. Returns loss as the first return value and a dictionary of other desired tensors as the second.

#### src.networks.conv_regression_network.ConvClassificationNetwork
Similar as `ConvRegressionNetwork`. Applies `nn.CrossEntropyLoss`

### Matchers (Regularizers, Initializers, Constraints)
See [`src.matchers`](https://github.com/gifford-lab/pytorch-dl-utility/tree/master/src/matchers) for definitions. In general matchers are operations that apply to all layers with a matching name toa glob string or a matching layer type

## Adversarial Training
Uses Projected Gradient Descent (PGD), a k-step iterative version of the Fast Gradient Sign Method (FGSM), to generate adversarial examples to be included in the training set for adversarial training. See examples of adversarial training with corresponding bash scripts in `examples/` subdirectory.

## TFBS Motif Detection + Saliency Analysis
See the following examples (bash scripts) for each of the below components of the pipeline.

### Training + Testing Binary Classifiers
See `examples/tfbs_saliency/tfbs_train.sh` for training.   
See `examples/tfbs_saliency/tfbs_test.sh` for testing.

### Calculate Saliency Scores + Create PWMs
Performs integrated gradients on a set of sequences from a given set of HDF5 files and writes both the sequences and their corresponding nucleotide-level saliency scores to files. The script then uses these sequences and saliency scores to detect subsequences with high saliency using a sliding window approach (i.e. extracting the subsequences that correspond with the highest integrated gradients values). Subsequences with integrated gradient values that are above the 80th percentile of scores are retained, while those that do not satisfy this criterion are discarded. We perform this thresholding to ensure that the most salient subsequences are more highly represented. (Note: This thresholding can be easily removed if desired.) The remaining subsequences are then aggregated together to generate a PWM for each dataset/experiment.   
See `examples/tfbs_saliency/motif.sh` for example usage.

### Compare Generated PWMs with Database PWMs using TOMTOM
See `examples/tfbs_saliency/meme.sh` for example that employs Saber's `match_pwm.py` script.

### Aggregate Results
Collects the outputs of the TOMTOM analysis across all experiments and organizes them under a single file.  
See `examples/tfbs_saliency/collect_tomtom.sh`. 

## Examples
Some examples can be found in the `examples/` subdirectory.

### conv_seq_classifier
#### Preprocessing
```bash
cd examples/conv_seq_classifier
mkdir data
for dtype in 'train' 'valid' 'test'
do
	paste - - -d' ' < raw_data/$dtype.fa > data/tmp.tsv
	python embedH5.py data/tmp.tsv raw_data/$dtype.target data/$dtype.h5
done
```

#### Running Model
```bash
python ../../main.py -f conv_model.py -d data -r results -y -e -p test.h5.batch* -ep 20 -bs 100
```

### function_regressor
#### Running Model
```bash
python ../../main.py -f reg_model.py -d data -r results -y -e -p X_pred.csv -ep 50 -bs 100
```
