# PyTorch Deep Learning Utility
This framework runs hyperparameter tuning, training, and prediction on a provided PyTorch model and data.

## Overview
In general, the user needs to define three components in a python file (without loss of generality call this file `def.py`):
* Loaders for datasets (there are predefined loaders that handle certain data formats)
* Model network (layers, loss) and optimizer (there are some predefined networks and models for certain tasks)
* Hyperparameter search space

It's also recommended that the user create a data directory `data_dir` where all the data resides.

## Install
```bash
pip install -r requirements.txt
```

## Run
```
python main.py [--help] --model def.py [--hyper] [--eval] [--debug] [--cpu] --data data_dir --result result_dir --predict pred_subpath --epoch num_epochs --batch-size training_batch_size
```

### Environmental Variables
If you need additional arguments for any part of `def.py`, you may do so with `ENV_VAR_NAME=ENV_VAR_VALUE ... python main.py ...`.


## Writing `def.py`
See https://github.com/gifford-lab/pytorch-dl-utility/blob/master/example/model.py for an example of `def.py`.

### Dataloader
User needs to define several functions for creating loaders for training, validation, test, and prediction data. Training, validation, and test dataloaders iterate over both input and labels, while predict dataloader only iterate over input.

```python
def get_train_generator(data_dir, batch_size)
def get_val_generator(data_dir)
def get_test_generator(data_dir)
def get_pred_generator(pred_path)
```

User must implement `get_train_generator` to train and hyperparam search the model (`-y` flag), `get_val_generator` to provide a validation set while training, `get_test_generator` to test trained model on a held out set (`-e` flag), and `get_pred_generator` to run prediction on label-less data.

`data_dir`, `pred_subpath`, and `batch_size` are passed in via command line flags, and `pred_path = data_dir + pred_subpath`. Validation, test, and prediction will run model on entire dataset at once. See [environmental variables](#environmental-variables) for passing in arbitrary arguments for loading your data.

Each function returns a subclass of `[batch_generator.MatrixBatchGenerator]`(https://github.com/gifford-lab/pytorch-dl-utility/blob/master/batch_generator.py#L10) or a generator over (x, y) when y is provided and over x when y is not provided. See [predefined subclasses](#batch-generators) of `MatrixBatchGenerator`. Alternatively, `get_train_generator` may return tuple `(train_loader, val_loader)` (do not define `get_val_generator`), which lets user split the data after loading.

### Model (Network, Optimizer)
User must define a `Model` class that wraps around the network and optimizer definitions. `Model` should extend `models.base_model.BaseModel` or one of its subclasses. The methods that need to be implemented are as follows:

```python
class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.loss = lambda y_true_t, y_pred_t:  # anything that takes the tensors y_true_t and y_pred_t and returns a tensor loss scalar. See https://pytorch.org/docs/stable/nn.html#loss-functions for examples

    def forward(self, x_t, y_t):
        y_pred_t = # apply layers to x_t

        if y_t is not None:
            loss_t = self.loss(y_pred_t, y_t)
        else:
            loss_t = None
        pred_t = { # this return value is converted to numpy then fed into Model.train_metrics and Model.eval_metrics during training / evaluation
            'loss': loss_t,
            'y': y_pred_t
        }
        return loss_t, pred_t

class Model(BaseModel):
    def init_model(self):
        config = self.config
        network = Network() # anything that implements nn.Module. See https://pytorch.org/docs/stable/nn.html#containers
        optimizer = # See https://pytorch.org/docs/stable/optim.html
        constraints = # See [predefined constraints](#constraints)
        super(Model, self).init_model(network, optimizer, constraints=constraints)
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

    def get_hyperband_reward(self, result):
        """
        :param result: mapping from outputs of train_metrics (prefixed by 'train_') and eval_metrics (prefixed by 'val_')
        :returns: the scalar value that the Hyperband algorithm wants to maximize
        """
```

User can extend [predefined models and networks](#models) for convenience, and users can override any additional methods of `models.base_model.BaseModel` for custom functionality.

### Hyperparameter Search Space
Hyperparameters should be a randomly sampled dictionary from the hyperparameter space.

```python
def get_config():
    return {
        'hyperparam1': random.choice([])
    }
```

## Defined Classes
### Batch Generators
See `batch_generator.py` for definition. In general takes in input as `X` and labels as `Y` (optional for when there is no labels), give option for shuffling, and iterate over `batch_size` samples at a time.

#### `MatrixBatchGenerator`
For when `X` and `Y` are numpy matrices.

#### `H5pyBatchGenerator`
When data is in the form of several h5py files. Each file contains 'data' key for `X` and 'label' key for `Y`. Concatenates `X` and `Y` for all h5py files together.

### Models
See `models/` for definitions.

#### BaseModel
Contains most of the training, evaluation, and prediction logic. When subclassing, need to implement `init_model`, `train_metrics`, `eval_metrics`, and `get_hyperband_reward`. Also contains logic for saving and loading results and model weights.

#### RegressionModel
Model for regression that defines mean squared error and pearson correlation as metrics, and the negative validation loss as reward. Need to implement `init_model`. See [ConvRegressionNetwork](#convregressionnetwork) for sample network implementation with convolution

#### ClassificationModel
Model for classification with `nn.CrossEntropy` loss that defines accuracy and auroc as metrics, and the negative validation loss as reward. Need to implement `init_model` with a network and optimizer. The `forward` method of the Network must return a dictonary with `y_pred` (prediction with softmax) and `y_pred_loss` (prediction without softmax) as keys; see [ConvClassificationNetwork](#convclassificationnetwork) for a sample network implementation with convolution.


### Networks
See `base_model.py` for definitions.

#### ConvRegressionNetwork
Define `features` and `regressor` where `features` is a sequential series of operation on batches with shape (num_samples, depth, height, width), the output of `features` is flattened, and `regressor` is a sequential series of operations on tensors of shape (num_samples, num_features) that result in a (num_samples) shaped output. Applies `nn.MSELoss`, and returns the loss and prediction in a dictionary.

#### ConvClassificationNetwork
Define `features` and `classifier` where `features` is a sequential series of operation on batches with shape (num_samples, depth, height, width), the output of `features` is flattened, and `classifier` is a sequential series of operations on tensors of shape (num_samples, num_features) that result in a (num_samples, 2) shaped output (without softmax applied). Applies `nn.CrossEntropyLoss`, and returns the loss and prediction in a dictionary.

### Constraints
See `constaints.py` for definitions. In general constraints apply to all layers with weights matching a certain glob string.

#### MaxNormConstraint
Applies a max norm constraint.

## Adversarial Training
Uses Projected Gradient Descent, a $k$-step iterative version of the Fast Gradient Sign Method (FGSM), to generate adversarial examples to be included in the training set for adversarial training.
$\boldsymbol{x}^{t+1} + \epsilon \text{sign}(\nabla_{\boldsymbol{x}}\ell(f(\boldsymbol{x}^t),y))$ for $t=0,...,k-1$, where $\boldsymbol{x}^0=\boldsymbol{x}$
See examples of adversarial training with corresponding bash scripts in 'examples/' subdirectory.

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
