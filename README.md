# Deep Sets: digit sum
Sum of handwritten digits (MNIST), implemented in PyTorch.

The approach is based on the [Deep Sets paper](https://arxiv.org/abs/1703.06114), by Zaheer *et al.* The original implementation by the authors (using Keras) is available [here](https://github.com/manzilzaheer/DeepSets/blob/master/DigitSum/image_sum.ipynb).

This repo includes a complete PyTorch implementation of both the permutation equivariant and permutation invariant layers, available in ``deepsetlayers.py``.

## Data acquisition
We use the [InfiMNIST digits](https://leon.bottou.org/projects/infimnist) exactly as described [here](https://github.com/manzilzaheer/DeepSets/tree/master/DigitSum). For your convenience, you may use our `infimnist_parser.py` to parse the InfiMNIST binaries. This was adapted from [this script](https://github.com/CY-dev/infimnist-parser/blob/master/infimnist_parser.py) in order to produce precisely the required files to run our model.

## Usage
The required packages and the respective versions are in the `requirements.txt` file.

Training and testing: `python image_sum.py --train=1`

Testing only: `python image_sum.py`

The following optional arguments may be passed to `image_sum.py`:

    -h, --help         show this help message and exit
    -t , --train       train the model (default: False)
    -m , --model       model file (output of train, input of test) (default: ./mnist_adder_cnn.pth)
    -d , --data_path   data directory path (default: ./infimnist_data)
    --n_train          number of training examples (default: None)
    --n_test           number of test examples (default: 5000)
    --n_valid          number of validation examples (default: 1000)
    --max_size_train   maximum size of training sets (default: 10)
    --min_size_test    minimum size of test sets (default: 5)
    --max_size_test    maximum size of test sets (default: 100)
    --lr               learning rate (default: 0.001)
    --epochs           number of training epochs (default: 100)
    --batch_size       batch size (default: 128)
    --use_cuda         use CUDA capable GPU (default: True)
    --use_visdom       use Visdom to visualize plots (default: False)
    --visdom_env       Visdom environment name (default: MNIST Adder)
    --visdom_port      Visdom port (default: 8888)

## Results
Using the default parameters, one obtains better results than those reported in the paper. This is mostly due to the fact that we use a CNN instead of an MLP to extract image features. The test accuracy for set cardinalities between 5 and 100 is shown below.

<p align="center">
  <img src='mnist_sum_test_acc.png', width="75%">
</p>
