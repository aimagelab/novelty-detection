from datasets.mnist import MNIST
from datasets.cifar10 import CIFAR10
from models.LSA_mnist import LSAMNIST
from models.LSA_cifar import LSACIFAR10
from result_helpers import OneClassResultHelper
from utils import set_random_seed


def test_mnist():
    # type: () -> None
    """
    Performs One-class classification tests on MNIST
    """
    set_random_seed(30101990)

    # Build dataset and model
    dataset = MNIST(path='data/MNIST')
    model = LSAMNIST(input_shape=dataset.shape, code_length=64, cpd_channels=100).cuda().eval()

    # Set up result helper and perform test
    helper = OneClassResultHelper(dataset, model, checkpoints_dir='checkpoints/mnist/', output_file='mnist.txt')
    helper.test_one_class_classification()


def test_cifar():
    # type: () -> None
    """
    Performs One-class classification tests on CIFAR
    """
    set_random_seed(30101990)

    # Build dataset and model
    dataset = CIFAR10(path='data/CIFAR10')
    model = LSACIFAR10(input_shape=dataset.shape, code_length=64, cpd_channels=100).cuda().eval()

    # Set up result helper and perform test
    helper = OneClassResultHelper(dataset, model, checkpoints_dir='checkpoints/cifar10/', output_file='cifar10.txt')
    helper.test_one_class_classification()


test_cifar()
