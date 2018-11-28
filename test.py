import argparse
from argparse import Namespace

from datasets import CIFAR10
from datasets import MNIST
from datasets import SHANGHAITECH
from datasets import UCSDPed2
from models import LSACIFAR10
from models import LSAMNIST
from models import LSAShanghaiTech
from models import LSAUCSD
from result_helpers import OneClassResultHelper
from result_helpers import VideoAnomalyDetectionResultHelper
from utils import set_random_seed


def test_mnist():
    # type: () -> None
    """
    Performs One-class classification tests on MNIST
    """

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

    # Build dataset and model
    dataset = CIFAR10(path='data/CIFAR10')
    model = LSACIFAR10(input_shape=dataset.shape, code_length=64, cpd_channels=100).cuda().eval()

    # Set up result helper and perform test
    helper = OneClassResultHelper(dataset, model, checkpoints_dir='checkpoints/cifar10/', output_file='cifar10.txt')
    helper.test_one_class_classification()


def test_ucsdped2():
    # type: () -> None
    """
    Performs video anomaly detection tests on UCSD Ped2.
    """

    # Build dataset and model
    dataset = UCSDPed2(path='data/UCSD_Anomaly_Dataset.v1p2')
    model = LSAUCSD(input_shape=dataset.shape, code_length=64, cpd_channels=100).cuda().eval()

    # Set up result helper and perform test
    helper = VideoAnomalyDetectionResultHelper(dataset, model,
                                               checkpoint='checkpoints/ucsd_ped2.pkl', output_file='ucsd_ped2.txt')
    helper.test_video_anomaly_detection()


def test_shanghaitech():
    # type: () -> None
    """
    Performs video anomaly detection tests on ShanghaiTech.
    """

    # Build dataset and model
    dataset = SHANGHAITECH(path='data/shanghaitech')
    model = LSAShanghaiTech(input_shape=dataset.shape, code_length=64, cpd_channels=100).cuda().eval()

    # Set up result helper and perform test
    helper = VideoAnomalyDetectionResultHelper(dataset,
                                               model,
                                               checkpoint='checkpoints/shanghaitech.pkl',
                                               output_file='shanghaitech.txt')
    helper.test_video_anomaly_detection()


def parse_arguments():
    # type: () -> Namespace
    """
    Argument parser.

    :return: the command line arguments.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset', type=str,
                        help='The name of the dataset to perform tests on.'
                             'Choose among `mnist`, `cifar10`, `ucsd-ped2`, `shanghaitech`', metavar='')

    return parser.parse_args()


def main():

    # Parse command line arguments
    args = parse_arguments()

    # Lock seeds
    set_random_seed(30101990)

    # Run test
    if args.dataset == 'mnist':
        test_mnist()
    elif args.dataset == 'cifar10':
        test_cifar()
    elif args.dataset == 'ucsd-ped2':
        test_ucsdped2()
    elif args.dataset == 'shanghaitech':
        test_shanghaitech()
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')


# Entry point
if __name__ == '__main__':
    main()
