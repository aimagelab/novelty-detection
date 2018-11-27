from datasets import MNIST
from datasets import CIFAR10
from datasets import UCSDPed2
from datasets import SHANGHAITECH
from models import LSAMNIST
from models import LSACIFAR10
from models import LSAUCSD
from models import LSAShanghaiTech
from result_helpers import OneClassResultHelper
from result_helpers import VideoAnomalyDetectionResultHelper
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


def test_ucsdped2():
    # type: () -> None
    """
    Performs video anomaly detection tests on UCSD Ped2.
    """
    set_random_seed(30101990)

    # Build dataset and model
    dataset = UCSDPed2(path='data/UCSD')
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
    set_random_seed(30101990)

    # Build dataset and model
    dataset = SHANGHAITECH(path='data/SHANGHAITECH')
    model = LSAShanghaiTech(input_shape=dataset.shape, code_length=64, cpd_channels=100).cuda().eval()

    # Set up result helper and perform test
    helper = VideoAnomalyDetectionResultHelper(dataset,
                                               model,
                                               checkpoint='checkpoints/shanghaitech.pkl',
                                               output_file='shanghaitech.txt')
    helper.test_video_anomaly_detection()


test_shanghaitech()
