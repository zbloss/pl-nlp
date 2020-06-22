import logging
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


def main():
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    train = MNIST('./data/', train=True, download=True, transform=transforms.ToTensor())
    logger.info('Train data saved')

    test = MNIST('./data/', train=False, download=True, transform=transforms.ToTensor())
    logger.info('False data saved')

    val = MNIST('./data/', train=True, download=True, transform=transforms.ToTensor())
    logger.info('Validation data saved')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    main()