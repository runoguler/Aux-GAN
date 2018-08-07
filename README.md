# Auxiliary Classifier GANs

This project is uploaded for the Visual Media course. It is about generating images similar to the ones in MNIST and CIFAR-10 dataset. The results are as follows:

![Generated MNIST](./results/generated_mnist(500epochs).png)
![Generated CIFAR-10](./results/generated_cifar10(dcnn750epochs).png)


## Used Libraries 

Libraries are installed using Anaconda(4.5.8)

* pytorch 0.4.0
* numpy 1.14.2
* torchvision 0.2.1
* matplotlib 2.2.2


## Important Notes about the code

### Running from an IDE (such as PyCharm)

- Inside *gan_mnist.py* and *gan_cifar10*, there are number of parameters that can be changed: (in main() function)
  -train_or_display: Display mode if 0, Train mode if otherwise
  -lr_g, lr_d: Learning rate for Generator and Discriminator
  -epochs: Number of epochs to train
  -resume: Train from scratch if 0, resume training if otherwise (Warning: Gives error if there is no saved model file to resume)
  
There are 2 different Aux-GAN Neural Network type defined in *aux_gan.py* and *aux_dcgan.py*. It needs to be manually switched inside the code to change the Neural Network type. In order to change it, switch the commented section in the import part of the files *gan_mnist.py* and *gan_cifar10* just like below:

```
from models.aux_gan import Generator, Discriminator
# from models.aux_dcgan import Generator, Discriminator
```

Depending on which dataset (MNIST or CIFAR-10) to use with the models, the following comments should be switched in the *aux_gan.py* and *aux_dcgan.py* model files:
```
# img_shape = (1, 28, 28) # MNIST dataset
img_shape = (3, 32, 32) # CIFAR-10 dataset
```

### Running from the Command Line

Command Line Parameters:
* --batch-size: Batch size for training (default: 64)
* --num-workers: Number of workers for cuda (default: 1)
* --lr-g: Learning rate for the Generator (default: 0.0002)
* --lr-d: Learning rate for the Discriminator (default: 0.0002)
* --epochs: Epoch number to train (default: 100)
* --train: 0 if display, train otherwise (default: 0(display))
* --resume: Continue training if non-zero (default: 0)

Examples:
```
python gan_mnist.py --epochs 100 --train 1
python gan_cifar10.py --lr-g 0.0002 --lr-d 0.0002 --epochs 250 --train 1 --resume 1
python gan_mnist.py --train 0
python gan_cifar10.py --train 0
```
