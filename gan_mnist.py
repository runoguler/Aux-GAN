import torch
import argparse
import numpy as np
from torchvision import datasets, transforms
from torchvision.utils import save_image

from torch.autograd import Variable

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

'''IMPORTANT: Change the following to switch between Network Models'''
from models.aux_gan import Generator, Discriminator
# from models.aux_dcgan import Generator, Discriminator


def display():
    gen_len = 100
    generator = Generator(gen_len)
    generator.load_state_dict(torch.load('./acgan_models/generator.pth', map_location='cpu'))
    generator.eval()

    x = []
    for i in range(100):
        x.append(i%10)
    labels = Variable(torch.LongTensor(np.array(x)))
    z = Variable(torch.FloatTensor(np.random.normal(0, 1, (100, gen_len))))

    fake = generator(z, labels)

    save_image(fake.detach().data, 'generated_mnist.png', 10, normalize=True)

    img = plt.imread('generated_mnist.png')
    plt.imshow(img)
    plt.show()


def train(args, train_loader, device, Tensor, LongTensor):
    gen_len = 100

    epochs = args.epochs

    validity_loss = torch.nn.BCELoss()
    result_loss = torch.nn.CrossEntropyLoss()

    generator = Generator(gen_len)
    discriminator = Discriminator()
    if args.resume:
        generator.load_state_dict(torch.load('./acgan_models/generator.pth'))
        discriminator.load_state_dict(torch.load('./acgan_models/discriminator.pth'))

    generator.train()
    discriminator.train()

    generator.to(device)
    discriminator.to(device)
    validity_loss.to(device)
    result_loss.to(device)

    optim_gen = torch.optim.Adam(generator.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    optim_dis = torch.optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for (images, labels) in train_loader:
            batch_len = images.size(0)

            ones = Variable(Tensor(batch_len, 1).fill_(1.0), requires_grad=False)
            zeros = Variable(Tensor(batch_len, 1).fill_(0.0), requires_grad=False)

            images, labels = Variable(images.to(device).type(Tensor)), Variable(labels.to(device).type(LongTensor))

            rd_labels = Variable(LongTensor(np.random.randint(0, 10, batch_len)))
            z = Variable(Tensor(np.random.normal(0, 1, (batch_len, gen_len))))

            optim_gen.zero_grad()
            fake = generator(z, rd_labels)
            pred_label, validity = discriminator(fake)
            loss_g = (validity_loss(validity, ones) + result_loss(pred_label, rd_labels)) / 2
            loss_g.backward()
            optim_gen.step()

            optim_dis.zero_grad()
            real_labels, real_validity = discriminator(images)
            fake_labels, fake_validity = discriminator(fake.detach())
            real_loss = (validity_loss(real_validity, ones) + result_loss(real_labels, labels)) / 2
            fake_loss = (validity_loss(fake_validity, zeros) + result_loss(fake_labels, labels)) / 2
            loss_d = (real_loss + fake_loss) / 2
            loss_d.backward()
            optim_dis.step()

        print("Epoch: ", epoch, ", Loss(Gen): ", loss_g.item(), ", Loss(Dis): ", loss_d.item())

    torch.save(generator.state_dict(), './acgan_models/generator.pth')
    torch.save(discriminator.state_dict(), './acgan_models/discriminator.pth')


def main():
    epochs = 100
    train_or_display = 0
    lr_g = 0.0002
    lr_d = 0.0002
    resume = 0

    parser = argparse.ArgumentParser(description="Parameters for Training GAN on MNIST dataset")
    parser.add_argument('--batch-size', type=int, default=64, help='batch size for training (default: 64)')
    parser.add_argument('--num-workers', type=int, default=1, help='number of workers for cuda (default: 1)')
    parser.add_argument('--lr-g', type=float, default=lr_g, help='learning rate for the generator network (default: 0.0002)')
    parser.add_argument('--lr-d', type=float, default=lr_d, help='learning rate for the discriminator network (default: 0.0002)')
    parser.add_argument('--epochs', type=int, default=epochs, help='epoch number to train (default: 100)')
    parser.add_argument('--train', type=int, default=train_or_display, help='train(1) or display(0) (default: display(0))')
    parser.add_argument('--resume', type=int, default=resume, help='continue training if 1 (default: 0)')
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    cuda_args = {'num_workers': args.num_workers, 'pin_memory': True} if use_cuda else {}

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    training_data = datasets.MNIST("../data/MNIST", train=True, transform=data_transform, download=True)

    data_loader = torch.utils.data.DataLoader(training_data, batch_size=args.batch_size, shuffle=True, **cuda_args)

    if use_cuda:
        Tensor = torch.cuda.FloatTensor
        LongTensor = torch.cuda.LongTensor
    else:
        Tensor = torch.FloatTensor
        LongTensor = torch.LongTensor


    if args.train:
        train(args, data_loader, device, Tensor, LongTensor)
    else:
        display()


if __name__ == '__main__':
    main()
