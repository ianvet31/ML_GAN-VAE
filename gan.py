import struct
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from hw5_utils import BASE_URL, download, GANDataset


class DNet(nn.Module):
    """This is discriminator network."""

    def __init__(self):
        super(DNet, self).__init__()
        
        # TODO: implement layers here'=
        self.conv2d_1 = nn.Conv2d(1, 2, (3, 3), 1, 1, bias=True)
        self.relu_1 = nn.ReLU()
        self.maxpool_1 = nn.MaxPool2d((2, 2), 2)
        self.conv2d_2 = nn.Conv2d(2, 4, (3, 3), 1, 1, bias=True)
        self.relu_2 = nn.ReLU()
        self.maxpool_2 = nn.MaxPool2d((2, 2), 2)
        self.conv2d_3 = nn.Conv2d(4, 8, (3, 3), 1, 0, bias=True)main
        self.relu_3 = nn.ReLU()
        self.linear = nn.Linear(200, 1, bias=True)

        self._weight_init()

    def _weight_init(self):
        # TODO: implement weight initialization here
        for layer in self.children():
            
            if hasattr(layer, 'weight'):
                nn.init.kaiming_uniform_(layer.weight.data)
            if hasattr(layer, 'bias'):
                nn.init.constant_(layer.bias.data, 0)

                

    def forward(self, x):
        # TODO: complete forward function

        f = self.conv2d_1(x)
        f = self.relu_1(f)
        f = self.maxpool_1(f)
        f = self.conv2d_2(f)
        f = self.relu_2(f)
        f = self.maxpool_2(f)
        f = self.conv2d_3(f)
        f = self.relu_3(f)
        f = torch.flatten(f, start_dim=1)
        f = self.linear(f)
        return f



class GNet(nn.Module):
    """This is generator network."""

    def __init__(self, zdim):
        """
        Parameters
        ----------
            zdim: dimension for latent variable.
        """
        super(GNet, self).__init__()

        # TODO: implement layers here
        self.linear_1 = nn.Linear(zdim, 1568, bias=True)
        self.leakyrelu_1 = nn.LeakyReLU(0.2)
        self.upsample_1 = nn.Upsample(scale_factor=2)
        self.conv2d_1 = nn.Conv2d(32, 16, (3, 3), 1, 1, bias=True)
        self.leakyrelu_2 = nn.LeakyReLU(0.2)
        self.upsample_2 = nn.Upsample(scale_factor=2)
        self.conv2d_2 = nn.Conv2d(16, 8, (3, 3), 1, 1, bias=True)
        self.leakyrelu_3 = nn.LeakyReLU(0.2)
        self.conv2d_3 = nn.Conv2d(8, 1, (3,3), 1, 1, bias=True)
        self.sigmoid_1 = nn.Sigmoid()

        self._weight_init()



    def _weight_init(self):
        # TODO: implement weight initialization here
        for layer in self.children():
            if hasattr(layer, 'weight'):
                nn.init.kaiming_uniform_(layer.weight.data)
            if hasattr(layer, 'bias'):
                nn.init.constant_(layer.bias.data, 0)



    def forward(self, z):
        """
        Parameters
        ----------
            z: latent variables used to generate images.
        """
        # TODO: complete forward function
        f = self.linear_1(z)
        g = self.leakyrelu_1(f)
        h = g.view(z.shape[0], 32, 7, 7)
        i = self.upsample_1(h)
        j = self.conv2d_1(i)
        k = self.leakyrelu_2(j)
        l = self.upsample_2(k)
        m = self.conv2d_2(l)
        p = self.leakyrelu_3(m)
        q = self.conv2d_3(p)
        r = self.sigmoid_1(q)
    

        return r


class GAN:
    def __init__(self, zdim=64):
        """
        Parameters
        ----------
            zdim: dimension for latent variable.
        """
        torch.manual_seed(2)
        self._dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._zdim = zdim
        self.disc = DNet().to(self._dev)
        self.gen = GNet(self._zdim).to(self._dev)

    def _get_loss_d(self, batch_size, batch_data, z):
        """This function computes loss for discriminator.

        Parameters
        ----------
            batch_size: #data per batch.
            batch_data: data from dataset.
            z: random latent variable.
        """
        # TODO: implement discriminator's loss function
        ones = torch.ones([batch_size, 1])
        zeros = torch.zeros([batch_size, 1])

        criterion = nn.BCEWithLogitsLoss(pos_weight=ones)
        

        # TODO: implement generator's loss function
        loss_1 = self._get_loss_g(batch_size, z, p=0)


        Dx = self.disc(batch_data)

        Dx_l = criterion(Dx, ones)
        Dxl2 = criterion(Dx, ones)
    


        loss = (Dx_l + loss_1)/2

        return loss
    
    def _get_loss_g(self, batch_size, z, p=1):
        """This function computes loss for generator.
        Compute -\sum_z\log{D(G(z))} instead of \sum_z\log{1-D(G(z))}
        
        Parameters
        ----------
            batch_size: #data per batch.
            z: random latent variable.
        """

        criterion_ = nn.BCEWithLogitsLoss(reduction='mean')

        if (p==1):  
            ones = torch.ones([batch_size, 1])
        else:
            ones = torch.zeros([batch_size, 1])
    
        # TODO: implement generator's loss function

        Gz = self.gen(z)
        DG_ = torch.tensor(self.disc(Gz))
        '''
        Gz = self.gen(z[0])
        Dg = self.disc(Gz)
        DG_ = torch.tensor(Dg)

        for i in range(1, batch_size):

            Gz = self.gen(z[i])
            Dg = self.disc(Gz)
            DG_ = torch.cat((DG_, Dg))
        '''
        loss = criterion_(DG_, ones)
    
        return loss


    def train(self, iter_d=1, iter_g=1, n_epochs=1, batch_size=256, lr=0.0002):

        # first download
        f_name = "train-images-idx3-ubyte.gz"
        download(BASE_URL + f_name, f_name)

        print("Processing dataset ...")
        train_data = GANDataset(
            f"./data/{f_name}",
            self._dev,
            transform=transforms.Compose([transforms.Normalize((0.0,), (255.0,))]),
        )
        print(f"... done. Total {len(train_data)} data entries.")

        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )

        dopt = optim.Adam(self.disc.parameters(), lr=lr, weight_decay=0.0)
        dopt.zero_grad()
        gopt = optim.Adam(self.gen.parameters(), lr=lr, weight_decay=0.0)
        gopt.zero_grad()

        for epoch in tqdm(range(n_epochs)):
            for batch_idx, data in tqdm(
                enumerate(train_loader), total=len(train_loader)
            ):

                z = 2 * torch.rand(data.size()[0], self._zdim, device=self._dev) - 1

                if batch_idx == 0 and epoch == 0:
                    plt.imshow(data[0, 0, :, :].detach().cpu().numpy())
                    plt.savefig("goal.pdf")

                if batch_idx == 0 and epoch % 10 == 0:
                    with torch.no_grad():
                        tmpimg = self.gen(z)[0:64, :, :, :].detach().cpu()
                    save_image(
                        tmpimg, "test_{0}.png".format(epoch), nrow=8, normalize=True
                    )

                dopt.zero_grad()
                for k in range(iter_d):
                    loss_d = self._get_loss_d(batch_size, data, z)
                    loss_d.backward()
                    dopt.step()
                    dopt.zero_grad()

                gopt.zero_grad()
                for k in range(iter_g):
                    loss_g = self._get_loss_g(batch_size, z)
                    loss_g.backward()
                    gopt.step()
                    gopt.zero_grad()

            print(f"E: {epoch}; DLoss: {loss_d.item()}; GLoss: {loss_g.item()}")


if __name__ == "__main__":
    gan = GAN()
    gan.train()
