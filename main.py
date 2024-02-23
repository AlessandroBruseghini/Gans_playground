import torch
from torch import nn

import torchvision
from torchvision import datasets, transforms
from torchvision.utils import make_grid

from torch.utils.data import DataLoader

from tqdm import tqdm
import matplotlib.pyplot as plt


def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()



class Generator(nn.Module):
    '''
    Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        im_chan: the number of channels in the images, fitted for the dataset used, a scalar
              (MNIST is black-and-white, so 1 channel is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, z_dim=10, im_chan=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        
        self.G = nn.Sequential(
            self.make_G_block(z_dim, hidden_dim * 4),
            self.make_G_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            self.make_G_block(hidden_dim * 2, hidden_dim),
            self.make_G_block(hidden_dim, im_chan, kernel_size=4, final_layer=True),
        )

    def make_G_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a Gerator block of DCGAN, 
        corresponding to a transposed convolution, a batchnorm (except for in the last layer), and an activation.
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise 
                      (affects activation and batchnorm)
        '''

        #     Steps:
        #       1) Do a transposed convolution using the given parameters.
        #       2) Do a batchnorm, except for the last layer.
        #       3) Follow each batchnorm with a ReLU activation.
        #       4) If its the final layer, use a Tanh activation after the deconvolution.

        
        if not final_layer:
            return nn.Sequential(
                
                nn.ConvTranspose2d(in_channels=input_channels,
                                   out_channels=output_channels,
                                   kernel_size=kernel_size,
                                   stride=stride),
                nn.BatchNorm2d(num_features=output_channels),
                nn.ReLU()
                
            )
        else: # Final Layer
            return nn.Sequential(
                
                nn.ConvTranspose2d(in_channels=input_channels,
                                   out_channels=output_channels,
                                   kernel_size=kernel_size,
                                   stride=stride),
                nn.Tanh()
                
            )

    def unsqueeze_noise(self, noise):
        '''
        Function for completing a forward pass of the Gerator: Given a noise tensor, 
        returns a copy of that noise with width and height = 1 and channels = z_dim.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        return noise.view(len(noise), self.z_dim, 1, 1)

    def forward(self, noise):
        '''
        Function for completing a forward pass of the Gerator: Given a noise tensor, 
        returns Gerated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        x = self.unsqueeze_noise(noise)
        return self.G(x)

def get_noise(n_samples, z_dim, device):
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim)
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
        n_samples: the number of samples to Gerate, a scalar
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    '''
    return torch.randn(n_samples, z_dim, device=device)





class Discriminator(nn.Module):
    '''
    Discriminator Class
    Values:
        im_chan: the number of channels in the images, fitted for the dataset used, a scalar
              (MNIST is black-and-white, so 1 channel is your default)
    hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, im_chan=1, hidden_dim=16):
        super(Discriminator, self).__init__()
        self.D = nn.Sequential(
            self.make_D_block(im_chan, hidden_dim),
            self.make_D_block(hidden_dim, hidden_dim * 2),
            self.make_D_block(hidden_dim * 2, 1, final_layer=True),
        )

    def make_D_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a Driminator block of DCGAN, 
        corresponding to a convolution, a batchnorm (except for in the last layer), and an activation.
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise 
                      (affects activation and batchnorm)
        '''
        #     Steps:
        #       1) Add a convolutional layer using the given parameters.
        #       2) Do a batchnorm, except for the last layer.
        #       3) Follow each batchnorm with a LeakyReLU activation with slope 0.2.
        #       Note: Don't use an activation on the final layer
        
        
        if not final_layer:
            return nn.Sequential(
                
                nn.Conv2d(in_channels=input_channels,
                          out_channels=output_channels,
                          kernel_size=kernel_size,
                          stride=stride),
                nn.BatchNorm2d(num_features=output_channels),
                nn.LeakyReLU(negative_slope=0.2)
                
            )
        else: # Final Layer
            return nn.Sequential(
                
                nn.Conv2d(in_channels=input_channels,
                          out_channels=output_channels,
                          kernel_size=kernel_size,
                          stride=stride)
                
            )

    def forward(self, image):
        '''
        Function for completing a forward pass of the Driminator: Given an image tensor, 
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            image: a flattened image tensor with dimension (im_dim)
        '''
        D_pred = self.D(image)
        return D_pred.view(len(D_pred), -1)




criterion = nn.BCEWithLogitsLoss()
z_dim = 64
display_step = 500
batch_size = 128

lr = 0.0002


beta_1 = 0.5 
beta_2 = 0.999
device = 'cuda' if torch.cuda.is_available() else 'cpu'



transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataloader = DataLoader(
    dataset=datasets.MNIST('datasets', download=False, transform=transform),
    batch_size=batch_size,
    shuffle=True)



G= Generator(z_dim).to(device)
D = Discriminator().to(device)

G_opt = torch.optim.Adam(G.parameters(), lr=lr, betas=(beta_1, beta_2)) 
D_opt = torch.optim.Adam(D.parameters(), lr=lr, betas=(beta_1, beta_2))



def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

G = G.apply(weights_init)
D = D.apply(weights_init)



n_epochs = 7
current_step = 0
mean_Gerator_loss = 0
mean_Driminator_loss = 0


for epoch in range(n_epochs):
    print(f'Epoch: {epoch}\n----------------')
    
    for real, _ in tqdm(dataloader):
        current_batch_size = len(real)
        real = real.to(device)

        
        
        fake_noise = get_noise(current_batch_size, z_dim, device=device)
        fake = G(fake_noise)
        
        D_fake_pred = D(fake.detach())
        D_fake_loss = criterion(D_fake_pred, torch.zeros_like(D_fake_pred))
        
        D_real_pred = D(real)
        D_real_loss = criterion(D_real_pred, torch.ones_like(D_real_pred))
        
        D_loss = (D_fake_loss + D_real_loss) / 2
        
        
        mean_Driminator_loss += D_loss.item() / display_step
        
        D_opt.zero_grad()
        D_loss.backward(retain_graph=True)
        D_opt.step()

        

        fake_noise_2 = get_noise(current_batch_size, z_dim, device=device)
        fake_2 = G(fake_noise_2)

        D_fake_pred = D(fake_2)
        G_loss = criterion(D_fake_pred, torch.ones_like(D_fake_pred))
        
        
        mean_Gerator_loss += G_loss.item() / display_step

        G_opt.zero_grad()
        G_loss.backward()
        G_opt.step()



        
        if current_step % display_step == 0 and current_step > 0:
            print(f"Step {current_step}: Generator loss: {mean_Gerator_loss}, Discriminator loss: {mean_Driminator_loss}")
            show_tensor_images(fake)
            show_tensor_images(real)
            mean_Gerator_loss = 0
            mean_Driminator_loss = 0
        current_step += 1




