from models import Generator, Discriminator

import torch

if __name__ == "__main__":
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device: ', device, '\n')

    # Hyperparameters
    lr = 1e-4
    noise_dim = 128
    image_dim = 32 * 32 * 1 
    batch_size = 32
    epochs = 50

    discriminator = Discriminator(in_features=image_dim).to(device)
    generator = Generator(noise_input_dim=noise_dim, img_output_dim=image_dim).to(device)

    fixed_noise = torch.randn((batch_size, noise_dim)).to(device)

    

   
