from torchaudio import datasets
from torchvision import transforms
from models import Generator, Discriminator
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision.datasets as datasets
import torchvision
import torch

if __name__ == "__main__":
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device: ', device, '\n')

    # Hyperparameters
    lr = 3e-4
    noise_dim = 64 
    image_dim = 28 * 28 * 1 
    batch_size = 64
    epochs = 50

    discriminator = Discriminator(in_features=image_dim).to(device)
    generator = Generator(noise_input_dim=noise_dim, img_output_dim=image_dim).to(device)

    fixed_noise = torch.randn((batch_size, noise_dim)).to(device)

    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307), std=(0.3081))
    ])
    
    dataset = datasets.MNIST("./data", train=True, transform=transforms, download=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

    criterion = torch.nn.BCELoss()

    writer_fake = SummaryWriter(log_dir="runs/GAN_MNIST/fake")
    writer_real = SummaryWriter(log_dir="runs/GAN_MNIST/real")
    step = 0

    for epoch in range(epochs):
        for batch_idx, (real, _) in enumerate(loader):
            real = real.view(-1, 784).to(device)
            batch_size = real.shape[0]

            noise = torch.randn(batch_size, noise_dim).to(device)
            fake = generator(noise)
          
            # Train Discriminator
            disc_real = discriminator(real).view(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))

            disc_fake = discriminator(fake.detach()).view(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_disc =  (loss_disc_real+loss_disc_fake)/2
            
            discriminator.zero_grad()
            loss_disc.backward()
            optimizer_discriminator.step()
            
            # Train Generator
            discriminator_output = discriminator(fake).view(-1)
            loss_gen = criterion(discriminator_output, torch.ones_like(discriminator_output))
            generator.zero_grad()
            loss_gen.backward()
            optimizer_generator.step()
            
            if batch_idx == 0:
                print(f"Epoch [{epoch}/{epochs}] Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f}")

                with torch.no_grad():
                    fake = generator(fixed_noise).reshape(-1, 1, 28, 28)
                    real = real.reshape(-1, 1, 28, 28)

                    img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                    img_grid_real = torchvision.utils.make_grid(real, normalize=True)

                    writer_fake.add_image(
                        "MNIST Fake Images", img_grid_fake, global_step=step
                    )
                    writer_real.add_image(
                        "MNIST Real Images", img_grid_real, global_step=step
                    )

                    step+=1