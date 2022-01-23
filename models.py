from torch import nn

class Discriminator(nn.Module):
    """
    Discriminator model is trained to classify whether the input its given is real or aritificially generated
    """

    def __init__(self, in_features):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
           nn.Linear(in_features, 128),
           nn.LeakyReLU(0.1),
           nn.Linear(128, 1),
           nn.Sigmoid()
        )
    
    def forward(self, X):
        return self.model(X)

class Generator(nn.Module):
    
    def __init__(self, noise_input_dim, img_output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_input_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_output_dim),
            nn.Tanh(),
        )
    
    def forward(self, X):
        return self.model(X)

        