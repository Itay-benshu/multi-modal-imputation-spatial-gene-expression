import torch
from torch import nn

class AutoEncoder(nn.Module):
    def __init__(self, n, hidden_size=512, init_std=0.03, binarize_after_first=False,
                 **kwargs):
        super().__init__()
        self.first_linear = nn.Linear(n, hidden_size)

        # Customizing the weight initialization std, and setting initial biases to zero on both layers
        torch.nn.init.trunc_normal_(self.first_linear.weight, mean=0.0, std=init_std)
        torch.nn.init.zeros_(self.first_linear.bias)
        self.first_activation = nn.ReLU()
        self.second_linear = nn.Linear(hidden_size, n)
        torch.nn.init.trunc_normal_(self.second_linear.weight, mean=0.0, std=init_std)
        torch.nn.init.zeros_(self.second_linear.bias)
        self.second_activation = nn.ReLU()

    def forward(self, x):
        return self.second_activation(
            self.second_linear(self.first_activation(self.first_linear(x)))
        )
    
class SpotPositionMLP(nn.Module):
    def __init__(self, x_size, n_genes):
        super().__init__()
        self.layers = nn.Sequential(
          nn.Flatten(),
          nn.Linear(x_size, 32),
          nn.ReLU(),
          nn.Linear(32, 64),
          nn.ReLU(),
          nn.Linear(64, 256),
          nn.ReLU(),
          nn.Linear(256, 1024),
          nn.ReLU(),
          nn.Linear(1024, n_genes),
        )


    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)

class SpotImageCNN(nn.Module):
    def __init__(self, n_genes):
        super().__init__()
        self.layers = nn.Sequential(
          nn.Conv2d(3, 96, 11, stride=4),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=3, stride=2),
          nn.Conv2d(96, 256, kernel_size=5, padding=2),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=3, stride=2),
          nn.Conv2d(256, 384, kernel_size=3, padding=1),
          nn.ReLU(),
          nn.Conv2d(384, 384, kernel_size=3, padding=1),
          nn.ReLU(),
          nn.Conv2d(384, 256, kernel_size=3, padding=1),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=3, stride=2),
          nn.Flatten(),
          nn.Linear(256 * 6 * 6, 4096),
          nn.ReLU(),
          nn.Linear(4096, 1024),
          nn.ReLU(),
          nn.Linear(1024, 4096),
          nn.ReLU(),
          nn.Linear(4096, n_genes)
        )

    def forward(self, x):
        '''Forward pass'''
        res = self.layers(x)
#         print(res.shape)
        return res


def load_model_weights(model, weights):
    model_dict = model.state_dict()
    weights = {k: v for k, v in weights.items() if k in model_dict}
    if weights == {}:
        print('No weight could be loaded..')
    model_dict.update(weights)
    model.load_state_dict(model_dict)

    return model

def create_final_histology_resnet_model(hidden_size):
    MODEL_PATH = 'pytorchnative_tenpercent_resnet18.ckpt'
    model = torchvision.models.__dict__['resnet18'](pretrained=False)
    state = torch.load(MODEL_PATH, map_location='cuda:0')
    state_dict = state['state_dict']
    for key in list(state_dict.keys()):
        state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)

    model = load_model_weights(model, state_dict)
    model.fc = torch.nn.Sequential(
                    torch.nn.Linear(model.fc.in_features, 1024),
                    torch.nn.ReLU(),
                    torch.nn.Linear(1024, hidden_size),
                    torch.nn.ReLU()

    )
    model = model.cuda()
    images = torch.rand((10, 3, 224, 224), device='cuda')
    
    # Just to verify it works
    out = model(images)
    
    return model

class FinalModel(nn.Module):
    def __init__(self, n_genes, spot_info_size, hidden_size=1024, input_noise_factor=0.1, init_std=0.03, binarize_after_first=False,
                 lambda_auto=1, lambda_pos=1, lambda_image=1,
                 **kwargs):
        super().__init__()
        self.lambda_auto = lambda_auto
        self.lambda_pos = lambda_pos
        self.lambda_image = lambda_image
        self.input_noise_factor = input_noise_factor
        
        self.auto_net = torch.nn.Sequential(
            nn.Linear(n_genes, hidden_size),
            nn.ReLU(),
        )
        
        torch.nn.init.trunc_normal_(self.auto_net[0].weight, mean=0.0, std=init_std)
        torch.nn.init.zeros_(self.auto_net[0].bias)
        
        self.position_net = torch.nn.Sequential(
              nn.Flatten(),
              nn.Linear(spot_info_size, 32),
              nn.ReLU(),
              nn.Linear(32, 64),
              nn.ReLU(),
              nn.Linear(64, 256),
              nn.ReLU(),
              nn.Linear(256, 1024),
              nn.ReLU(),
              nn.Linear(1024, hidden_size),
              nn.ReLU()
        )
        
        self.image_net = create_final_histology_resnet_model(hidden_size)
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, n_genes),
            nn.ReLU()
        )
        
    def forward(self, gene_info, spot_position_info, spot_image):
        # Not adding the noise if we are evaluating
#         noise = 1 if self.training else 0
#         if self.training:
#             gene_info = torch.nn.functional.dropout(gene_info, p=0.25)
            
        noise = 1
#         auto_encoder_input = gene_info
        auto_encoder_input = gene_info + torch.clip((torch.normal(mean=torch.ones_like(gene_info) * gene_info.mean(), std=gene_info.std()) * self.input_noise_factor * float(noise)),
                           min=0)
#          = torch.clip(gene_info + (torch.randn_like(gene_info) * self.input_noise_factor * float(noise)), min=0)
        
        # Manual (non-zero) dropout
        
        
        auto_res = self.auto_net(auto_encoder_input)
        position_res = self.position_net(spot_position_info)
        image_res = self.image_net(spot_image)
#         return self.decoder(torch.cat((auto_res, position_res, image_res), 1))
        encoded = (auto_res * self.lambda_auto +
                   position_res * self.lambda_pos +
                   image_res * self.lambda_image)
        
#         return encoded
        
#         noisy_encoded = torch.clip(encoded + (torch.normal(mean=torch.ones_like(encoded) * encoded.mean(), std=encoded.std()) * self.input_noise_factor * float(noise)),
#                                    min=0)
#         noisy_encoded = torch.ones_like(encoded)
        return self.decoder(encoded)