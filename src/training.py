import torch
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, StepLR
from scipy.sparse import csr_matrix, vstack as sparse_vstack
import torch.optim.lbfgs
from src.evaluation import AbstractModelEvaluator


class AbstractModelTrainer:
    def __init__(self, train_dataloader, validation_dataloader,
                 model, device, optimizer_type=torch.optim.Adam, optimizer_kwargs=None,
                 lr=1e-3, n_epochs=2000, verbose=True, gradient_clipping=False,
                 patience=0, eps=1e-4, **kwargs):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.model = model
        self.device = device
        self.optimizer_type = optimizer_type
        self.lr = lr
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer = optimizer_type(model.parameters(), lr=lr, **optimizer_kwargs)
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.patience = patience
        self.eps = eps
        self.lr_scheduler = None
        self.gradient_clipping = gradient_clipping

    def train(self):
        train_hist = []
        validation_hist = []
        train_rmse_hist = []
        validation_rmse_hist = []
        best_validation_mse = np.inf
        remaining_patience = self.patience
        for epoch in range(1, self.n_epochs + 1):
            self.model.train()
            epoch_train_losses = []
            epoch_train_mses = []
            pbar = tqdm(iter(self.train_dataloader),
                        desc=f'Train epoch {epoch}/{self.n_epochs}',
                        disable=not self.verbose)

            # Training epoch
            for batch in pbar:
                loss, mse = self.calc_loss(batch)
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                if self.gradient_clipping:
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=5)

                self.optimizer.step()
                epoch_train_losses.append(loss.item())
                epoch_train_mses.append(mse.item())
                pbar.set_postfix({'Loss': np.mean(epoch_train_losses), 
                                  'rmse':str(round(np.sqrt(np.nanmean(epoch_train_mses)), 5))})

            pbar.close()

            train_hist.append(np.mean(epoch_train_losses))
            train_rmse_hist.append(np.sqrt(np.nanmean(epoch_train_mses)))
              
            if self.validation_dataloader:
                epoch_validation_losses = []
                epoch_validation_mses = []
                pbar = tqdm(iter(self.validation_dataloader),
                            desc=f'Validation', disable=not self.verbose)

                # Evaluating epoch
                self.model.eval()
                with torch.no_grad():
                    for batch in pbar:
                        val_loss, val_mse = self.calc_loss(batch)
                        epoch_validation_losses.append(val_loss.item())
                        epoch_validation_mses.append(val_mse.item())
                        pbar.set_postfix({'Loss': np.mean(epoch_validation_losses), 
                                          'rmse':str(round(np.sqrt(np.nanmean(epoch_validation_mses)), 5))})

                pbar.close()

                curr_validation_loss = np.nanmean(epoch_validation_losses)
                validation_hist.append(curr_validation_loss)

                curr_validation_mse = np.nanmean(epoch_validation_mses)
                validation_rmse_hist.append(np.sqrt(curr_validation_mse))

                # Early stopping
                if curr_validation_mse <= best_validation_mse - self.eps:
                    best_validation_mse = curr_validation_mse
                    remaining_patience = self.patience
                else:
                    remaining_patience -= 1
                    if remaining_patience == 0:
                        break

            if self.lr_scheduler:
                self.lr_scheduler.step()

        return train_hist, validation_hist, train_rmse_hist, validation_rmse_hist
    
    def calc_loss(self, batch):
        raise NotImplementedError('Subclasses must implement calc_loss')
    

class AutoEncoderTrainer(AbstractModelTrainer):
    def __init__(self, *args, spot_distance_matrix = None, reg_lambda=1e-2, spatial_reg=1e-3, K=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_lambda = reg_lambda
        self.spatial_reg_factor = spatial_reg
        self.K = K
        self.spot_distance_matrix = spot_distance_matrix
#         self.lr_scheduler = StepLR(self.optimizer, step_size=50, gamma=0.96)
        self.lr_scheduler = None

    def calc_loss(self, batch):
        r, mask, spot_indices = batch
        r = r.to(self.device, dtype=torch.float32)
        mask = mask.to(self.device, dtype=torch.float32)
#         mask = torch.tensor(1)
        pred = self.model(r)

#         recommendation_loss = torch.sum(((r - pred)) ** 2)
        recommendation_loss = torch.sum(((r - pred) * mask) ** 2)
        
        regularization_loss = 0

        # Regularization loss
        for param_name, param in self.model.named_parameters():

            # As seen in the code of the original implementation, the biases aren't part of the regularization loss
            if not param_name.endswith('bias'):
                regularization_loss += torch.norm(param) ** 2
        
        loss = (recommendation_loss + self.reg_lambda / 2 * regularization_loss)
        spatial_reg_loss = torch.tensor(0)
        if (self.spot_distance_matrix is not None and self.spatial_reg_factor):
            spatial_reg_loss = (self.spatial_reg_factor * torch.sum(((pred[:, :] - pred[torch.argsort(self.spot_distance_matrix[:, :])[:, :self.K], :].mean(axis=1)[0]) * mask) ** 2)) 
#             spatial_reg_loss = self.spatial_reg_factor * torch.sum(((pred[:, :] - pred[torch.argsort(self.spot_distance_matrix[:, :])[:, 1:self.K + 1], :].mean(axis=1)[0])) ** 2)
            
            loss += spatial_reg_loss
        mse = recommendation_loss / torch.sum(mask)
        
        return loss, mse
    

class SpotPositionTrainer(AbstractModelTrainer):
    def __init__(self, *args, reg_lambda=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr_scheduler = None
        self.reg_lambda = reg_lambda

    def calc_loss(self, batch):
        x, y, mask = batch
        x = x.to(self.device, dtype=torch.float32)
        y = y.to(self.device, dtype=torch.float32)
        mask = mask.to(self.device, dtype=torch.float32)
        y_pred = self.model(x)
        loss = torch.sum(((y - y_pred) * mask) ** 2)
        
        regularization_loss = 0 
        
        # Regularization loss
        for param_name, param in self.model.named_parameters():
            # As seen in the code of the original implementation, the biases aren't part of the regularization loss
            if not param_name.endswith('bias'):
                regularization_loss += torch.norm(param) ** 2
                
        loss += regularization_loss * self.reg_lambda
        mse = loss / mask.sum()
        
        return loss, mse


import torchvision

class SpotImageTrainer(AbstractModelTrainer):
    def __init__(self, *args, reg_lambda=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr_scheduler = None
        self.reg_lambda = reg_lambda
        self.optimizer = self.optimizer_type(self.model.fc.parameters(), lr=self.lr, **self.optimizer_kwargs)

    def calc_loss(self, batch):
        x, y, mask = batch
        x = x.to(self.device, dtype=torch.float32)
        y = y.to(self.device, dtype=torch.float32)
        mask = mask.to(self.device, dtype=torch.float32)
        y_pred = self.model(x)
        loss = torch.sum(((y - y_pred) * mask) ** 2)
        
        regularization_loss = 0 
        
        # Regularization loss
        for param_name, param in self.model.named_parameters():
            # As seen in the code of the original implementation, the biases aren't part of the regularization loss
            if not param_name.endswith('bias'):
                regularization_loss += torch.norm(param) ** 2
                
        loss += regularization_loss * self.reg_lambda
        mse = loss / mask.sum()
        
        return loss, mse
    
    
class FinalModelTrainer(AbstractModelTrainer):
    def __init__(self, train_dataloader, validation_dataloader,
                 model, device, optimizer_type=torch.optim.Adam, optimizer_kwargs=None,
                 lr=1e-3, n_epochs=2000, verbose=True, gradient_clipping=False,
                 patience=0, eps=1e-4, reg_lambda=0, **kwargs):
        
        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.model = model
        self.device = device
        self.optimizer = optimizer_type(list(model.auto_net.parameters()) + 
                                        list(model.position_net.parameters()) + 
                                        list(model.image_net.fc.parameters()) +
                                        list(model.decoder.parameters()), lr=lr, **optimizer_kwargs)
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.patience = patience
        self.eps = eps
        self.lr_scheduler = None
        self.reg_lambda = reg_lambda
        self.gradient_clipping = gradient_clipping
        
    def calc_loss(self, batch):
        r, mask, pos_info, tile_image, index = batch
        r = r.to(self.device, dtype=torch.float32)
        mask = mask.to(self.device, dtype=torch.float32)
        pos_info = pos_info.to(self.device, dtype=torch.float32)
        tile_image = tile_image.to(self.device, dtype=torch.float32)
        y_pred = self.model(r, pos_info, tile_image)
        loss = torch.sum(((r - y_pred) * mask) ** 2)
        
        regularization_loss = 0 
        
        # Regularization loss
#         for param_name, param in self.model.decoder.named_parameters():
        for param in (list(final_model.auto_net.parameters())):
#             As seen in the code of the original implementation, the biases aren't part of the regularization loss
#             if not param_name.endswith('bias'):
            regularization_loss += torch.norm(param) ** 2
                
        loss += regularization_loss * self.reg_lambda
        mse = loss / mask.sum()
        
        return loss, mse