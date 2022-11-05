import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack as sparse_vstack

# Abstract evaluator class, each model should implement its own `process_batch` and `prepare_evaluation_dataframe`
class AbstractModelEvaluator:
    def __init__(self):
        self.reset()
        
    def evaluate(self, model, dataloader, device, cutoff=4):
        self.reset()
        with torch.no_grad():
            for batch in tqdm(iter(dataloader)):
                batch = batch[0].to(device)
                self.process_batch(model, batch, device)

        eval_df = self.prepare_evaluation_dataframe()
        
        return eval_df
    
    def reset(self):
        pass
    
    def process_batch(self, model, batch, device):
        raise NotImplementedError('Subclasses must implement process_batch')
        
    def prepare_evaluation_dataframe(self):
        raise NotImplementedError('Subclasses must implement prepare_evaluation_dataframe')
        
class AutoEncoderEvaluator(AbstractModelEvaluator):
    def evaluate(self, model, dataloader, device, cutoff=4):
        self.reset()
        with torch.no_grad():
            for batch in tqdm(iter(dataloader)):
                self.process_batch(model, batch, device)

        eval_df = self.prepare_evaluation_dataframe()
        
        return eval_df
    
    def reset(self):
        self.y = None
        self.y_pred = None
    
    def process_batch(self, model, batch, device):
        r, mask, indices = batch
        r_sparse = csr_matrix(r)
        if self.y is None:
            self.y = r_sparse
        else:
            self.y = sparse_vstack([self.y, r_sparse])
            
        r = r.to(device, dtype=torch.float32)
        mask = mask.to(device, dtype=torch.float32)
        pred_sparse = csr_matrix(((model(r))).detach().cpu().numpy())
        if self.y_pred is None:
            self.y_pred = pred_sparse
        else:
            self.y_pred = sparse_vstack([self.y_pred, pred_sparse])
    
    def prepare_evaluation_dataframe(self, clip=True):
#         spot_ids, gene_ids = self.y.nonzero()
#         print(self.y_pred[spot_ids, gene_ids].shape)
#         print(np.asarray(self.y_pred.todense()).flatten().shape)
#         return self.y
#         pred_rating = np.asarray(self.y_pred[spot_ids, gene_ids])[0]
        
        pred_count = np.asarray(self.y_pred.todense()).flatten()
        spot_ids, gene_ids = np.unravel_index(np.arange(pred_count.shape[0]), self.y_pred.shape)
        print(spot_ids, gene_ids)

        return pd.DataFrame({
            'spot_id': spot_ids,
            'gene_id': gene_ids,
            'count': np.asarray(self.y.todense()).flatten(),
            'pred_count': pred_count
        })
    
    
class PositionalMLPEvaluator(AbstractModelEvaluator):
    def evaluate(self, model, dataloader, device, cutoff=4):
        self.reset()
        with torch.no_grad():
            for batch in tqdm(iter(dataloader)):
                self.process_batch(model, batch, device)

        eval_df = self.prepare_evaluation_dataframe()
        
        return eval_df
    
    def reset(self):
        self.y = None
        self.y_pred = None
    
    def process_batch(self, model, batch, device):
        x, y, mask = batch
        y_sparse = csr_matrix(y)
        
        if self.y is None:
            self.y = y_sparse
        else:
            self.y = sparse_vstack([self.y, y_sparse])
            
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)
        mask = mask.to(device, dtype=torch.float32)
        pred_sparse = csr_matrix(((model(x))).detach().cpu().numpy())
        if self.y_pred is None:
            self.y_pred = pred_sparse
        else:
            self.y_pred = sparse_vstack([self.y_pred, pred_sparse])
    
    def prepare_evaluation_dataframe(self, clip=True):
        pred_count = np.asarray(self.y_pred.todense()).flatten()
        spot_ids, gene_ids = np.unravel_index(np.arange(pred_count.shape[0]), self.y_pred.shape)
        print(spot_ids, gene_ids)

        return pd.DataFrame({
            'spot_id': spot_ids,
            'gene_id': gene_ids,
            'count': np.asarray(self.y.todense()).flatten(),
            'pred_count': pred_count
        })

class ImageCNNEvaluator(AbstractModelEvaluator):
    def evaluate(self, model, dataloader, device, cutoff=4):
        self.reset()
        with torch.no_grad():
            for batch in tqdm(iter(dataloader)):
                self.process_batch(model, batch, device)

        eval_df = self.prepare_evaluation_dataframe()
        
        return eval_df
    
    def reset(self):
        self.y = None
        self.y_pred = None
    
    def process_batch(self, model, batch, device):
        x, y, mask = batch
        y_sparse = csr_matrix(y)
        
        if self.y is None:
            self.y = y_sparse
        else:
            self.y = sparse_vstack([self.y, y_sparse])
            
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)
        mask = mask.to(device, dtype=torch.float32)
        pred_sparse = csr_matrix(((model(x))).detach().cpu().numpy())
        if self.y_pred is None:
            self.y_pred = pred_sparse
        else:
            self.y_pred = sparse_vstack([self.y_pred, pred_sparse])
    
    def prepare_evaluation_dataframe(self, clip=True):
        pred_count = np.asarray(self.y_pred.todense()).flatten()
        spot_ids, gene_ids = np.unravel_index(np.arange(pred_count.shape[0]), self.y_pred.shape)
        print(spot_ids, gene_ids)

        return pd.DataFrame({
            'spot_id': spot_ids,
            'gene_id': gene_ids,
            'count': np.asarray(self.y.todense()).flatten(),
            'pred_count': pred_count
        })
    
    
class FinalModelEvaluator(AbstractModelEvaluator):
    def evaluate(self, model, dataloader, device, cutoff=4):
        self.reset()
        with torch.no_grad():
            for batch in tqdm(iter(dataloader)):
                self.process_batch(model, batch, device)

        eval_df = self.prepare_evaluation_dataframe()
        
        return eval_df
    
    def reset(self):
        self.y = None
        self.y_pred = None
    
    def process_batch(self, model, batch, device):
        r, mask, pos_info, tile_image, index = batch
        mask = mask.to(device, dtype=torch.float32)
        pos_info = pos_info.to(device, dtype=torch.float32)
        tile_image = tile_image.to(device, dtype=torch.float32)
        y_sparse = csr_matrix(r)
        
        r = r.to(device, dtype=torch.float32)
        
        if self.y is None:
            self.y = y_sparse
        else:
            self.y = sparse_vstack([self.y, y_sparse])
            
        pred_sparse = csr_matrix(((model(r, pos_info, tile_image))).detach().cpu().numpy())
        if self.y_pred is None:
            self.y_pred = pred_sparse
        else:
            self.y_pred = sparse_vstack([self.y_pred, pred_sparse])
    
    def prepare_evaluation_dataframe(self, clip=True):
        pred_count = np.asarray(self.y_pred.todense()).flatten()
        spot_ids, gene_ids = np.unravel_index(np.arange(pred_count.shape[0]), self.y_pred.shape)
        print(spot_ids, gene_ids)

        return pd.DataFrame({
            'spot_id': spot_ids,
            'gene_id': gene_ids,
            'count': np.asarray(self.y.todense()).flatten(),
            'pred_count': pred_count
        })
