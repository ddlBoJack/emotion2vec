import os
import sys
from pathlib import Path

import hydra 
from omegaconf import DictConfig

import torch
from torch import nn, optim

from data import load_ssl_features, train_valid_test_iemocap_dataloader
from model import BaseModel
from utils import train_one_epoch, validate_and_test

import logging

logger = logging.getLogger('IEMOCAP_Downstream')

def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        param = parameter.numel()
        print(f"{name}: {param}")
        total_params += param
    print(f"\nTotal number of parameters: {total_params}")

@hydra.main(config_path='config', config_name='default.yaml')
def train_iemocap(cfg: DictConfig):
    torch.manual_seed(cfg.common.seed)

    label_dict={'ang': 0, 'hap': 1, 'neu': 2, 'sad': 3}
    n_samples = [1085, 1023, 1151, 1031, 1241] # Session1, 2, 3, 4, 5
    idx_sessions = [0, 1, 2, 3, 4]

    test_wa_avg, test_ua_avg, test_f1_avg = 0., 0., 0.
    
    for fold in idx_sessions: # extract the $fold$th as test set
        logger.info(f"------Now it's {fold+1}th fold------")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.cuda.empty_cache()
        
        dataset = load_ssl_features(cfg.dataset.feat_path, label_dict)

        test_len = n_samples[fold] 
        test_idx_start = sum(n_samples[:fold])
        test_idx_end = test_idx_start + test_len 
        train_loader, val_loader, test_loader = train_valid_test_iemocap_dataloader(
            dataset,
            cfg.dataset.batch_size,
            test_idx_start,
            test_idx_end,
            eval_is_test=cfg.dataset.eval_is_test,
        )

        model = BaseModel(input_dim=768, output_dim=len(label_dict))
        model = model.to(device)

        # count_parameters(model)
        optimizer = optim.RMSprop(model.parameters(), lr=cfg.optimization.lr, momentum=0.9)
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=cfg.optimization.lr, max_lr=1e-3, step_size_up=10)
        criterion = nn.CrossEntropyLoss()

        best_val_wa = 0
        best_val_wa_epoch = 0
        # Training loop
        
        save_dir = os.path.join(str(Path.cwd()), f"model_{fold+1}.pth")
        for epoch in range(cfg.optimization.epoch):  # Adjust the number of epochs as per your requirement
            train_loss = train_one_epoch(model, optimizer, criterion, train_loader, device)
            scheduler.step()
            # Validation step
            val_wa, val_ua, val_f1 = validate_and_test(model, val_loader, device, num_classes=len(label_dict))

            if val_wa > best_val_wa:
                best_val_wa = val_wa
                best_val_wa_epoch = epoch
                torch.save(model.state_dict(), save_dir)

            # Print losses for every epoch
            logger.info(f"Epoch {epoch+1}, Training Loss: {train_loss/len(train_loader):.6f}, Validation WA: {val_wa:.2f}%; UA: {val_ua:.2f}%; F1: {val_f1:.2f}%")

        ckpt = torch.load(save_dir)
        model.load_state_dict(ckpt, strict=True)
        test_wa, test_ua, test_f1 = validate_and_test(model, test_loader, device, num_classes=len(label_dict))
        logger.info(f"The {fold+1}th Fold at epoch {best_val_wa_epoch + 1}, test WA {test_wa}%; UA {test_ua}%; F1 {test_f1}%")
        
        test_wa_avg += test_wa
        test_ua_avg += test_ua
        test_f1_avg += test_f1

    logger.info(f"Average WA: {test_wa_avg/len(idx_sessions)}%; UA: {test_ua_avg/len(idx_sessions)}%; F1: {test_f1_avg/len(idx_sessions)}%")

if __name__ == '__main__':
    train_iemocap()
