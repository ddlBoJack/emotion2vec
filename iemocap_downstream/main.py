import os
import sys
from datetime import datetime

import hydra 
from omegaconf import DictConfig

import torch
from torch import nn, optim

from data import load_ssl_features, train_valid_test_iemocap_dataloader
from model import BaseModel
from misc import train_one_epoch, validate_and_test

import logging

def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        param = parameter.numel()
        print(f"{name}: {param}")
        total_params += param
    print(f"\nTotal number of parameters: {total_params}")


@hydra.main(config_path='config', config_name='default.yaml', version_base="1.2")
def train_iemocap(cfg: DictConfig):
    torch.manual_seed(cfg.common.seed)

    if cfg.dataset._name == 'IEMOCAP':
        label_dict={'ang': 0, 'hap': 1, 'neu': 2, 'sad': 3}
        fold_sizes = [1085, 1023, 1151, 1031, 1241] # Session1, 2, 3, 4, 5
        fold_list = [0, 1, 2, 3, 4]

    test_wa_avg, test_ua_avg, test_f1_avg = 0., 0., 0.
    
    log_name = 'emotion2vec_downstream.log'
    
    os.makedirs('outputs/' + cfg.model._name, exist_ok=True)
    log_path = os.path.join('outputs/', cfg.model._name, log_name)
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        # stream=sys.stdout,
        filename=log_path,
    )
    logger = logging.getLogger(cfg.dataset._name)

    for fold in fold_list:# [4, 3, 2, 1, 0] latter 20% first
        logger.info(f"------Now it's {fold+1}th fold------")

        torch.cuda.set_device(cfg.common.device)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.cuda.empty_cache()
        
        dataset = load_ssl_features(
            cfg.dataset.feat_path,
            label_dict
        )
        test_len = fold_sizes[fold] 
        test_idx_start = sum(fold_sizes[:fold])
        test_idx_end = test_idx_start + test_len 
        train_loader, val_loader, test_loader = train_valid_test_iemocap_dataloader(
            dataset,
            cfg.dataset.batch_size,
            test_idx_start,
            test_idx_end,
            eval_is_test=cfg.dataset.eval_is_test,
        )

        model = BaseModel(
            input_dim=768, 
            output_dim=len(label_dict),
        )

        model = model.to(device)

        # count_parameters(model)
        optimizer = optim.RMSprop(model.parameters(), lr=cfg.optimization.lr, momentum=0.9)
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=cfg.optimization.lr, max_lr=1e-3, step_size_up=10)
        criterion = nn.CrossEntropyLoss()

        best_val_wa = 0
        best_val_wa_epoch = 0
        # Training loop
        
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs(os.path.join(cfg.model.save_dir, timestamp_str), exist_ok=True)
        save_dir = os.path.join(cfg.model.save_dir, timestamp_str, "checkpoint.pt")
        
        for epoch in range(cfg.optimization.epoch):  # Adjust the number of epochs as per your requirement
            train_loss = train_one_epoch(model, optimizer, criterion, train_loader, device)
            scheduler.step()
            # Validation step
            val_wa, val_ua, val_f1 = validate_and_test(model, val_loader, device, num_classes=len(label_dict))

            if val_wa > best_val_wa:
                best_val_wa = val_wa
                best_val_wa_epoch = epoch
                torch.save(model, save_dir)

            # Print losses for every epoch
            logger.info(f"Epoch {epoch+1}, Training Loss: {train_loss/len(train_loader):.6f}, Validation WA: {val_wa:.2f}%; UA: {val_ua:.2f}%; F1: {val_f1:.2f}%")

        model = torch.load(save_dir).to(device)
        test_wa, test_ua, test_f1 = validate_and_test(model, test_loader, device, num_classes=len(label_dict))
        logger.info(f"\n\nThe {fold+1}th Fold at epoch {best_val_wa_epoch + 1}, test WA {test_wa}%; UA {test_ua}%; F1 {test_f1}%")
        
        test_wa_avg += test_wa
        test_ua_avg += test_ua
        test_f1_avg += test_f1

    logger.info(f"Average WA: {test_wa_avg/len(fold_list)}%; UA: {test_ua_avg/len(fold_list)}%; F1: {test_f1_avg/len(fold_list)}%")

if __name__ == '__main__':
    train_iemocap()
