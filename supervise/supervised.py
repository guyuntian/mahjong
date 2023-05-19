from dataset import MahjongGBDataset
from torch.utils.data import DataLoader
from model import CNNModel, Encoder
import torch.nn.functional as F
import torch
import os
from transformers import get_scheduler, set_seed

splitRatio = 0.9
batchSize = 256
lr = 5e-5
epoch = 15
warmup = 2
logdir = 'model/Data_51(214_235'
file = "data/Data_51(214_235"
vocab_size = 214

def set_optimizer_scheduler(model):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
            "lr": lr,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": lr,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    scheduler = get_scheduler(name='cosine', optimizer=optimizer, \
                num_warmup_steps=warmup, num_training_steps=epoch)
    return optimizer, scheduler

if __name__ == '__main__':
    set_seed(2023)
    os.makedirs(logdir + 'checkpoint', exist_ok=True)
    device = 'cuda'
    # Load dataset
    trainDataset = MahjongGBDataset(file, 0, splitRatio, True)
    validateDataset = MahjongGBDataset(file, splitRatio, 1, False)
    loader = DataLoader(dataset = trainDataset, batch_size = batchSize, shuffle = True)
    vloader = DataLoader(dataset = validateDataset, batch_size = batchSize, shuffle = False)
    
    # Load model
    model = Encoder(vocab_size=vocab_size).to(device)
    optimizer, scheduler = set_optimizer_scheduler(model)
    
    # Train and validate
    for e in range(epoch):
        print('Epoch', e)
        torch.save(model.state_dict(), logdir + 'checkpoint/%d.pkl' % e)
        for i, d in enumerate(loader):
            input_dict = {'is_training': True, 'obs': {'observation': d[0].to(device), 'action_mask': d[1].to(device)}}
            logits = model(input_dict)
            loss = F.cross_entropy(logits, d[2].long().cuda())
            if i % 500 == 0:
                print('Iteration %d/%d'%(i, len(trainDataset) // batchSize + 1), 'policy_loss', loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        print('Run validation:')
        correct = 0
        for i, d in enumerate(vloader):
            input_dict = {'is_training': False, 'obs': {'observation': d[0].cuda(), 'action_mask': d[1].cuda()}}
            with torch.no_grad():
                logits = model(input_dict)
                pred = logits.argmax(dim = 1)
                correct += torch.eq(pred, d[2].cuda()).sum().item()
        acc = correct / len(validateDataset)
        print('Epoch', e + 1, 'Validate acc:', acc)