from dataset import MahjongGBDataset
from torch.utils.data import DataLoader
from model import CNNModel, Encoder
import torch.nn.functional as F
import torch
import os

splitRatio = 0.9
batchSize = 2000
lr = 5e-5
epoch = 15
warmup = 2
logdir = 'model/Data_51(214_235checkpoint'
file = "data/Data_51(214_235"
vocab_size = 214

if __name__ == '__main__':
    device = 'cuda'
    # Load dataset
    validateDataset = MahjongGBDataset(file, splitRatio, 1, False)
    vloader = DataLoader(dataset = validateDataset, batch_size = batchSize, shuffle = False)
    
    # Load model
    model = Encoder(vocab_size=vocab_size).to(device)
    model.load_state_dict(torch.load(f"{logdir}/7.pkl", map_location = torch.device('cuda')))

    print('Run validation:')
    correct = 0
    for i, d in enumerate(vloader):
        input_dict = {'is_training': False, 'obs': {'observation': d[0].cuda(), 'action_mask': d[1].cuda()}}
        with torch.no_grad():
            logits = model(input_dict)
            pred = logits.argmax(dim = 1)
            correct += torch.eq(pred, d[2].cuda()).sum().item()
    acc = correct / len(validateDataset)
    print('Validate acc:', acc)