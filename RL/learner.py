from multiprocessing import Process
import time
import numpy as np
import torch
from torch.nn import functional as F

from replay_buffer import ReplayBuffer
from model_pool import ModelPoolServer
from model import Encoder, Head

class Learner(Process):
    
    def __init__(self, config, replay_buffer):
        super(Learner, self).__init__()
        self.replay_buffer = replay_buffer
        self.config = config
        def set_optimizer(model, head):
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": 0.01,
                    "lr": config['lr'],
                },
                {
                    "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                    "lr": config['lr'],
                },
                {
                    "params": [p for n, p in head.named_parameters()],
                    "weight_decay": 0.01,
                    "lr": config['headlr'],
                },
            ]
            return torch.optim.AdamW(optimizer_grouped_parameters)
        self.set = set_optimizer
    
    def run(self):
        # create model pool
        model_pool = ModelPoolServer(self.config['model_pool_size'], self.config['model_pool_name'])
        
        # initialize model params
        device = torch.device(self.config['device'])
        model = Encoder()
        head = Head()
        model.load_state_dict(torch.load(self.config["model"]))
        
        # send to model pool
        model_pool.push(model.state_dict(), head.state_dict()) # push cpu-only tensor to model_pool
        model = model.to(device)
        head = head.to(device)
        
        # training
        optimizer = self.set(model, head)
        
        # wait for initial samples
        while self.replay_buffer.size() < self.config['min_sample']:
            time.sleep(0.1)
        
        cur_time = time.time()
        iterations = 0
        while True:
            # sample batch
            batch = self.replay_buffer.sample(self.config['batch_size'])
            obs = torch.tensor(batch['state']['observation']).to(device)
            mask = torch.tensor(batch['state']['action_mask']).to(device)
            states = {
                'observation': obs,
                'action_mask': mask
            }
            actions = torch.tensor(batch['action']).unsqueeze(-1).to(device)
            advs = torch.tensor(batch['adv']).to(device)
            targets = torch.tensor(batch['target']).to(device)
            
            # print('Iteration %d, replay buffer in %d out %d' % (iterations, self.replay_buffer.stats['sample_in'], self.replay_buffer.stats['sample_out']))
            
            # calculate PPO loss
            model.train(True) # Batch Norm training mode
            old_logits, _ = model(states, head)
            old_probs = F.softmax(old_logits, dim = 1).gather(1, actions)
            old_log_probs = torch.log(old_probs).detach()
            for _ in range(self.config['epochs']):
                logits, values = model(states, head)
                action_dist = torch.distributions.Categorical(logits = logits)
                probs = F.softmax(logits, dim = 1).gather(1, actions)
                log_probs = torch.log(probs)
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advs
                surr2 = torch.clamp(ratio, 1 - self.config['clip'], 1 + self.config['clip']) * advs
                policy_loss = -torch.mean(torch.min(surr1, surr2))
                value_loss = torch.mean(F.mse_loss(values.squeeze(-1), targets))
                entropy_loss = -torch.mean(action_dist.entropy())
                loss = policy_loss + self.config['value_coeff'] * value_loss + self.config['entropy_coeff'] * entropy_loss
                optimizer.zero_grad()
                loss.backward()
                # for name, param in model.named_parameters():
                #     if 'weight' in name:
                #         print(name)
                #         print(param.data.cpu().numpy().shape)
                #         print('gradient is \t', param.grad, '\trequires grad: ', param.requires_grad)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                torch.nn.utils.clip_grad_norm_(head.parameters(), 10)
                optimizer.step()

            with open("loss.txt", 'a') as f:
                print("iter:", iterations, "loss:", loss.item(), file=f)
            # push new model
            model = model.to('cpu')
            head = head.to('cpu')
            model_pool.push(model.state_dict(), head.state_dict()) # push cpu-only tensor to model_pool
            model = model.to(device)
            head = head.to(device)
            
            # save checkpoints
            t = time.time()
            if t - cur_time > self.config['ckpt_save_interval']:
                path_model = self.config['ckpt_save_path'] + 'model_%d.pkl' % iterations
                path_head = self.config['ckpt_save_path'] + 'head_%d.pkl' % iterations
                torch.save(model.state_dict(), path_model)
                torch.save(head.state_dict(), path_head)
                cur_time = t
            iterations += 1