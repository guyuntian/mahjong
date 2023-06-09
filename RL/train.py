from replay_buffer import ReplayBuffer
from actor import Actor
from learner import Learner

if __name__ == '__main__':
    config = {
        'replay_buffer_size': 50000,
        'replay_buffer_episode': 400,
        'model_pool_size': 20,
        'model_pool_name': 'model-pool',
        'num_actors': 2,
        'episodes_per_actor': 2000,
        'gamma': 0.98,
        'lambda': 0.95,
        'min_sample': 1000,
        'batch_size': 1500,
        'epochs': 5,
        'clip': 0.2,
        'lr': 3e-5,
        'headlr': 3e-4,
        'value_coeff': 1,
        'entropy_coeff': 0.01,
        'device': 'cuda',
        'ckpt_save_interval': 100,
        'ckpt_save_path': 'finetune19_clip10/',
        'model': '19.pkl'
    }

    import os
    os.makedirs(config['ckpt_save_path'], exist_ok=True)

    replay_buffer = ReplayBuffer(config['replay_buffer_size'], config['replay_buffer_episode'])
    
    actors = []
    for i in range(config['num_actors']):
        config['name'] = 'Actor-%d' % i
        actor = Actor(config, replay_buffer)
        actors.append(actor)
    learner = Learner(config, replay_buffer)
    
    for actor in actors: actor.start()
    learner.start()
    
    for actor in actors: actor.join()
    learner.terminate()