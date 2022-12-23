import argparse, os
import torch
import numpy as np
import itertools
import datetime
import random
import yaml
from easydict import EasyDict
import time

import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

from tqdm import tqdm

from src.enviroment import DashCamEnv
from RLlib.MARC_3.marc_3 import MARC_3
from RLlib.MARC_3.replay_buffer import ReplayMemory, ReplayMemoryGPU

def parse_configs():
    parser = argparse.ArgumentParser(description='PyTorch MARC_3 implementation considering autoencoder structure')
    # For training and testing
    parser.add_argument('--config', default="cfgs/marc_3.yml",
                        help='Configuration file for MARC_3 algorithm.')
    parser.add_argument('--phase', default='test', choices=['train', 'test'],
                        help='Training or testing phase.')
    parser.add_argument('--gpu_id', type=int, default=0, metavar='N',
                        help='The ID number of GPU. Default: 0')
    parser.add_argument('--num_workers', type=int, default=4, metavar='N',
                        help='The number of workers to load dataset. Default: 4')
    parser.add_argument('--baseline', default='none', choices=['random', 'all_pos', 'all_neg', 'none'],
                        help='setup baseline results for testing comparison')
    parser.add_argument('--seed', type=int, default=123, metavar='N',
                        help='random seed (default: 123)')
    parser.add_argument('--num_epoch', type=int, default=50, metavar='N',
                        help='number of epoches (default: 50)')
    parser.add_argument('--snapshot_interval', type=int, default=1, metavar='N',
                        help='The epoch interval of model snapshot (default: 5)')
    parser.add_argument('--test_epoch', type=int, default=-1, 
                        help='The snapshot id of trained model for testing.')
    parser.add_argument('--output', default='./output/MARC_3',
                        help='Directory of the output. ')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = EasyDict(yaml.safe_load(f))
    cfg.update(vars(args))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    cfg.update(device=device)

    cfg.MARC_3.image_shape = cfg.ENV.image_shape
    cfg.MARC_3.input_shape = cfg.ENV.input_shape

    return cfg


def set_deterministic(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

def test():
    env = DashCamEnv(cfg.ENV, device=cfg.device)
    env.set_model(pretrained=True, weight_file=cfg.ENV.env_model)
    cfg.ENV.output_shape = env.output_shape

    # AgentENV
    agent = MARC_3(cfg.MARC_3, device=cfg.device)

    # load agent models (by default: the last epoch)
    ckpt_dir = os.path.join(cfg.output, 'checkpoints')
    agent.load_models(ckpt_dir, cfg)

    # start to test 
    agent.set_status('eval')

    rnn_state_1 = (torch.zeros((cfg.ENV.batch_size, cfg.MARC_3.hidden_size), dtype=torch.float32).to(cfg.device),
                    torch.zeros((cfg.ENV.batch_size, cfg.MARC_3.hidden_size), dtype=torch.float32).to(cfg.device))
    rnn_state_2 = (torch.zeros((cfg.ENV.batch_size, cfg.MARC_3.hidden_size), dtype=torch.float32).to(cfg.device),
                    torch.zeros((cfg.ENV.batch_size, cfg.MARC_3.hidden_size), dtype=torch.float32).to(cfg.device))
    rnn_state_3 = (torch.zeros((cfg.ENV.batch_size, cfg.MARC_3.hidden_size), dtype=torch.float32).to(cfg.device),
                    torch.zeros((cfg.ENV.batch_size, cfg.MARC_3.hidden_size), dtype=torch.float32).to(cfg.device))
        
    
    image = torch.rand(1,1,3,480,640).cuda()        ##### image input
    state = env.observe(image, fixation=None)

    # init vars before each episode
    actions, rnn_state_1, rnn_state_2, rnn_state_3 = agent.select_action(state, rnn_state_1, rnn_state_2, rnn_state_3, evaluate=True)
    pred_score = 0.5*(actions[0][0]+1)
    
    print("#####")
    print(pred_score)



if __name__ == "__main__":
    
    # parse input arguments
    cfg = parse_configs()

    # fix random seed 
    set_deterministic(cfg.seed)

    if cfg.phase == 'train':
        # train()
        raise NotImplementedError
    elif cfg.phase == 'test':
        test()
    else:
        raise NotImplementedError