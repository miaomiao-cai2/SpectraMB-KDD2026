
import argparse
import os
import random
import time
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import torch
from loguru import logger

from data_set import DataSet
from model import MBGCN

from trainer import Trainer 


seed = 2026
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False  # True can improve train speed
    torch.backends.cudnn.deterministic = True  # Guarantee that the convolution algorithm returned each time will be deterministic
torch.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Set args', add_help=False)

    parser.add_argument('--embedding_size', type=int, default=64, help='')
    parser.add_argument('--layers', type=int, default=3)

    parser.add_argument('--data_name', type=str, default='taobao', help='')
    parser.add_argument('--behaviors', help='', action='append')
    parser.add_argument('--loss_type', type=str, default='bpr', help='')
    parser.add_argument('--if_load_model', type=bool, default=False, help='')
    parser.add_argument('--topk', type=list, default=[10, 20], help='')
    parser.add_argument('--metrics', type=list, default=['hit', 'ndcg'], help='')

    parser.add_argument('--lr', type=float, default=0.001, help='')
    parser.add_argument('--decay', type=float, default=0.001, help='')

    parser.add_argument('--batch_size', type=int, default=1024, help='')
    parser.add_argument('--test_batch_size', type=int, default=1024, help='')
    parser.add_argument('--min_epoch', type=int, default=5, help='')
    parser.add_argument('--epochs', type=int, default=200, help='')
    parser.add_argument('--model_path', type=str, default='./check_point', help='')
    parser.add_argument('--check_point', type=str, default='a_jdata_base.pth', help='')
    parser.add_argument('--model_name', type=str, default='', help='')
    parser.add_argument('--device', type=str, default='cuda:6', help='')
    # Early stopping control
    parser.add_argument('--early_stop_patience', type=int, default=10, help='Early stopping patience in epochs.')
    parser.add_argument('--no_early_stop', action='store_true', help='Disable early stopping to run fixed epochs.')
    
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    
    parser.add_argument('--reg_coeff', type=float, default=0.001)
    parser.add_argument('--main_coeff', type=float, default=1)
    
    parser.add_argument('--processed_global_coeff', type=float, default=0)
    parser.add_argument('--content_coeff', type=float, default=1.25)
    
    parser.add_argument('--mean_agg_coeff', type=float, default=1)
    

    args = parser.parse_args()
    _default_reg = parser.get_default('reg_coeff')
    _default_lambda_cl = parser.get_default('lambda_cl')
    if args.data_name == 'tmall':
        args.data_path = './data/Tmall'
        args.behaviors = ['click', 'collect', 'cart', 'buy']
    elif args.data_name == 'jdata':
        args.data_path = './data/jdata'
        args.behaviors = ['view', 'collect', 'cart', 'buy']
    elif args.data_name == 'jdata_disview':
        args.data_path = './data/jdata_disview'
        args.behaviors = ['collect', 'cart', 'buy']
    elif args.data_name == 'jdata_discart':
        args.data_path = './data/jdata_discart'
        args.behaviors = ['view', 'collect', 'buy']
    elif args.data_name == 'yelp':
        args.data_path = './data/Yelp'
        args.behaviors = ['tip', 'neutral', 'neg', 'pos']
    elif args.data_name == 'ml':
        args.data_path = './data/ML10M'
        args.behaviors = ['neutral', 'neg', 'pos']
    elif args.data_name == 'ml_disneg':
        args.data_path = './data/ML10M_disneg'
        args.behaviors = ['neutral', 'pos']
    elif args.data_name == 'ml_disneutral':
        args.data_path = './data/ML10M_disneutral'
        args.behaviors = [ 'neg', 'pos']
    elif args.data_name == 'taobao':
        args.data_path = './data/taobao'
        args.behaviors = ['view','cart', 'buy']
    elif args.data_name == 'taobao_add_5':
        args.data_path = './data/taobao_add_5'
        args.behaviors = ['view','cart', 'buy']
    elif args.data_name == 'taobao_add_10':
        args.data_path = './data/taobao_add_10'
        args.behaviors = ['view','cart', 'buy']
    elif args.data_name == 'taobao_add_15':
        args.data_path = './data/taobao_add_15'
        args.behaviors = ['view','cart', 'buy']
    elif args.data_name == 'taobao_add_20':
        args.data_path = './data/taobao_add_20'
        args.behaviors = ['view','cart', 'buy']
    elif args.data_name == 'taobao_delete10':
        args.data_path = './data/taobao_delete10'
        args.behaviors = ['view','cart', 'buy']
    elif args.data_name == 'taobao_delete30':
        args.data_path = './data/taobao_delete30'
        args.behaviors = ['view','cart', 'buy']
    elif args.data_name == 'taobao_delete50':
        args.data_path = './data/taobao_delete50'
        args.behaviors = ['view','cart', 'buy']
    elif args.data_name == 'taobao_add10':
        args.data_path = './data/taobao_add10'
        args.behaviors = ['view','cart', 'buy']
    elif args.data_name == 'taobao_add30':
        args.data_path = './data/taobao_add30'
        args.behaviors = ['view','cart', 'buy']
    elif args.data_name == 'taobao_add50':
        args.data_path = './data/taobao_add50'
        args.behaviors = ['view','cart', 'buy']
    elif args.data_name == 'taobao_disview':
        args.data_path = './data/taobao_disview'
        args.behaviors = ['cart', 'buy']
    elif args.data_name == 'taobao_discart':
        args.data_path = './data/taobao_discart'
        args.behaviors = ['view', 'buy']
    elif args.data_name == 'beibei':
        args.data_path = './data/beibei'
        args.behaviors = ['ipv','cart', 'buy']
    elif args.data_name == 'Tmall_cart_perturbed':
        args.data_path = './data/Tmall_cart_perturbed'
        args.behaviors = ['click', 'collect', 'cart', 'buy']
    elif args.data_name == 'tmall_discart':
        args.data_path = './data/Tmall_discart'
        args.behaviors = ['click', 'collect', 'buy']
    elif args.data_name == 'tmall_disclick':
        args.data_path = './data/Tmall_disclick'
        args.behaviors = ['collect', 'cart', 'buy']
    elif args.data_name == 'tmall_discollect':
        args.data_path = './data/Tmall_discollect'
        args.behaviors = ['click','cart', 'buy']   
    elif args.data_name == 'ijcai':
        args.data_path = './data/IJCAI'
        args.behaviors = ['click', 'collect', 'cart', 'buy'] 
    else:
        raise Exception('data_name cannot be None')

    if getattr(args, 'disable_orm', False):
        args.reg_coeff = 0.0
        args.decay = 0.0

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # args.device = device

    TIME = time.strftime("%Y-%m-%d %H_%M_%S", time.localtime())
    args.TIME = TIME

    logfile = '{}_enb_{}_{}'.format(args.data_name, args.embedding_size, TIME)
    args.train_writer = SummaryWriter('./log/train/' + logfile)
    args.test_writer = SummaryWriter('./log/test/' + logfile)
    logger.add('./log/{}/{}.log'.format(args.model_name, logfile), encoding='utf-8')

    start = time.time()
    dataset = DataSet(args)
    model = MBGCN(args, dataset)

    logger.info(args.__str__())
    logger.info(model)
    trainer = Trainer(model, dataset, args)
    trainer.train_model()
    # trainer.evaluate(0, 1, dataset.test_dataset(), dataset.test_interacts, dataset.test_gt_length, args.test_writer)
    logger.info('train end total cost time: {}'.format(time.time() - start))



