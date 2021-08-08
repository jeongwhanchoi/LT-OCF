import os
from os.path import join
import torch
from torch.nn import parallel
from parse import parse_args
import multiprocessing

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_args()
ROOT_PATH = os.path.dirname(os.path.abspath(__file__ + "/../"))
CODE_PATH = join(ROOT_PATH, 'code')
DATA_PATH = join(ROOT_PATH, 'data')
BOARD_PATH = join(CODE_PATH, 'runs')
FILE_PATH = join(CODE_PATH, 'checkpoints')
PRETRAINED_FILE_PATH = join(CODE_PATH, 'pretrain')
import sys
sys.path.append(join(CODE_PATH, 'sources'))

if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)


config = {}
all_dataset = ['lastfm', 'gowalla', 'yelp2018', 'amazon-book']
all_models  = ['mf', 'lgn', 'ltocf', 'ltocf2', 'ltocf1']
# config['batch_size'] = 4096
config['bpr_batch_size'] = args.bpr_batch
config['latent_dim_rec'] = args.recdim
config['lightGCN_n_layers']= args.layer
config['dropout'] = args.dropout
config['keep_prob']  = args.keepprob
config['A_n_fold'] = args.a_fold
config['test_u_batch_size'] = args.testbatch
config['multicore'] = args.multicore
config['lr'] = args.lr
config['lr_time'] = args.lr_time
config['decay'] = args.decay
config['pretrain'] = args.pretrain
config['A_split'] = False
config['bigdata'] = False
config['time_split'] = args.timesplit
config['solver'] = args.solver
config['learnable_time'] = args.learnable_time
config['dual_res'] = args.dual_res
config['pretrained_file_name'] = args.pretrained_file
config['K'] = args.K

CORES = multiprocessing.cpu_count() // 2
seed = args.seed

GPU_NUM = args.gpuid
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device) # change allocation of current GPU

parallel = args.parallel

dataset = args.dataset
model_name = args.model
if dataset not in all_dataset:
    raise NotImplementedError(f"Haven't supported {dataset} yet!, try {all_dataset}")
if model_name not in all_models:
    raise NotImplementedError(f"Haven't supported {model_name} yet!, try {all_models}")


adjoint = args.adjoint
rtol = args.rtol
atol = args.atol

TRAIN_epochs = args.epochs
LOAD = args.load
PATH = args.path
topks = eval(args.topks)
tensorboard = args.tensorboard
comment = args.comment
# let pandas shut up
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)



def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")

logo = r"""
 ██████╗████████╗    ██████╗  ██████╗███████╗
██╔════╝╚══██╔══╝   ██╔═══██╗██╔════╝██╔════╝
██║        ██║█████╗██║   ██║██║     █████╗  
██║        ██║╚════╝██║   ██║██║     ██╔══╝  
╚██████╗   ██║      ╚██████╔╝╚██████╗██║     
 ╚═════╝   ╚═╝       ╚═════╝  ╚═════╝╚═╝     
                                                      
"""
# font: ANSI Shadow
# http://patorjk.com/software/taag/#p=display&f=ANSI%20Shadow&t=CT-OCF
print(logo)
print ('Current cuda device ', torch.cuda.current_device()) # check