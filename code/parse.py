import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Go LT-NCF")
    parser.add_argument('--bpr_batch', type=int,default=2048,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--recdim', type=int,default=64,
                        help="the embedding size of LT-NCF")
    parser.add_argument('--layer', type=int,default=3,
                        help="the layer num of LT-NCF")
    parser.add_argument('--lr', type=float,default=0.001,
                        help="the learning rate")
    parser.add_argument('--lr_time', type=float,default=0.0001,
                        help="the learning rate")                        
    parser.add_argument('--decay', type=float,default=1e-4,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--dropout', type=int,default=0,
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float,default=0.6,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--a_fold', type=int,default=100,
                        help="the fold num used to split large adj matrix, like gowalla")
    parser.add_argument('--testbatch', type=int,default=100,
                        help="the batch size of users for testing")
    parser.add_argument('--dataset', type=str,default='gowalla',
                        help="available datasets: [lastfm, gowalla, yelp2018, amazon-book]")
    parser.add_argument('--path', type=str,default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?',default="[20]",
                        help="@k test list, e.g. [10,20,30,40,50]")
    parser.add_argument('--tensorboard', type=int,default=1,
                        help="enable tensorboard")
    parser.add_argument('--comment', type=str,default="lt-ncf")
    parser.add_argument('--load', type=int,default=0)
    parser.add_argument('--epochs', type=int,default=1000)
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--model', type=str, default='ltocf', help='rec-model, support [mf, lgn, ltocf, ltocf2, ltocf1]')
    parser.add_argument('--timesplit', type=int, default='4', help='split time e.g. timesplit=4 -> #T=3, timesplit=3 -> #T=2')
    parser.add_argument('--gpuid', type=int, default=0, help="Please give a value for gpu id")
    parser.add_argument('--solver', type=str, default='euler', help="ode solver: [dopri5, euler, rk4, adaptive_heun, bosh3, explicit_adams, implicit_adams]")
    parser.add_argument('--adjoint', type=eval, default=False, choices=[True, False])
    parser.add_argument('--learnable_time', type=eval, default=False, choices=[True, False])
    parser.add_argument('--dual_res', type=eval, default=False, choices=[True, False])
    parser.add_argument('--parallel', type=eval, default=False, choices=[True, False])
    parser.add_argument('--rtol', type=float,default=1e-7,help="rtol")
    parser.add_argument('--atol', type=float,default=1e-9,help="atol")
    parser.add_argument('--pretrained_file', type=str,default="ltocf")
    parser.add_argument('--K', type=float, default=4, help="final integral time K")
    return parser.parse_args()
