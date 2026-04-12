"""
run_partial.py — Entry point for partial-loss fine-tuning.

Identical to run.py except:
  1. Uses Exp_Forecast_Partial (loss on first --loss_pred_len steps per token)
  2. Adds --loss_pred_len argument (default: output_token_len, i.e. no change)
  3. test() is inherited unchanged — full 96-step predictions are still saved

Example:
    python run_partial.py --loss_pred_len 15 --is_training 1 ...
"""
import os
import argparse
import random
import numpy as np
import torch
import torch.distributed as dist
from exp.exp_forecast_partial import Exp_Forecast_Partial

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Timer-XL (partial loss)')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='forecast')
    parser.add_argument('--is_training', type=int, required=True, default=1)
    parser.add_argument('--model_id', type=str, required=True, default='test')
    parser.add_argument('--model', type=str, required=True, default='timer_xl')
    parser.add_argument('--seed', type=int, default=2021)

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT-small/')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv')
    parser.add_argument('--split_file', type=str, default=None)
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default=None)
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    parser.add_argument('--test_flag', type=str, default='T')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=672)
    parser.add_argument('--input_token_len', type=int, default=576)
    parser.add_argument('--output_token_len', type=int, default=96)
    parser.add_argument('--test_seq_len', type=int, default=672)
    parser.add_argument('--test_pred_len', type=int, default=96)

    # === NEW: partial loss ===
    parser.add_argument('--loss_pred_len', type=int, default=None,
                        help='Only compute loss on the first N steps of each token prediction. '
                             'Default: same as output_token_len (full token, no change).')

    # model define
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--e_layers', type=int, default=1)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--covariate', action='store_true', default=False)
    parser.add_argument('--node_num', type=int, default=100)
    parser.add_argument('--node_list', type=str, default='23,37,40')
    parser.add_argument('--use_norm', action='store_true', default=False)
    parser.add_argument('--nonautoregressive', action='store_true', default=False)
    parser.add_argument('--test_dir', type=str, default='./test')
    parser.add_argument('--test_file_name', type=str, default='checkpoint.pth')
    parser.add_argument('--output_attention', action='store_true', default=False)
    parser.add_argument('--visualize', action='store_true', default=False)
    parser.add_argument('--flash_attention', action='store_true', default=False)

    # adaptation
    parser.add_argument('--adaptation', action='store_true', default=False)
    parser.add_argument('--pretrain_model_path', type=str, default='pretrain_model.pth')
    parser.add_argument('--subset_rand_ratio', type=float, default=1)

    # optimization
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--des', type=str, default='test')
    parser.add_argument('--loss', type=str, default='MSE')
    parser.add_argument('--lradj', type=str, default='type1')
    parser.add_argument('--cosine', action='store_true', default=False)
    parser.add_argument('--tmax', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--valid_last', action='store_true', default=False)
    parser.add_argument('--last_token', action='store_true', default=False)

    # GPU
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--ddp', action='store_true', default=False)
    parser.add_argument('--dp', action='store_true', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3')

    # LLM-based model
    parser.add_argument('--gpt_layers', type=int, default=6)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--kernel_size', type=int, default=25)
    parser.add_argument('--stride', type=int, default=8)

    # TTM
    parser.add_argument("--n_vars", type=int, default=7)
    parser.add_argument("--factor", type=int, default=2)
    parser.add_argument("--mode", type=str, default="mix_channel")
    parser.add_argument("--AP_levels", type=int, default=0)
    parser.add_argument("--use_decoder", action="store_true", default=True)
    parser.add_argument("--d_mode", type=str, default="common_channel")
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=16)

    # Time-LLM
    parser.add_argument("--ts_vocab_size", type=int, default=1000)
    parser.add_argument("--domain_des", type=str, default="")
    parser.add_argument("--llm_model", type=str, default="LLAMA")
    parser.add_argument("--llm_layers", type=int, default=6)

    args = parser.parse_args()

    # default loss_pred_len = output_token_len (full token, backward compatible)
    if args.loss_pred_len is None:
        args.loss_pred_len = args.output_token_len

    fix_seed = args.seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    args.node_list = [int(x) for x in args.node_list.split(',')]

    if args.dp:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    elif args.ddp:
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", "64209")
        hosts = int(os.environ.get("WORLD_SIZE", "8"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        gpus = torch.cuda.device_count()
        args.local_rank = local_rank
        print(ip, port, hosts, rank, local_rank, gpus)
        dist.init_process_group(backend="nccl", init_method=f"tcp://{ip}:{port}",
                                world_size=hosts, rank=rank)
        torch.cuda.set_device(local_rank)

    Exp = Exp_Forecast_Partial

    if args.is_training:
        for ii in range(args.itr):
            exp = Exp(args)
            setting = '{}_{}_{}_{}_sl{}_it{}_ot{}_lr{}_bt{}_wd{}_el{}_dm{}_dff{}_nh{}_cos{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.seq_len,
                args.input_token_len,
                args.output_token_len,
                args.learning_rate,
                args.batch_size,
                args.weight_decay,
                args.e_layers,
                args.d_model,
                args.d_ff,
                args.n_heads,
                args.cosine,
                args.des, ii)
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            if not args.ddp and not args.dp:
                exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_sl{}_it{}_ot{}_lr{}_bt{}_wd{}_el{}_dm{}_dff{}_nh{}_cos{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.seq_len,
            args.input_token_len,
            args.output_token_len,
            args.learning_rate,
            args.batch_size,
            args.weight_decay,
            args.e_layers,
            args.d_model,
            args.d_ff,
            args.n_heads,
            args.cosine,
            args.des, ii)
        exp = Exp(args)
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
