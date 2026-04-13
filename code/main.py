import argparse
import logging
import os
import random
from data.construct_loader import Fault_dataset
from utils.SetSeed import set_random_seed
from utils.logger import  result_log, setup_logging
from utils.train_test import train_test


def parse_args():
    parser = argparse.ArgumentParser(description='Train')

    #default_dataset = "Bearing.BJTU"
    default_dataset = "GearBox.BJUT"


    ''' ================= Data-related parameters ================= '''
    parser.add_argument('--dataset_name', type=str, default=default_dataset, help='name of dataset',  choices=["Bearing.BJTU", "GearBox.BJUT"])
    parser.add_argument('--source_id', type=str, default='1800',     help='source domain')
    parser.add_argument('--data_ratio', type=int, default=0.5,help='percentage of dataset division')
    parser.add_argument('--miss_class', nargs='+', type=int, default=[],   help='deleting labels from a class')
    parser.add_argument('--FFT', type=bool, default=False,  help='whether to Fourier transform the data')
    parser.add_argument('--normalize_type', type=str, default='mean-std',  help='data normalization methods',choices=['0', '0-1', '-1-1', 'mean-std'])
    ''' ================= Training related parameters ================= '''
    parser.add_argument('--model_name', type=str, default='DWCN', help='the name of the model',choices=['B1', 'DWCN'])
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--lr', type=float, default=0.01, help='the initial learning rate')
    parser.add_argument('--epoch', type=int, default=100, help='the max number of epoch')
    parser.add_argument('--operation_num', type=int, default=1, help='the repeat operation of model')
    parser.add_argument('--base_seed', type=int, default=42, help='base torch/random seed used for model initialization and training randomness')
    parser.add_argument('--vary_seed_per_operation', type=int, default=0, choices=[0, 1], help='whether to use base_seed + i for each repeated run')
    parser.add_argument('--deterministic', type=int, default=1, choices=[0, 1], help='whether to enable deterministic CUDA behavior')
    ''' ================= Frequency Augmentation parameters ================= '''
    parser.add_argument('--sigma', type=float, default=0.1, help='amplitude scaling strength for freq_aug')
    parser.add_argument('--noise_std', type=float, default=0, help='noise injection strength (0 for GearBox, >0 for Bearing)')
    parser.add_argument('--step_size', type=int, default=100, help='learning rate decay period (epochs)')
    parser.add_argument('--split_seed_base', type=int, default=-1, help='base seed used to replay the exact train/test split of one run')
    parser.add_argument('--use_contrastive_loss', type=int, default=0, choices=[0, 1], help='whether to enable the contrastive-loss branch')
    parser.add_argument('--alpha_contrastive', type=float, default=0.005, help='weight of the contrastive loss')
    args = parser.parse_args()
    return args

def train_and_evaluate(args,operation, dataset, source_id, target_id):
    global target_list_string
    target_list_string = "-".join(map(str, target_id))
    accuracy_t, f1_values_t = [], []
    for i in range(args.operation_num):
        run_seed = args.base_seed + i if args.vary_seed_per_operation else args.base_seed
        set_random_seed(run_seed, deterministic=bool(args.deterministic))
        split_seed_base = args.split_seed_base if args.split_seed_base >= 0 else random.SystemRandom().randrange(0, 2**31 - 1)
        source_split_seed = split_seed_base + int(source_id)
        target_split_seeds = {str(domain): split_seed_base + int(domain) for domain in target_id}
        source_train_loader, source_test_loader = dataset.Loader([source_id], train=True, split_seed_base=split_seed_base)
        target_test_loader, _ = dataset.Loader(target_id, train=False, split_seed_base=split_seed_base)
        logging.info(
            "Operation_%d replay_info: torch_seed=%d, deterministic=%d, split_seed_base=%d, source_split_seed=%d, target_split_seeds=%s",
            i,
            run_seed,
            args.deterministic,
            split_seed_base,
            source_split_seed,
            target_split_seeds,
        )
        logging.info("Train_Source: %s | Test_Target: %s", source_id, target_list_string)
        # ---------- 训练 ----------
        operation.setup(dataset.n_class)
        operation.train(i, source_train_loader, source_test_loader, target_test_loader, source_id)
        # ---------- 测试 ----------
        acc_t, f1_t = operation.test(i, source_test_loader, target_test_loader, source_id)
        accuracy_t.append(acc_t)
        f1_values_t.append(f1_t)
        result_log(Indicators="Ac_t", target=target_list_string, source=source_id, results=accuracy_t)
        result_log(Indicators="F1_t", target=target_list_string, source=source_id, results=f1_values_t)






if __name__ == '__main__':
    args = parse_args()
    Dataset = Fault_dataset(args)
    operation = train_test(args)
    setattr(args, 'num_class', Dataset.n_class)
    # 设置日志
    save_dir = os.path.join('./results/{}'.format(args.dataset_name))
    setup_logging(args, save_dir)
    # 准备任务
    task_mapping = Dataset.task_loaders(Dataset)
    target_id = task_mapping[args.source_id]
    # 训练和评估模型
    train_and_evaluate(args,operation, Dataset,args.source_id,target_id)

