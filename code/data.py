import argparse #接收命令行参数
import os #创建文件夹、处理路径
import random
from random import sample #随机采样

import pandas as pd #数据处理
from tqdm import tqdm #进度条显示

from utils import Logger #项目自己封装的日志工具

random.seed(2020)

# 命令行参数
parser = argparse.ArgumentParser(description='数据处理')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'数据处理，mode: {mode}')


def data_offline(df_train_click, df_test_click):
    # 把每一行的 user_id 都拿出来。所以同一个用户点了多篇新闻，会出现多次。
    train_users = df_train_click['user_id'].values.tolist()
    # 随机采样出一部分样本，从 train_users 里随机抽 50000 个用户 ID，作为线下验证用户。
    #因为 train_users 没去重，所以热门/点击多的用户更容易被抽中。
    val_users = sample(train_users, 50000)
    log.debug(f'val_users num: {len(set(val_users))}')

    # 训练集用户 抽出行为数据最后一条作为线下验证集
    click_list = [] # 装“保留下来的历史点击”
    valid_query_list = [] # 装“被藏起来的最后一次点击答案”

    # groupby('user_id') 的意思是按用户分组
    groups = df_train_click.groupby(['user_id'])
    for user_id, g in tqdm(groups):
         # g 是当前用户的所有点击记录，是一个 DataFrame
         # user_id 当前分组名，也就是用户ID
        if user_id in val_users:
            # 如果这个用户被抽中了做验证用户，就把他的最后一次点击藏起来。
            valid_query = g.tail(1) # tail(1) 是取最后一行，head(1) 是取第一行
            valid_query_list.append(
                valid_query[['user_id', 'click_article_id']])

            train_click = g.head(g.shape[0] - 1)# head(g.shape[0] - 1) 是取前面所有行，去掉最后一行
            click_list.append(train_click)
        else:
            click_list.append(g)

    df_train_click = pd.concat(click_list, sort=False)
    df_valid_query = pd.concat(valid_query_list, sort=False)

    test_users = df_test_click['user_id'].unique()
    test_query_list = []

    for user in tqdm(test_users):
        test_query_list.append([user, -1])

    df_test_query = pd.DataFrame(test_query_list,
                                 columns=['user_id', 'click_article_id'])
    
    #df_valid_query：训练集中被藏起来的答案
    #df_test_query：测试集用户，答案未知，用 -1 表示
    # 要预测谁
    df_query = pd.concat([df_valid_query, df_test_query],
                         sort=False).reset_index(drop=True)
    
    # 根据哪些历史点击来预测
    #训练用户的历史点击，但不包括被藏起来的验证答案
    #测试用户的历史点击
    df_click = pd.concat([df_train_click, df_test_click],
                         sort=False).reset_index(drop=True) #.reset_index(drop=True) 重置行号
    
    # 先按 user_id 排，同一个用户内部，再按 click_timestamp 点击时间排
    df_click = df_click.sort_values(['user_id',
                                     'click_timestamp']).reset_index(drop=True)

    log.debug(
        f'df_query shape: {df_query.shape}, df_click shape: {df_click.shape}')# df_click.shape表示表有多少行、多少列。
    log.debug(f'{df_query.head()}') # 表示打印前 5 行，方便检查数据长什么样。
    log.debug(f'{df_click.head()}')

    # 保存文件
    os.makedirs('../user_data/data/offline', exist_ok=True)

    df_click.to_pickle('../user_data/data/offline/click.pkl')
    df_query.to_pickle('../user_data/data/offline/query.pkl')


def data_online(df_train_click, df_test_click):
    test_users = df_test_click['user_id'].unique()
    test_query_list = []

    for user in tqdm(test_users):
        test_query_list.append([user, -1])

    df_test_query = pd.DataFrame(test_query_list,
                                 columns=['user_id', 'click_article_id'])

    df_query = df_test_query
    df_click = pd.concat([df_train_click, df_test_click],
                         sort=False).reset_index(drop=True)
    df_click = df_click.sort_values(['user_id',
                                     'click_timestamp']).reset_index(drop=True)

    log.debug(
        f'df_query shape: {df_query.shape}, df_click shape: {df_click.shape}')
    log.debug(f'{df_query.head()}')
    log.debug(f'{df_click.head()}')

    # 保存文件
    os.makedirs('../data/online', exist_ok=True)

    df_click.to_pickle('../user_data/data/online/click.pkl')
    df_query.to_pickle('../user_data/data/online/query.pkl')


if __name__ == '__main__':
    df_train_click = pd.read_csv('../tcdata/train_click_log.csv')
    df_test_click = pd.read_csv('../tcdata/testB_click_log_Test_B.csv')

    log.debug(
        f'df_train_click shape: {df_train_click.shape}, df_test_click shape: {df_test_click.shape}'
    )

    if mode == 'valid':
        data_offline(df_train_click, df_test_click)
    else:
        data_online(df_train_click, df_test_click)
