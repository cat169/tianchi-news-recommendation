import argparse
import math # 需要导入 math 模块来使用 math.log 函数
import os
import pickle #保存 Python 对象，比如 item_sim 相似度字典
import random
import signal
from collections import defaultdict #创建默认值字典，统计次数很方便
from random import shuffle

import multitasking # 多进程并行召回
import numpy as np
import pandas as pd 
from tqdm import tqdm

from utils import Logger, evaluate  # Logger 用于日志记录，evaluate 用于计算召回指标

max_threads = multitasking.config['CPU_CORES']
multitasking.set_max_threads(max_threads)
# thread：多线程 像一个厨房里多个厨师共用一套工具
# process：多进程 像开了多个独立厨房，每个厨房有自己的工具
multitasking.set_engine('process')
# 捕获 Ctrl+C 信号，终止所有子进程，signal.SIGINT 是 Ctrl+C 的信号，multitasking.killall 是终止所有子进程的函数
signal.signal(signal.SIGINT, multitasking.killall)

random.seed(2020)

# 命令行参数
parser = argparse.ArgumentParser(description='itemcf 召回')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'itemcf 召回，mode: {mode}')


def cal_sim(df):
    # 将每个用户的点击文章 ID 列表化，得到一个新的 DataFrame，
    # 其中 user_id 是用户 ID，click_article_id 是一个列表，包含该用户点击过的所有文章 ID
    # .agg(lambda x: list(x)) 
    # .reset_index() 
    user_item_ = df.groupby('user_id')['click_article_id'].agg(
        lambda x: list(x)).reset_index() 
    
    user_item_dict = dict(
        zip(user_item_['user_id'], user_item_['click_article_id']))
    # {
    #     1001: [A, B, C],
    #     1002: [A, D]
    # }


    item_cnt = defaultdict(int) # 统计每个文章被点击的次数，默认值为 0
    sim_dict = {} # 存储每个文章与其他文章的相似度，默认值为空字典

    for _, items in tqdm(user_item_dict.items()):
        for loc1, item in enumerate(items):
            item_cnt[item] += 1
            sim_dict.setdefault(item, {})

            for loc2, relate_item in enumerate(items):
                if item == relate_item:
                    continue

                sim_dict[item].setdefault(relate_item, 0)

                # 位置信息权重
                # 考虑文章的正向顺序点击和反向顺序点击
                # 用户点了 A 后又点 B，比先点 B 再点 A，对 A→B 推荐更有参考价值。
                loc_alpha = 1.0 if loc2 > loc1 else 0.7
                
                # 考虑距离。
                # 如果两篇文章在点击序列里越近，权重越大。
                loc_weight = loc_alpha * (0.9**(np.abs(loc2 - loc1) - 1))

                sim_dict[item][relate_item] += loc_weight  / \
                    math.log(1 + len(items))
                # 在同一个用户历史里共同出现一次，就给它们的相似度加
                # 位置权重 / log(1 + 用户点击序列长度)
                # 因为有些用户点击很多新闻，比如一天点了 200 篇。
                # 如果他点过 A 和 B，不一定说明 A 和 B 很相关，可能只是他什么都点
                # 所以点击越多的用户，贡献要被削弱。

    for item, relate_items in tqdm(sim_dict.items()):
        for relate_item, cij in relate_items.items():
            sim_dict[item][relate_item] = cij / \
                math.sqrt(item_cnt[item] * item_cnt[relate_item])
            #前面算的是“共现分数”。但热门文章天然会和很多文章共同出现。
            # 比如热门新闻 A 被 10 万人点过，它很容易和任何文章一起出现。
            # 如果不处理，热门文章会到处相似。

    return sim_dict, user_item_dict

# 给 query.pkl 里的每个用户，召回一批候选新闻。
@multitasking.task # 这个装饰器表示这个函数是一个多进程任务，可以并行执行
def recall(df_query, item_sim, user_item_dict, worker_id):# worker_id：当前并行任务编号，用来保存临时文件
    
    data_list = []

    for user_id, item_id in tqdm(df_query.values):
        rank = {} # 存储候选新闻的相似度得分，key 是候选新闻 ID，value 是相似度得分

        # 如果用户没有点击过任何新闻，就无法基于历史点击进行推荐，所以直接跳过这个用户
        if user_id not in user_item_dict:
            continue

        interacted_items = user_item_dict[user_id]
        interacted_items = interacted_items[::-1][:2] # [::-1] 表示将列表反转，[:2] 表示取前两篇文章。也就是说，只考虑用户最近点击的两篇文章来进行推荐。

        for loc, item in enumerate(interacted_items):
            
            # item_sim[item] 是当前文章的相似文章字典。
            # item_sim[D] = {
            #     X: 0.9,
            #     Y: 0.7,
            #     Z: 0.4
            # }
            # .items() 变成： (X, 0.9), (Y, 0.7), (Z, 0.4)
            # sorted(..., key=lambda d: d[1], reverse=True) 按相似度分数从高到低排序
            for relate_item, wij in sorted(item_sim[item].items(),
                                           key=lambda d: d[1],
                                           reverse=True)[0:200]:
                # 如果用户最近已经点过某篇文章，就不要再推荐它。
                if relate_item not in interacted_items:
                    rank.setdefault(relate_item, 0)
                    # 按相似度和时间衰减累计得分
                    rank[relate_item] += wij * (0.7**loc)

        sim_items = sorted(rank.items(), key=lambda d: d[1],
                           reverse=True)[:100]
        item_ids = [item[0] for item in sim_items]
        item_sim_scores = [item[1] for item in sim_items]

        df_temp = pd.DataFrame()
        df_temp['article_id'] = item_ids
        df_temp['sim_score'] = item_sim_scores
        df_temp['user_id'] = user_id

        if item_id == -1:
            df_temp['label'] = np.nan
        else:
            df_temp['label'] = 0
            df_temp.loc[df_temp['article_id'] == item_id, 'label'] = 1

        df_temp = df_temp[['user_id', 'article_id', 'sim_score', 'label']]
        df_temp['user_id'] = df_temp['user_id'].astype('int')
        df_temp['article_id'] = df_temp['article_id'].astype('int')

        data_list.append(df_temp)

    df_data = pd.concat(data_list, sort=False)

    os.makedirs('../user_data/tmp/itemcf', exist_ok=True)
    df_data.to_pickle(f'../user_data/tmp/itemcf/{worker_id}.pkl')


if __name__ == '__main__':
    if mode == 'valid':
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')

        os.makedirs('../user_data/sim/offline', exist_ok=True)
        sim_pkl_file = '../user_data/sim/offline/itemcf_sim.pkl'
    else:
        df_click = pd.read_pickle('../user_data/data/online/click.pkl')
        df_query = pd.read_pickle('../user_data/data/online/query.pkl')

        os.makedirs('../user_data/sim/online', exist_ok=True)
        sim_pkl_file = '../user_data/sim/online/itemcf_sim.pkl'

    log.debug(f'df_click shape: {df_click.shape}')
    log.debug(f'{df_click.head()}')

    item_sim, user_item_dict = cal_sim(df_click)
    f = open(sim_pkl_file, 'wb')
    pickle.dump(item_sim, f)
    f.close()

    # 召回
    n_split = max_threads
    all_users = df_query['user_id'].unique()
    shuffle(all_users)
    total = len(all_users)
    n_len = total // n_split

    # 清空临时文件夹
    for path, _, file_list in os.walk('../user_data/tmp/itemcf'):
        for file_name in file_list:
            os.remove(os.path.join(path, file_name))

    for i in range(0, total, n_len):
        part_users = all_users[i:i + n_len]
        df_temp = df_query[df_query['user_id'].isin(part_users)]
        recall(df_temp, item_sim, user_item_dict, i)

    multitasking.wait_for_tasks()
    log.info('合并任务')

    df_data = pd.DataFrame()
    for path, _, file_list in os.walk('../user_data/tmp/itemcf'):
        for file_name in file_list:
            df_temp = pd.read_pickle(os.path.join(path, file_name))
            df_data = df_data.append(df_temp)

    # 必须加，对其进行排序
    df_data = df_data.sort_values(['user_id', 'sim_score'],
                                  ascending=[True,
                                             False]).reset_index(drop=True)
    log.debug(f'df_data.head: {df_data.head()}')

    # 计算召回指标
    if mode == 'valid':
        log.info(f'计算召回指标')

        total = df_query[df_query['click_article_id'] != -1].user_id.nunique()

        hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
            df_data[df_data['label'].notnull()], total)

        log.debug(
            f'itemcf: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
        )
    # 保存召回结果
    if mode == 'valid':
        df_data.to_pickle('../user_data/data/offline/recall_itemcf.pkl')
    else:
        df_data.to_pickle('../user_data/data/online/recall_itemcf.pkl')
