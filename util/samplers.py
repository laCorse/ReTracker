from __future__ import absolute_import
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data.sampler import Sampler

class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, num_instance=4):
        """

        :param data_source: data loader
        :param num_instance: 每个id挑选几张图片
        """
        self.data_source = data_source
        self.num_instance = num_instance

        # dic:
        # key:每个ID
        # value:ID对应的所有图片的序号
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)


    def __iter__(self):
        """
        返回一个epoch中需要的所有数据
        比如如果是751个id，每个id选择4张图
        [id1_1, id1_2, id1_3, id1_4, id2_1, id2_2, ..., id751_4]
        如果batch是32，则第一个batch会挑选到id8_4
        :return:
        """
        indices = torch.randperm(self.num_identities)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            # np.random.choice作用为从t列表中挑选出size个，replace是是否采样能重复
            # 因此，如果当长度小于self.num_instance个时候就应该允许重复
            replace = True if len(t) < self.num_instance else False
            t = np.random.choice(t, size=self.num_instance, replace=replace)
            # extend是直接相连，append是放入元素
            ret.extend(t)
        return iter(ret)

    def __len__(self):
        return self.num_instance*self.num_identities



if __name__ == '__main__':
    from util.data_manager import Market1501
    dataset = Market1501()
    sampler = RandomIdentitySampler(dataset.train,num_instance=4)
