import random
import itertools
import numpy as np
from torch.utils.data import Dataset
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler


SEEDS = [7, 9, 10, 20, 21, 29, 32, 41, 42, 56, 95, 103, 118, 141, 145, 150, 156, 164, 176, 198, 204, 236, 238, 244, 248, 250, 260, 282, 285, 301, 305, 317, 318, 331, 335, 341, 348, 350, 353, 354, 369, 371, 394, 400, 402, 414, 428, 430, 440, 460, 471, 483, 497, 513, 519, 527, 530, 537, 558, 562, 574, 585, 614, 629, 634, 635, 672, 679, 685, 686, 691, 699, 760, 761, 762, 767, 774, 786, 804, 807, 822, 826, 840, 847, 858, 862, 869, 876, 880, 894, 896, 898, 908, 923, 925, 936, 961, 962, 977, 989]


class HearDataset_MAE(Dataset):
    def __init__(self, df, prev_len, mode):
        self.x, self.y = [], []
        self.prev_len = prev_len
        self.mode = mode
        self.process_data(df)
        self.x = np.array(self.x)
        self.y = np.array(self.y)

    def process_data(self, df):
        side_cols = [
            'Age',
            'Age_diff',
            'Genotype',
            'Gender',
            # 'AC025',
            # 'AC05',
            # 'AC1',
            # 'AC2',
            # 'AC4',
            # 'AC8',
            'AC025_trend',
            'AC05_trend',
            'AC1_trend',
            'AC2_trend',
            'AC4_trend',
            'AC8_trend',
            'AC025_trend_label',
            'AC05_trend_label',
            'AC1_trend_label',
            'AC2_trend_label',
            'AC4_trend_label',
            'AC8_trend_label',
            'AC025_std_in_time',
            'AC05_std_in_time',
            'AC1_std_in_time',
            'AC2_std_in_time',
            'AC4_std_in_time',
            'AC8_std_in_time',
            'avg_trend_labels_0',
            'avg_trend_labels_1',
            'avg_trend_labels_2',
            'avg_trend_labels_3',
            'avg_trend_labels_4',
            # 'four_avg',
            # 'avg1',
            # 'avg2',
            # 'avg3',
            # 'avg4',
            # 'four_avg',
            # 'VAsize',
            # 'mondini',
            'peak',
            'peak_agg',
            'peak_diff',
            'big_peak' ,
            'big_peak_agg' ,
            'big_peak_diff',
            'recently_peak',
            'recently_big_peak',
            'recently_total',
            ]
        for _, row in df.iterrows():
            data_len = len(row['Age'])
            if data_len < self.prev_len + 1:
                continue
            for i in range(data_len - self.prev_len):
                if self.mode == '1D':
                    subarr_x = []
                    for j in range(self.prev_len):
                        subarr_x.extend(self.create_subarray(row, side_cols, i + j, 0))
                    subarr_y = self.create_subarray(row, side_cols, i + self.prev_len, 1)
                    subarr_x.append(subarr_y[0])
                    subarr_x.append(subarr_y[1])
                    self.x.append(subarr_x)
                    self.y.append(subarr_y)
                elif self.mode == 'RNN':
                    subarr_x = [self.create_subarray(row, side_cols, i + j, 0) for j in range(self.prev_len)]
                    subarr_y = self.create_subarray(row, side_cols, i + self.prev_len, 1)
                    self.x.append(subarr_x)
                    self.y.append(subarr_y)



    def create_subarray(self, row, cols, index, istest):
        subarr = []
        for col in cols:
            subarr.append(row[col][index])
        if istest:
            subarr.append(row['Age'][index-1])
            subarr.append(row['Age_diff'][index-1])
            subarr.append(row['occur'][index-1])
        return subarr

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class HearDataset_Peak(Dataset):
    def __init__(self, df, prev_len, mode):
        self.x, self.y = [], []
        self.prev_len = prev_len
        self.mode = mode
        self.process_data(df)
        self.x = np.array(self.x)
        self.y = np.array(self.y)

    def process_data(self, df):
        side_cols = [
            'Age',
            'Age_diff',
            'Genotype',
            'Gender',
            # 'AC025',
            # 'AC05',
            # 'AC1',
            # 'AC2',
            # 'AC4',
            # 'AC8',
            'AC025_trend',
            'AC05_trend',
            'AC1_trend',
            'AC2_trend',
            'AC4_trend',
            'AC8_trend',
            'AC025_trend_label',
            'AC05_trend_label',
            'AC1_trend_label',
            'AC2_trend_label',
            'AC4_trend_label',
            'AC8_trend_label',
            'AC025_std_in_time',
            'AC05_std_in_time',
            'AC1_std_in_time',
            'AC2_std_in_time',
            'AC4_std_in_time',
            'AC8_std_in_time',
            'avg_trend_labels_0',
            'avg_trend_labels_1',
            'avg_trend_labels_2',
            'avg_trend_labels_3',
            'avg_trend_labels_4',
            # 'four_avg',
            # 'avg1',
            # 'avg2',
            # 'avg3',
            # 'avg4',
            # 'four_avg',
            # 'VAsize',
            # 'mondini',
            'peak',
            'peak_agg',
            'peak_diff',
            'big_peak' ,
            'big_peak_agg' ,
            'big_peak_diff',
            'recently_peak',
            'recently_big_peak',
            'recently_total',
            ]
        for _, row in df.iterrows():
            data_len = len(row['Age'])
            if data_len < self.prev_len:
                continue
            for i in range(data_len - self.prev_len):
                if self.mode == '1D':
                    subarr_x = []
                    for j in range(self.prev_len):
                        subarr_x.extend(self.create_subarray(row, side_cols, i + j, 0))
                    subarr_y = self.create_subarray(row, side_cols, i + self.prev_len, 1)
                    subarr_x.append(subarr_y[0])
                    subarr_x.append(subarr_y[1])
                    self.x.append(subarr_x)
                    self.y.append(subarr_y)
                elif self.mode == 'RNN':
                    subarr_x = [self.create_subarray(row, side_cols, i + j, 0) for j in range(self.prev_len)]
                    subarr_y = self.create_subarray(row, side_cols, i + self.prev_len - 1, 1)
                    self.x.append(subarr_x)
                    self.y.append(subarr_y)



    def create_subarray(self, row, cols, index, istest):
        subarr = []
        for col in cols:
            subarr.append(row[col][index])
        if istest:
            subarr.append(row['Age'][index])
            subarr.append(row['Age_diff'][index])
            subarr.append(row['occur'][index])
        return subarr

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]