from utils import *
from dataset import *
from models import *
from sklearn.metrics import (
    roc_auc_score, auc, precision_recall_curve, precision_score, recall_score,
    f1_score, accuracy_score, confusion_matrix, mean_absolute_error, roc_curve
)
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import catboost as cb
from sklearn.linear_model import ElasticNetCV
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import sys

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

import os
os.makedirs('log', exist_ok=True)
sys.stdout = Tee(sys.stdout, open('log/kmeans_ablation_study.log', 'w', encoding='utf-8'))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

df_s2 = load_s2_table('S2 Table.csv')
df_new, raw_data = process_acoustic_data(df_s2)

def create_XY(filtered_df):
    ages = filtered_df['Age_diff'].tolist()
    labels = filtered_df['occur'].tolist()
    ac = []
    for _, row in filtered_df.iterrows():
        ac_sublist = []
        for i, col in enumerate(['AC025_trend', 'AC05_trend', 'AC1_trend', 'AC2_trend', 'AC4_trend', 'AC8_trend']):
            ac_sublist.append(row[col])
        ac.append(np.array(ac_sublist).T.tolist())

    def generate_subarrays(arr):
        result = []
        for i in range(1, len(arr) + 1):
            result.append(arr[:i])
        return result

    ac = [generate_subarrays(x) for x in ac]
    ac = [y for x in ac for y in x]

    ages = [generate_subarrays(x) for x in ages]
    ages = [y for x in ages for y in x]

    labels = [y for x in labels for y in x]
    return ac, ages, labels

def fill_slot(X, Y, diff = 0.5):
    Y_new = []
    X_array = np.array(X)
    Y_array = np.array(Y)

    complete_list = np.arange(min(X), max(X), diff)
    missing_values = [value for value in complete_list if value not in X]
    X_new = sorted(X + missing_values)
    for i in range(6):
        Y_new.append(np.interp(X_new, X_array, Y_array[:, i]))
    return np.array(X_new), np.array(Y_new).T

def interpolate_data(values, ages, diff = 0.5):
    interpolated_ages = []
    interpolated_values = []
    for x, y in zip(ages, values):
        age, new_x = fill_slot(x, y, diff)
        interpolated_ages.append(age)
        interpolated_values.append(new_x)
    return interpolated_values, interpolated_ages

def concate(X, Y):
    new_x = []
    for x, y in zip(X, Y):
        y_reshaped = y.reshape(-1, 1)
        result = np.concatenate((x, y_reshaped), axis=1)
        new_x.append(result)
    return new_x

def fix_sz(X, A, Y, k = 4):
    new_x, new_age, new_y = [], [], []
    for xt, at, yt in zip(X, A, Y):
        if len(xt) >= k and len(at) >= k:
            new_x.append(xt[-k:])
            new_age.append(at[-k:])
            new_y.append(yt)
    return new_x, new_age, new_y

kx, k_age, ky = create_XY(raw_data)
kx, k_age, ky = fix_sz(kx, k_age, ky, 4)
kx = [np.array(x) for x in kx]
k_age = np.stack([np.array(x) for x in k_age])

kx = [x.mean(axis=1, keepdims=True) for x in kx]
kx = np.stack(kx)
n_samples = len(kx)
kx = np.array(kx).reshape(n_samples, -1)  
kx = np.concatenate([kx, k_age], axis=1)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(8, 4),
            nn.Mish(),
            nn.Linear(4, 2)  
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 4),
            nn.Mish(),
            nn.Linear(4, 8),
        )

    def forward(self, x):
        latent = self.encoder(x)
        x = self.decoder(latent)
        return x, latent

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict_ae_latent(model, x):
    model.eval()  
    with torch.no_grad():
        x = x.to(device)
        _, latent = model(x)
    return latent.cpu()

def train_ae_and_kmeans(train_data, device):
    kx_train, k_age_train, ky_train = create_XY(train_data)
    kx_train, k_age_train, ky_train = fix_sz(kx_train, k_age_train, ky_train, 4)
    kx_train = [np.array(x) for x in kx_train]
    k_age_train = np.stack([np.array(x) for x in k_age_train])
    kx_train = [x.mean(axis=1, keepdims=True) for x in kx_train]
    kx_train = np.stack(kx_train)
    n_samples = len(kx_train)
    kx_train = np.array(kx_train).reshape(n_samples, -1)
    kx_train = np.concatenate([kx_train, k_age_train], axis=1)

    X_train_flattened = torch.tensor(kx_train).float()
    data_mean = X_train_flattened.mean(dim=0, keepdim=True)
    data_std = X_train_flattened.std(dim=0, keepdim=True)
    data_std[data_std == 0.0] = 1.0
    X_train_normalized = (X_train_flattened - data_mean) / data_std

    dataset = TensorDataset(X_train_normalized, X_train_normalized)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    encoder = Autoencoder().to(device)
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(encoder.parameters(), lr=0.001)

    num_epochs = 1000
    for epoch in range(num_epochs):
        for batch_data, _ in train_loader:
            batch_data = batch_data.to(device)
            output, _ = encoder(batch_data)
            loss = criterion(output, batch_data)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if loss.item() < 0.061:
            break

    encoder.eval()
    with torch.no_grad():
        _, latent_train = encoder(X_train_normalized.to(device))
        latent_train = latent_train.cpu().numpy()

    kmeans = KMeans(n_clusters=5, random_state=0)
    kmeans.fit(latent_train)

    return encoder, kmeans, data_mean, data_std

def compute_avg_list(row):
    arrays = [np.array(row[col]) for col in ['AC025_trend', 'AC05_trend', 'AC1_trend', 'AC2_trend', 'AC4_trend', 'AC8_trend']]
    avg_list = np.mean(arrays, axis=0)
    return avg_list.tolist()

def create_kmeans_for_fold(raw_data, kmeans, encoder, data_mean, data_std):
    raw_data['avg_trend'] = raw_data.apply(compute_avg_list, axis=1)

    for i in range(5):
        raw_data[f'avg_trend_labels_{i}'] = [[] for _ in range(len(raw_data))]

    for idx, row in raw_data.iterrows():
        avg_trend = row['avg_trend']
        age_list = row['Age_diff']
        age = np.array(row['Age'])
        trend_labels = [[0 for _ in range(len(avg_trend))] for _ in range(5)]
        
        for i in range(4, len(avg_trend)):
            values = torch.tensor([avg_trend[i - 4:i]  + age_list[i - 4:i]]).float()

            values = (values - data_mean) / data_std
            values = predict_ae_latent(encoder, values)
            k_label = kmeans.predict(values.numpy().astype(kmeans.cluster_centers_.dtype))[0]
            trend_labels[k_label][i - 1] += 1
        
        for i, trend_labels_list in enumerate(trend_labels):
            assert len(trend_labels_list) == len(age)
            sum_valid_elements = np.zeros_like(trend_labels_list)
            for j in range(len(trend_labels_list)):
                valid_indices = np.where((age >= age[j] - 1.5) & (age <= age[j]))[0]
                sum_valid_elements[j] = sum([trend_labels_list[idx] for idx in valid_indices])
            trend_labels[i] = sum_valid_elements
        
        for i in range(5):
            raw_data.at[idx, f'avg_trend_labels_{i}'] = trend_labels[i]

# Custom Dataset classes supporting parameterizable extra features
class HearDataset_MAE(Dataset):
    def __init__(self, df, prev_len, mode, extra_features):
        self.x, self.y = [], []
        self.prev_len = prev_len
        self.mode = mode
        self.extra_features = extra_features
        self.process_data(df)
        self.x = np.array(self.x)
        self.y = np.array(self.y)

    def process_data(self, df):
        side_cols = [
            'Age',
            'Age_diff',
        ] + self.extra_features + [
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

            'avg1',
            'avg2',
            'avg3',
            'avg4',
            'four_avg',

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
                if self.mode == 'RNN':
                    subarr_x = [self.create_subarray(row, side_cols, i + j, 0) for j in range(self.prev_len)]
                    subarr_y = self.create_subarray(row, side_cols, i + self.prev_len, 1)
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

class HearDataset_Peak(Dataset):
    def __init__(self, df, prev_len, mode, extra_features):
        self.x, self.y = [], []
        self.prev_len = prev_len
        self.mode = mode
        self.extra_features = extra_features
        self.process_data(df)
        self.x = np.array(self.x)
        self.y = np.array(self.y)

    def process_data(self, df):
        side_cols = [
            'Age',
            'Age_diff',
        ] + self.extra_features + [
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

            'avg1',
            'avg2',
            'avg3',
            'avg4',
            'four_avg',

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
            for i in range(data_len - self.prev_len + 1):
                if self.mode == 'RNN':
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

def prepare_data(dataset, id):
    sz = dataset.x.shape[0] 
    X = dataset.x.reshape(sz, -1)
    diff = dataset.y[:, 1]
    X = np.hstack((X, diff[:, None]))
    ac_offset = 2 + len(dataset.extra_features)
    y = dataset.y[:, ac_offset + id]
    return X, y

def prepare_data_peak(dataset, id):
    sz = dataset.x.shape[0] 
    X = dataset.x.reshape(sz, -1)
    diff = np.full((len(X),), 0.5)
    X = np.hstack((X, diff[:, None]))
    return X

# Ablation study configurations
ablation_combinations = {
    "1. No extra features": [],

    "2. Only Genotype": ["Genotype"],
    "3. Only Gender": ["Gender"],
    "4. Only EVA": ["EVA"],
    "5. Only VAsize": ["VAsize"],
    "6. Only mondini": ["mondini"],

    "7. Genotype and Gender": ["Genotype", "Gender"],
    "8. Genotype and EVA": ["Genotype", "EVA"],
    "9. Genotype and VAsize": ["Genotype", "VAsize"],
    "10. Genotype and mondini": ["Genotype", "mondini"],
    "11. Gender and EVA": ["Gender", "EVA"],
    "12. Gender and VAsize": ["Gender", "VAsize"],
    "13. Gender and mondini": ["Gender", "mondini"],
    "14. EVA and VAsize": ["EVA", "VAsize"],
    "15. EVA and mondini": ["EVA", "mondini"],
    "16. VAsize and mondini": ["VAsize", "mondini"],

    "17. Genotype, Gender, EVA": ["Genotype", "Gender", "EVA"],
    "18. Genotype, Gender, VAsize": ["Genotype", "Gender", "VAsize"],
    "19. Genotype, Gender, mondini": ["Genotype", "Gender", "mondini"],
    "20. Genotype, EVA, VAsize": ["Genotype", "EVA", "VAsize"],
    "21. Genotype, EVA, mondini": ["Genotype", "EVA", "mondini"],
    "22. Genotype, VAsize, mondini": ["Genotype", "VAsize", "mondini"],
    "23. Gender, EVA, VAsize": ["Gender", "EVA", "VAsize"],
    "24. Gender, EVA, mondini": ["Gender", "EVA", "mondini"],
    "25. Gender, VAsize, mondini": ["Gender", "VAsize", "mondini"],
    "26. EVA, VAsize, mondini": ["EVA", "VAsize", "mondini"],

    "27. No Genotype": ["Gender", "EVA", "VAsize", "mondini"],
    "28. No Gender": ["Genotype", "EVA", "VAsize", "mondini"],
    "29. No EVA": ["Genotype", "Gender", "VAsize", "mondini"],
    "30. No VAsize": ["Genotype", "Gender", "EVA", "mondini"],
    "31. No mondini": ["Genotype", "Gender", "EVA", "VAsize"],

    "32. All extra features": ["Genotype", "Gender", "EVA", "VAsize", "mondini"]
} 

results = {}

# Precompute fold-specific processed data to prevent unsupervised data leakage
# without repeating Autoencoder and K-means training 32 times.
print("Precomputing fold-specific Autoencoder and K-means features...")
folder_data_raw = k_folder_split_data(raw_data, 1, 5, val=False)
fold_processed_data = []

for round_num in range(5):
    print(f"  Training representation models on fold {round_num + 1}/5...")
    train_data, test_data = folder_data_raw[round_num]
    
    # Copy dataframes to avoid in-place modification warnings
    train_data = train_data.copy()
    test_data = test_data.copy()
    
    # Fit AE and K-means strictly on train split
    encoder_fold, kmeans_fold, mean_fold, std_fold = train_ae_and_kmeans(train_data, device)
    
    # Generate K-means features
    create_kmeans_for_fold(train_data, kmeans_fold, encoder_fold, mean_fold, std_fold)
    create_kmeans_for_fold(test_data, kmeans_fold, encoder_fold, mean_fold, std_fold)
    
    fold_processed_data.append((train_data, test_data))

for name, extra_features in ablation_combinations.items():
    print(f"\nRunning Experiment: {name}")
    print(f"Features: {extra_features}")

    mae_list = []
    auc_list = []
    pr_auc_list = []
    f1_list = []
    precision_list = []
    recall_list = []
    spec_list = []
    bal_acc_list = []
    acc_list = []
    optimal_threshold_list = []

    # 5-Fold cross-validation
    for round_num in range(5):
        train_data, test_data = fold_processed_data[round_num]

        train_dataset_mae = HearDataset_MAE(train_data, 3, 'RNN', extra_features)
        test_dataset_mae = HearDataset_MAE(test_data, 3, 'RNN', extra_features)

        # Train ElasticNet for predicting hearing thresholds
        models = [ElasticNetCV(cv=5, max_iter=5000, tol=1e-3) for _ in range(6)]
        for id in range(6):
            X_train_mae, y_train_mae = prepare_data(train_dataset_mae, id)
            models[id].fit(X_train_mae, y_train_mae)

        # Evaluate Regression MAE on test dataset
        fold_maes = []
        for id in range(6):
            X_test_mae, y_test_mae = prepare_data(test_dataset_mae, id)
            y_pred_mae = models[id].predict(X_test_mae)
            fold_maes.append(mean_absolute_error(y_test_mae, y_pred_mae))
        mae_list.append(fold_maes)

        # Train CatBoostClassifier for predicting peak occurrences
        train_dataset_peak = HearDataset_Peak(train_data, 3, 'RNN', extra_features)
        test_dataset_peak = HearDataset_Peak(test_data, 3, 'RNN', extra_features)

        X_train, y_train = train_dataset_peak[:][0], train_dataset_peak[:][1][:, -1]
        X_test, y_test = test_dataset_peak[:][0], test_dataset_peak[:][1][:, -1]

        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

        y_train = np.nan_to_num(y_train).astype(int)
        y_test = np.nan_to_num(y_test).astype(int)

        # Append predictions from regression models
        for id in range(6):
            X_train_peak = prepare_data_peak(train_dataset_peak, id)
            y_pred = models[id].predict(X_train_peak)
            X_train = np.hstack((X_train, y_pred[:, None]))

            X_test_peak = prepare_data_peak(test_dataset_peak, id)
            y_pred = models[id].predict(X_test_peak)
            X_test = np.hstack((X_test, y_pred[:, None]))

        clf = cb.CatBoostClassifier(
            iterations=100,
            learning_rate=0.01,
            eval_metric='AUC',
            silent=True
        )
        clf.fit(X_train, y_train)

        y_scores_train = clf.predict_proba(X_train)[:, 1]
        y_scores_test = clf.predict_proba(X_test)[:, 1]

        # Calculate optimal threshold using training ROC curve
        fpr, tpr, thresholds = roc_curve(y_train, y_scores_train)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        # Calculate metrics on test data
        y_pred_test = (y_scores_test >= optimal_threshold).astype(int)

        precision = precision_score(y_test, y_pred_test, zero_division=0)
        recall = recall_score(y_test, y_pred_test, zero_division=0)
        f1 = f1_score(y_test, y_pred_test, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_scores_test)
        accuracy = accuracy_score(y_test, y_pred_test)

        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_scores_test)
        pr_auc = auc(recall_vals[::-1], precision_vals[::-1])

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        balanced_accuracy = (recall + specificity) / 2

        auc_list.append(roc_auc)
        pr_auc_list.append(pr_auc)
        f1_list.append(f1)
        precision_list.append(precision)
        recall_list.append(recall)
        spec_list.append(specificity)
        bal_acc_list.append(balanced_accuracy)
        acc_list.append(accuracy)
        optimal_threshold_list.append(optimal_threshold)

    mean_maes = np.mean(mae_list, axis=0)
    overall_mean_mae = np.mean(mean_maes)

    results[name] = {
        "overall_mean_mae": overall_mean_mae,
        "auc": np.mean(auc_list),
        "pr_auc": np.mean(pr_auc_list),
        "f1": np.mean(f1_list),
        "precision": np.mean(precision_list),
        "recall": np.mean(recall_list),
        "specificity": np.mean(spec_list),
        "balanced_accuracy": np.mean(bal_acc_list),
        "accuracy": np.mean(acc_list),
        "optimal_threshold": np.mean(optimal_threshold_list)
    }

    print(f"Result for {name}:")
    print(f"  Overall Regression MAE: {overall_mean_mae:.4f}")
    print(f"  Classification AUC: {np.mean(auc_list):.4f}")
    print(f"  Classification F1: {np.mean(f1_list):.4f}")

# Print summary table at the end
print("\n" + "="*80)
print(f"{'Ablation Study Summary (Average across 5 Folds)':^80}")
print("="*80)
print(f"{'Configuration':<25} | {'Reg. MAE':<9} | {'AUC':<6} | {'PR-AUC':<6} | {'F1':<6} | {'Recall':<6} | {'Spec.':<6} | {'Acc':<6} | {'Precision':<6} | {'optimal_threshold':<6}")
print("-"*80)
for name, metrics in results.items():
    print(f"{name:<25} | {metrics['overall_mean_mae']:<9.4f} | {metrics['auc']:<6.4f} | {metrics['pr_auc']:<6.4f} | {metrics['f1']:<6.4f} | {metrics['recall']:<6.4f} | {metrics['specificity']:<6.4f} | {metrics['accuracy']:<6.4f} | {metrics['precision']:<6.4f} | {metrics['optimal_threshold']:<6.4f}")
print("="*80)
