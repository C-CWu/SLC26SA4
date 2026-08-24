import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
import pandas as pd
import catboost as cb
import shap
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import (
    roc_auc_score, auc, precision_recall_curve, precision_score, recall_score,
    f1_score, accuracy_score, confusion_matrix, mean_absolute_error, roc_curve,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
from utils import *
from models import *

# Define dataset classes to guarantee 44 features as in kmeans.py
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

            'avg1',
            'avg2',
            'avg3',
            'avg4',
            'four_avg',

            'EVA',
            'VAsize',
            'mondini',
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
            subarr.append(row['Age'][index])
            subarr.append(row['Age_diff'][index])
            subarr.append(row['occur'][index])
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

            'avg1',
            'avg2',
            'avg3',
            'avg4',
            'four_avg',

            'EVA',
            'VAsize',
            'mondini',
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

def compute_avg_list(row):
    arrays = [np.array(row[col]) for col in ['AC025_trend', 'AC05_trend', 'AC1_trend', 'AC2_trend', 'AC4_trend', 'AC8_trend']]
    avg_list = np.mean(arrays, axis=0)
    return avg_list.tolist()

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

def fix_sz(X, A, Y, k = 4):
    new_x, new_age, new_y = [], [], []
    for xt, at, yt in zip(X, A, Y):
        if len(xt) >= k and len(at) >= k:
            new_x.append(xt[-k:])
            new_age.append(at[-k:])
            new_y.append(yt)
    return new_x, new_age, new_y

def predict_regression(model, data, device = 'cuda'):
    model.eval()
    age = data[:, -1, 0].float().to(device) + 0.5
    step = torch.tensor(np.full(data.shape[0], 0.5)).float().to(device)
    with torch.no_grad():
        outputs = model(data, age, step)
    return outputs

def predict_ae_latent(model, x, device='cuda'):
    model.eval()  
    with torch.no_grad():
        x = x.to(device)
        _, latent = model(x)
    return latent.cpu()

def train_ae_and_kmeans(train_data, device='cuda'):
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

def create_kmeans_for_fold(raw_data, kmeans, encoder, data_mean, data_std, device='cuda'):
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
            values = predict_ae_latent(encoder, values, device)
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

def prepare_data(dataset, id):
    sz = dataset.x.shape[0] 
    X = dataset.x.reshape(sz, -1)
    diff = dataset.y[:, 1]
    X = np.hstack((X, diff[:, None]))
    y = dataset.y[:, 4 + id]
    return X, y

def prepare_data_peak(dataset, id):
    sz = dataset.x.shape[0] 
    X = dataset.x.reshape(sz, -1)
    diff = np.full((len(X),), 0.5)
    X = np.hstack((X, diff[:, None]))
    return X

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

def save_plot_both_formats(filename, dpi=300):
    import os
    os.makedirs('figure', exist_ok=True)
    basename = os.path.basename(filename)
    root, _ = os.path.splitext(basename)
    filepath = os.path.join('figure', root)
    plt.savefig(filepath + ".tiff", dpi=dpi, format='tiff', pil_kwargs={"compression": "tiff_lzw"})
    plt.savefig(filepath + ".eps", dpi=dpi, format='eps')
    print(f"Saved {filepath}.tiff and {filepath}.eps")

def main():
    import matplotlib
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Arial']
    matplotlib.rcParams['font.size'] = 10
    matplotlib.rcParams['axes.labelsize'] = 10
    matplotlib.rcParams['axes.titlesize'] = 10
    matplotlib.rcParams['xtick.labelsize'] = 10
    matplotlib.rcParams['ytick.labelsize'] = 10
    matplotlib.rcParams['legend.fontsize'] = 10
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['pdf.fonttype'] = 42

    # === Configuration for X-axis Labels ===
    # You can modify these values to change the X-axis text shown on the plots
    xlabel_beeswarm = "SHAP value"
    xlabel_bar = "mean(|SHAP value|)"
    xlabel_beeswarm_grouped = "SHAP value"
    xlabel_bar_grouped = "mean(|SHAP value|)"

    # === Configuration for LOESS Calibration Plot ===
    # You can modify these values to change the text and font sizes shown on the LOESS calibration plot
    cal_loess_title = "LOESS Calibration Plot"
    cal_loess_xlabel = "Predicted Probability"
    cal_loess_ylabel = "Observed Probability"
    cal_loess_legend_perfect = "Perfectly calibrated"
    cal_loess_legend_curve = "LOESS Curve (Slope = {:.2f}, Intercept = {:.2f})"
    cal_loess_legend_fit = "Linear Fit (Slope = {:.2f}, Intercept = {:.2f})"
    cal_loess_legend_ci = "95% Confidence Interval"
    cal_loess_title_fontsize = 10
    cal_loess_label_fontsize = 10
    cal_loess_legend_fontsize = 10
    cal_loess_output_name = "calibration_plot_loess.png"
    cal_loess_n_bootstraps = 200  # Number of bootstrap resamples for CI estimation
    cal_loess_frac = 0.5          # Fraction of data to use when smoothing (between 0 and 1)
    cal_loess_plot_fit_line = True  # Set to True to draw the regression fit line on the plot, False otherwise

    # === Configuration for Precision-Recall Curve ===
    # You can modify these values to change the text and font sizes shown on the Precision-Recall curve
    pr_title = "Precision-Recall Curve"
    pr_xlabel = "Recall"
    pr_ylabel = "Precision"
    pr_legend_model = "CatBoost (PR AUC = {:.4f})"
    pr_legend_no_skill = "No Skill (PR AUC = {:.4f})"
    pr_title_fontsize = 10
    pr_label_fontsize = 10
    pr_legend_fontsize = 10
    pr_output_name = "precision_recall_curve.png"

    # === Configuration for Decision Curve Analysis ===
    # You can modify these values to change the text and font sizes shown on the Decision Curve Analysis plot
    dca_title = "Decision Curve Analysis"
    dca_xlabel = "Threshold Probability"
    dca_ylabel = "Net Benefit"
    dca_legend_model = "CatBoost"
    dca_legend_all = "Treat All"
    dca_legend_none = "Treat None"
    dca_title_fontsize = 10
    dca_label_fontsize = 10
    dca_legend_fontsize = 10
    dca_output_name = "decision_curve.png"

    print("Loading data...")
    df_s3 = load_s2_table('S3 Table.csv')
    df_new, raw_data = process_acoustic_data(df_s3)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Setting up cross-validation splits and training classifiers...")
    # Split using same function as in kmeans.py (val=False matches elasticnet_catboost_11)
    folder_data = k_folder_split_data(raw_data, 1, 5, False)
    
    all_shap_values = []
    all_test_features = []
    all_test_targets = []
    all_test_probs = []
    
    side_cols = [
        'Age', 'Age_diff', 'Genotype', 'Gender',
        'AC025_trend', 'AC05_trend', 'AC1_trend', 'AC2_trend', 'AC4_trend', 'AC8_trend',
        'AC025_trend_label', 'AC05_trend_label', 'AC1_trend_label', 'AC2_trend_label', 'AC4_trend_label', 'AC8_trend_label',
        'AC025_std_in_time', 'AC05_std_in_time', 'AC1_std_in_time', 'AC2_std_in_time', 'AC4_std_in_time', 'AC8_std_in_time',
        'avg_trend_labels_0', 'avg_trend_labels_1', 'avg_trend_labels_2', 'avg_trend_labels_3', 'avg_trend_labels_4',
        'avg1', 'avg2', 'avg3', 'avg4', 'four_avg',
        'EVA', 'VAsize', 'mondini', 'peak', 'peak_agg', 'peak_diff',
        'big_peak', 'big_peak_agg', 'big_peak_diff',
        'recently_peak', 'recently_big_peak', 'recently_total'
    ]
    
    feature_names = []
    for t in range(3):
        time_suffix = f" (t-{2-t})" if t < 2 else " (current t)"
        for col in side_cols:
            feature_names.append(f"{col}{time_suffix}")
    for col in ['AC025_trend', 'AC05_trend', 'AC1_trend', 'AC2_trend', 'AC4_trend', 'AC8_trend']:
        feature_names.append(f"pred_{col} (future)")

    for fold in range(len(folder_data)):
        print(f"--- Processing Fold {fold + 1}/5 ---")
        train_data, test_data = folder_data[fold]
        
        train_data = train_data.copy()
        test_data = test_data.copy()
        
        # Fit Autoencoder and KMeans fold-specifically
        encoder_fold, kmeans_fold, mean_fold, std_fold = train_ae_and_kmeans(train_data, device)
        create_kmeans_for_fold(train_data, kmeans_fold, encoder_fold, mean_fold, std_fold, device)
        create_kmeans_for_fold(test_data, kmeans_fold, encoder_fold, mean_fold, std_fold, device)
        
        train_dataset_mae = HearDataset_MAE(train_data, 3, 'RNN')
        # Train 6 ElasticNetCV models fold-specifically
        elasticnet_models = [ElasticNetCV(cv=5, max_iter=5000, tol=1e-3) for _ in range(6)]
        for id in range(6):
            X_train_mae, y_train_mae = prepare_data(train_dataset_mae, id)
            elasticnet_models[id].fit(X_train_mae, y_train_mae)
        
        train_dataset_peak = HearDataset_Peak(train_data, 3, 'RNN')
        test_dataset_peak = HearDataset_Peak(test_data, 3, 'RNN')
        
        X_train, y_train = train_dataset_peak[:][0], train_dataset_peak[:][1][:, -1]
        X_test, y_test = test_dataset_peak[:][0], test_dataset_peak[:][1][:, -1]
        
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
        
        y_train = np.nan_to_num(y_train).astype(int)
        y_test = np.nan_to_num(y_test).astype(int)
        
        # Append regression predictions from the 6 ElasticNet models
        for id in range(6):
            X_train_peak_id = prepare_data_peak(train_dataset_peak, id)
            y_pred_train = elasticnet_models[id].predict(X_train_peak_id)
            X_train = np.hstack((X_train, y_pred_train[:, None]))
            
            X_test_peak_id = prepare_data_peak(test_dataset_peak, id)
            y_pred_test = elasticnet_models[id].predict(X_test_peak_id)
            X_test = np.hstack((X_test, y_pred_test[:, None]))
        
        # Train CatBoostClassifier
        clf = cb.CatBoostClassifier(
            iterations=100,
            learning_rate=0.01,
            depth=6,
            silent=True
        )
        clf.fit(X_train, y_train)
        
        # Calculate predicted probabilities for class 1
        y_prob = clf.predict_proba(X_test)[:, 1]
        all_test_probs.append(y_prob)
        
        # Calculate SHAP values for test fold
        explainer = shap.TreeExplainer(clf)
        # Handle different SHAP output formats robustly
        shap_values_raw = explainer.shap_values(X_test)
        
        if isinstance(shap_values_raw, list):
            # For list outputs (class 0, class 1)
            shap_values_class1 = shap_values_raw[1] if len(shap_values_raw) == 2 else shap_values_raw[0]
        elif hasattr(shap_values_raw, 'values'):
            # For Explanation object
            if len(shap_values_raw.shape) == 3:
                shap_values_class1 = shap_values_raw.values[:, :, 1]
            else:
                shap_values_class1 = shap_values_raw.values
        else:
            # Array format
            if len(shap_values_raw.shape) == 3:
                shap_values_class1 = shap_values_raw[:, :, 1]
            else:
                shap_values_class1 = shap_values_raw
                
        all_shap_values.append(shap_values_class1)
        all_test_features.append(X_test)
        all_test_targets.append(y_test)

    # Combine out-of-fold results
    combined_shap_values = np.concatenate(all_shap_values, axis=0)
    combined_test_features = np.concatenate(all_test_features, axis=0)
    combined_test_targets = np.concatenate(all_test_targets, axis=0)
    combined_test_probs = np.concatenate(all_test_probs, axis=0)
    
    combined_test_df = pd.DataFrame(combined_test_features, columns=feature_names)
    
    print("\nGenerating and saving SHAP plots...")
    
    # 1. Summary Plot (Beeswarm)
    # 138 features need a very tall figure to be legible
    plt.figure(figsize=(3.5, 25.0))
    shap.summary_plot(combined_shap_values, combined_test_df, max_display=len(feature_names), show=False)
    plt.title("SHAP Beeswarm Plot (Out-of-Fold Test Data - All Features)", fontsize=10)
    plt.xlabel(xlabel_beeswarm)
    plt.tight_layout()
    save_plot_both_formats("shap_summary.png", dpi=300)
    plt.close()
    
    # 2. Bar Plot (Global Importance)
    plt.figure(figsize=(3.5, 25.0))
    shap.summary_plot(combined_shap_values, combined_test_df, plot_type="bar", max_display=len(feature_names), show=False)
    plt.title("SHAP Global Feature Importance (Bar Plot - All Features)", fontsize=10)
    plt.xlabel(xlabel_bar)
    plt.tight_layout()
    save_plot_both_formats("shap_bar.png", dpi=300)
    plt.close()
    
    # Calculate feature importances
    mean_abs_shap = np.mean(np.abs(combined_shap_values), axis=0)
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean Abs SHAP': mean_abs_shap
    }).sort_values(by='Mean Abs SHAP', ascending=False).reset_index(drop=True)
    
    print("\n" + "="*70)
    print(f"{'All Features by SHAP Importance (Mean Absolute SHAP Value)':^70}")
    print("="*70)
    print(importance_df.to_string(index=True))
    print("="*70)

    # === Grouped SHAP Analysis (Aggregating Similar Features) ===
    print("\nPerforming Grouped SHAP Analysis...")
    
    def get_group_name(feat_name):
        if "pred_" in feat_name:
            return "Model-Predicted Hearing Trend (future)"
        
        base_name = feat_name.split(" (")[0]
        
        if base_name == "Age":
            return "Age"
        elif base_name == "Age_diff":
            return "Age Difference (Age_diff)"
        elif base_name == "Genotype":
            return "Genotype"
        elif base_name == "Gender":
            return "Gender"
        elif base_name in ["EVA", "VAsize"]:
            return "EVA & VA Size"
        elif base_name == "mondini":
            return "Mondini Dysplasia"
        elif "std_in_time" in base_name:
            return "Hearing Volatility (_std_in_time)"
        elif "trend_label" in base_name:
            return "Hearing Trend Categories (_trend_label)"
        elif "trend" in base_name: # must be after trend_label and pred_
            return "Hearing Threshold Trend (_trend)"
        elif "avg_trend_labels" in base_name:
            return "KMeans Acoustic Trend Labels"
        elif base_name in ["avg1", "avg2", "avg3", "avg4", "four_avg"]:
            return "Multi-frequency Average Thresholds"
        elif base_name == "peak":
            return "Peak Occurrence History"
        elif base_name == "peak_agg":
            return "Cumulative Peak Count"
        elif base_name == "peak_diff":
            return "Time Since Last Peak"
        elif base_name == "big_peak":
            return "Big Peak Occurrence History"
        elif base_name == "big_peak_agg":
            return "Cumulative Big Peak Count"
        elif base_name == "big_peak_diff":
            return "Time Since Last Big Peak"
        elif base_name == "recently_peak":
            return "Recent Peaks Count"
        elif base_name == "recently_big_peak":
            return "Recent Big Peaks Count"
        elif base_name == "recently_total":
            return "Total Recent Peaks Count"
        
        return "Other Features"

    # Scale feature values for meaningful color representations in Beeswarm
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(combined_test_df)
    scaled_df = pd.DataFrame(scaled_features, columns=combined_test_df.columns)
    
    # Map groups to feature list
    group_to_feats = {}
    for feat in feature_names:
        grp = get_group_name(feat)
        if grp not in group_to_feats:
            group_to_feats[grp] = []
        group_to_feats[grp].append(feat)
        
    # Compute grouped SHAP and feature values
    grouped_shap_list = []
    grouped_feat_list = []
    grouped_names = list(group_to_feats.keys())
    
    for grp in grouped_names:
        feats_in_grp = group_to_feats[grp]
        
        # Sum SHAP values
        indices = [combined_test_df.columns.get_loc(f) for f in feats_in_grp]
        grp_shap_values = combined_shap_values[:, indices].sum(axis=1)
        grouped_shap_list.append(grp_shap_values)
        
        # Average scaled features
        grp_feat_values = scaled_df[feats_in_grp].mean(axis=1)
        grouped_feat_list.append(grp_feat_values)
        
    grouped_shap = np.column_stack(grouped_shap_list)
    grouped_feat = np.column_stack(grouped_feat_list)
    grouped_df = pd.DataFrame(grouped_feat, columns=grouped_names)
    
    # Save grouped plots
    # Beeswarm grouped summary plot
    plt.figure(figsize=(3.5, 3.0))
    shap.summary_plot(grouped_shap, grouped_df, show=False)
    plt.title("SHAP Beeswarm Plot", fontsize=10)
    plt.xlabel(xlabel_beeswarm_grouped)
    plt.tight_layout()
    save_plot_both_formats("shap_summary_grouped.png", dpi=300)
    plt.close()
    
    # Bar grouped global importance plot
    plt.figure(figsize=(3.5, 3.0))
    shap.summary_plot(grouped_shap, grouped_df, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance", fontsize=10)
    plt.xlabel(xlabel_bar_grouped)
    plt.tight_layout()
    save_plot_both_formats("shap_bar_grouped.png", dpi=300)
    plt.close()
    
    # Generate grouped importance table
    grouped_mean_abs_shap = np.mean(np.abs(grouped_shap), axis=0)
    grouped_importance_df = pd.DataFrame({
        'Feature Group': grouped_names,
        'Mean Abs SHAP': grouped_mean_abs_shap
    }).sort_values(by='Mean Abs SHAP', ascending=False).reset_index(drop=True)
    
    print("\n" + "="*70)
    print(f"{'Grouped Feature Importance by SHAP (Mean Absolute SHAP)':^70}")
    print("="*70)
    print(grouped_importance_df.to_string(index=True))
    print("="*70)

    # === LOESS Calibration Plot ===
    print("\nGenerating and saving LOESS Calibration Plot...")
    try:
        import statsmodels.api as sm
        from scipy.interpolate import interp1d
        
        # Grid to evaluate LOESS
        grid = np.linspace(np.min(combined_test_probs), np.max(combined_test_probs), 100)
        
        # Fit LOESS on original data
        loess_fit = sm.nonparametric.lowess(combined_test_targets.astype(float), combined_test_probs, frac=cal_loess_frac, it=0)
        # Sort by predicted probability
        xs = loess_fit[:, 0]
        ys = loess_fit[:, 1]
        
        print("DEBUG - LOESS Target counts:", np.unique(combined_test_targets, return_counts=True))
        print("DEBUG - LOESS Probabilities range:", np.min(combined_test_probs), np.max(combined_test_probs))
        print("DEBUG - LOESS Fitted values (first 10):", ys[:10])
        print("DEBUG - LOESS Fitted values min/max/mean:", np.min(ys), np.max(ys), np.mean(ys))
        
        # Deduplicate xs for interpolation to prevent division by zero in interp1d
        df_loess = pd.DataFrame({'x': xs, 'y': ys}).groupby('x').mean().reset_index()
        xs_unique = df_loess['x'].values
        ys_unique = df_loess['y'].values
        
        # Interpolate fitted curve to grid
        interp_func = interp1d(xs_unique, ys_unique, bounds_error=False, fill_value="extrapolate")
        y_fit_grid = interp_func(grid)
        
        # Bootstrapping for confidence intervals
        bootstrap_y = []
        n_samples = len(combined_test_probs)
        
        for _ in range(cal_loess_n_bootstraps):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            y_true_boot = combined_test_targets[indices]
            y_prob_boot = combined_test_probs[indices]
            
            try:
                boot_fit = sm.nonparametric.lowess(y_true_boot.astype(float), y_prob_boot, frac=cal_loess_frac, it=0)
                boot_xs = boot_fit[:, 0]
                boot_ys = boot_fit[:, 1]
                
                # Deduplicate for bootstrap sample
                df_boot = pd.DataFrame({'x': boot_xs, 'y': boot_ys}).groupby('x').mean().reset_index()
                boot_xs_unique = df_boot['x'].values
                boot_ys_unique = df_boot['y'].values
                
                boot_interp = interp1d(boot_xs_unique, boot_ys_unique, bounds_error=False, fill_value="extrapolate")
                bootstrap_y.append(boot_interp(grid))
            except Exception:
                continue
                
        bootstrap_y = np.array(bootstrap_y)
        lower_bound = np.percentile(bootstrap_y, 2.5, axis=0)
        upper_bound = np.percentile(bootstrap_y, 97.5, axis=0)
        
        # Calculate slope and intercept for LOESS curve (on the plotted grid points)
        slope_loess, intercept_loess = np.polyfit(grid, y_fit_grid, 1)
        
        # Format the LOESS legend string robustly
        try:
            loess_label = cal_loess_legend_curve.format(slope_loess, intercept_loess)
        except Exception:
            loess_label = cal_loess_legend_curve
            
        # Plotting
        plt.figure(figsize=(3.5, 3.5))
        
        # Plot perfectly calibrated reference line
        plt.plot([0, 1], [0, 1], "k--", label=cal_loess_legend_perfect)
        
        # Plot LOESS curve
        plt.plot(grid, y_fit_grid, "b-", label=loess_label)
        
        # Plot CI
        plt.fill_between(grid, lower_bound, upper_bound, color="blue", alpha=0.15, label=cal_loess_legend_ci)
        
        # Plot linear fit line
        if cal_loess_plot_fit_line:
            fit_line_x = np.linspace(0, 1, 100)
            fit_line_y = slope_loess * fit_line_x + intercept_loess
            try:
                fit_label = cal_loess_legend_fit.format(slope_loess, intercept_loess)
            except Exception:
                fit_label = cal_loess_legend_fit
            plt.plot(fit_line_x, fit_line_y, "r:", label=fit_label)
            
        plt.xlabel(cal_loess_xlabel, fontsize=cal_loess_label_fontsize)
        plt.ylabel(cal_loess_ylabel, fontsize=cal_loess_label_fontsize)
        plt.title(cal_loess_title, fontsize=cal_loess_title_fontsize)
        plt.ylim([-0.05, 1.05])
        plt.xlim([-0.05, 1.05])
        plt.legend(loc="lower right", fontsize=cal_loess_legend_fontsize)
        plt.grid(True)
        plt.tight_layout()
        save_plot_both_formats(cal_loess_output_name, dpi=300)
        plt.close()
    except ImportError:
        print("\nWarning: statsmodels is not installed. Skipping LOESS Calibration Plot.")
        print("Please install statsmodels by running: pip install statsmodels")

    # === Precision-Recall AUC Plot ===
    print("\nGenerating and saving Precision-Recall AUC Plot...")
    try:
        # Calculate precision-recall curve
        precision_vals, recall_vals, _ = precision_recall_curve(combined_test_targets, combined_test_probs)
        # Calculate PR AUC (using recall reversed to be monotonically increasing for auc())
        pr_auc = auc(recall_vals[::-1], precision_vals[::-1])
        
        # Calculate no-skill baseline (fraction of positives)
        no_skill = np.sum(combined_test_targets) / len(combined_test_targets)
        
        plt.figure(figsize=(3.5, 3.5))
        
        # Plot no-skill baseline
        plt.plot([0, 1], [no_skill, no_skill], "k--", label=pr_legend_no_skill.format(no_skill))
        
        # Plot model's Precision-Recall curve
        plt.plot(recall_vals, precision_vals, "b-", label=pr_legend_model.format(pr_auc))
        
        plt.xlabel(pr_xlabel, fontsize=pr_label_fontsize)
        plt.ylabel(pr_ylabel, fontsize=pr_label_fontsize)
        plt.title(pr_title, fontsize=pr_title_fontsize)
        plt.ylim([-0.05, 1.05])
        plt.xlim([-0.05, 1.05])
        plt.legend(loc="lower left", fontsize=pr_legend_fontsize)
        plt.grid(True)
        plt.tight_layout()
        save_plot_both_formats(pr_output_name, dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error generating Precision-Recall plot: {e}")

    # === Decision Curve Analysis Plot ===
    print("\nGenerating and saving Decision Curve Analysis Plot...")
    try:
        thresholds = np.linspace(0.0, 0.99, 100)
        net_benefit_model = []
        net_benefit_all = []
        
        n_samples = len(combined_test_targets)
        n_positives = np.sum(combined_test_targets == 1)
        n_negatives = np.sum(combined_test_targets == 0)
        
        for p_t in thresholds:
            # Net benefit for model
            tp_model = np.sum((combined_test_probs >= p_t) & (combined_test_targets == 1))
            fp_model = np.sum((combined_test_probs >= p_t) & (combined_test_targets == 0))
            nb_model = (tp_model - fp_model * (p_t / (1.0 - p_t))) / n_samples
            net_benefit_model.append(nb_model)
            
            # Net benefit for "Treat All"
            nb_all = (n_positives - n_negatives * (p_t / (1.0 - p_t))) / n_samples
            net_benefit_all.append(nb_all)
            
        net_benefit_none = np.zeros_like(thresholds)
        
        plt.figure(figsize=(3.5, 3.5))
        
        # Plot "Treat None"
        plt.plot(thresholds, net_benefit_none, "k-", label=dca_legend_none)
        
        # Plot "Treat All"
        plt.plot(thresholds, net_benefit_all, "grey", linestyle="--", label=dca_legend_all)
        
        # Plot model net benefit
        plt.plot(thresholds, net_benefit_model, "b-", linewidth=2, label=dca_legend_model)
        
        # Y-limit: start from -0.05 to avoid extremely negative values at thresholds close to 1
        # Capping the max limit slightly above the prevalence
        prevalence = n_positives / n_samples
        ymax = max(max(net_benefit_model), prevalence) * 1.1
        
        plt.xlabel(dca_xlabel, fontsize=dca_label_fontsize)
        plt.ylabel(dca_ylabel, fontsize=dca_label_fontsize)
        plt.title(dca_title, fontsize=dca_title_fontsize)
        plt.ylim([-0.05, ymax])
        plt.xlim([-0.05, 1.05])
        plt.legend(loc="upper right", fontsize=dca_legend_fontsize)
        plt.grid(True)
        plt.tight_layout()
        save_plot_both_formats(dca_output_name, dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error generating Decision Curve Analysis plot: {e}")

if __name__ == '__main__':
    main()
