import os
import sys
import random
import pickle
import warnings
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from sklearn.metrics import (
    roc_auc_score, auc, precision_recall_curve, precision_score, recall_score,
    f1_score, accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error, roc_curve
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import ElasticNetCV
import catboost as cb

from utils import load_s2_table, process_acoustic_data, k_folder_split_data, train_epoch, evaluate, predict
from dataset import SEEDS
from models import Transformer_peaks_Model, MLP_peaks_Model, AModel, LSTM_peaks_Model

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Matplotlib global settings for high-quality publication figures
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 10,
    'axes.labelsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'ps.fonttype': 42,
    'pdf.fonttype': 42
})

# Enable interactive non-blocking plotting mode
plt.ion()

# Tee class for dual stdout logging
class Tee(object):
    """
    Dual-output stream handler to write stdout simultaneously to console and a log file.
    """
    def __init__(self, *files):
        """
        Initialize the Tee instance with output stream targets.
        :param files: Target file objects to receive stdout stream.
        """
        self.files = files

    def write(self, obj):
        """
        Write content object to all registered output file streams and flush immediately.
        :param obj: Content string to write.
        """
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        """
        Flush all registered file streams.
        """
        for f in self.files:
            f.flush()

os.makedirs('log', exist_ok=True)
_log_file = open('log/kmeans.log', 'w', encoding='utf-8')
sys.stdout = Tee(sys.stdout, _log_file)
#sys.stderr = Tee(sys.stderr, _log_file)

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.backends.backend_ps').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('fontTools').setLevel(logging.WARNING)

# Utility artifact saving functions
def save_model_artifact(obj, filepath):
    """
    Save a general Python object (e.g., scikit-learn model, CatBoost model, dictionary) as a pickle file.
    :param obj: Python object to pickle.
    :param filepath: Target file path (.pkl).
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

def save_pytorch_model(model, filepath):
    """
    Save PyTorch model state dictionary to the specified file path.
    :param model: PyTorch nn.Module instance.
    :param filepath: Target file path (.pth).
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)

def save_numpy_array(arr, filepath):
    """
    Save a NumPy array (e.g., feature scaling mean and standard deviation) to a binary file (.npy).
    :param arr: NumPy array to save.
    :param filepath: Target file path (.npy).
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.save(filepath, arr)

# Data preprocessing helper functions
def create_XY(filtered_df):
    """
    Extract sliding window feature sequences of acoustic trends and age differences from patient DataFrame.
    :param filtered_df: Input DataFrame containing acoustic trends and age differences.
    :return: Tuple of (ac, ages, labels) containing acoustic trend prefixes, age prefixes, and occurrence labels.
    """
    ages = filtered_df['Age_diff'].tolist()
    labels = filtered_df['occur'].tolist()
    ac = []
    for _, row in filtered_df.iterrows():
        ac_sublist = []
        for i, col in enumerate(['AC025_trend', 'AC05_trend', 'AC1_trend', 'AC2_trend', 'AC4_trend', 'AC8_trend']):
            ac_sublist.append(row[col])
        ac.append(np.array(ac_sublist).T.tolist())

    def generate_subarrays(arr):
        """
        Helper function to generate all prefix sub-arrays from length 1 to full length.
        :param arr: Input sequence array.
        :return: List of prefix sub-arrays.
        """
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
    """
    Truncate time-series feature sequences to retain only the last k time points for sequences with length >= k.
    :param X: List of acoustic feature sequences.
    :param A: List of age difference sequences.
    :param Y: List of occurrence labels.
    :param k: Number of last time steps to extract (default is 4).
    :return: Tuple of (new_x, new_age, new_y) containing truncated features, age sequences, and labels.
    """
    new_x, new_age, new_y = [], [], []
    for xt, at, yt in zip(X, A, Y):
        if len(xt) >= k and len(at) >= k:
            new_x.append(xt[-k:])
            new_age.append(at[-k:])
            new_y.append(yt)
    return new_x, new_age, new_y

def compute_avg_list(row):
    """
    Compute average acoustic hearing trend sequence across 6 audiometric frequency channels (0.25k to 8kHz).
    :param row: Single Pandas DataFrame row representing a patient record.
    :return: List of mean acoustic trend values across frequencies.
    """
    arrays = [np.array(row[col]) for col in ['AC025_trend', 'AC05_trend', 'AC1_trend', 'AC2_trend', 'AC4_trend', 'AC8_trend']]
    avg_list = np.mean(arrays, axis=0)
    return avg_list.tolist()

# Neural Network Architecture for Autoencoder
class Autoencoder(nn.Module):
    """
    Multi-Layer Perceptron (MLP) Autoencoder architecture:
    Compresses 8-dimensional hearing and age features down to a 2-dimensional latent representation,
    and reconstructs back to 8-dimensional input space.
    """
    def __init__(self):
        """Initialize Encoder and Decoder neural network modules."""
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
        """
        Forward pass through encoder and decoder networks.
        :param x: Input feature tensor of shape (batch_size, 8).
        :return: Tuple of (reconstructed_x, latent_vector).
        """
        latent = self.encoder(x)
        x = self.decoder(latent)
        return x, latent

def predict_ae_latent(model, x, device):
    """
    Extract 2-dimensional latent feature representations for input tensor x using trained Autoencoder model.
    :param model: Trained Autoencoder model.
    :param x: Input feature tensor.
    :param device: Compute device ('cpu' or 'cuda').
    :return: Latent feature tensor transferred to CPU.
    """
    model.eval()  
    with torch.no_grad():
        x = x.to(device)
        _, latent = model(x)
    return latent.cpu()

def cal(model, loader, ac_offset, device='cuda'):
    """
    Calculate deep learning regression model predictions and ground truth targets across DataLoader batches.
    :param model: PyTorch regression model.
    :param loader: PyTorch DataLoader for validation or testing.
    :param ac_offset: Frequency channel offset index.
    :param device: Compute device ('cpu' or 'cuda').
    :return: Tuple of (outputs_array, targets_array) as NumPy arrays.
    """
    model.eval()
    outputs_list = []
    targets_list = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.float().to(device), targets.float().to(device)            
            outputs = model(inputs, targets[:, 0], targets[:, 1])
            outputs_list.append(outputs)
            targets_list.append(targets[:, ac_offset : ac_offset + 6])
    
    outputs_array = torch.cat(outputs_list).cpu().numpy()
    targets_array = torch.cat(targets_list).cpu().numpy()
    
    return outputs_array, targets_array

# Kneedle elbow method analysis helper
def find_kneedle(k_values, sse_values):
    """
    Implement Kneedle (Elbow Method) algorithm to find maximum distance from diagonal on SSE curve for optimal cluster count K.
    :param k_values: Range of candidate cluster K values (e.g., 2 to 10).
    :param sse_values: Sum of Squared Errors (Inertia) corresponding to each K.
    :return: Tuple of (best_k, distances) containing optimal K and normalized distance array.
    """
    k_arr = np.array(list(k_values))
    sse_arr = np.array(sse_values)
    k_min, k_max = k_arr.min(), k_arr.max()
    sse_min, sse_max = sse_arr.min(), sse_arr.max()
    if sse_max == sse_min:
        return k_arr[0], np.zeros_like(k_arr)
    x_norm = (k_arr - k_min) / (k_max - k_min)
    y_norm = (sse_arr - sse_min) / (sse_max - sse_min)
    distances = (1 - x_norm) - y_norm
    knee_idx = np.argmax(distances)
    return k_arr[knee_idx], distances

def run_elbow_method_analysis(train_data, device):
    """
    Execute 50 repeated independent runs of Elbow Method analysis (K=2..10) on training dataset to avoid data leakage.
    :param train_data: Training dataset DataFrame for current fold.
    :param device: Compute device ('cpu' or 'cuda').
    """

    #search_range = 11
    search_range = 51

    print("\n" + "="*80)
    print(f"Running Elbow Method Analysis on Training Dataset ({search_range-1} Runs) to prevent data leak")
    print("="*80)
    
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
    val_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    criterion = nn.HuberLoss()

    for run_idx in range(1, search_range):
        random.seed(run_idx)
        np.random.seed(run_idx)
        torch.manual_seed(run_idx)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(run_idx)
            
        encoder = Autoencoder().to(device)
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
                
        latent_vectors = []
        encoder.eval()
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(device)
                _, latent = encoder(data)
                latent_vectors.append(latent.cpu())
        latent_vectors = torch.cat(latent_vectors, dim=0)
        latent_vectors_np = latent_vectors.numpy()
        
        k_range = range(2, 11)
        sse = []
        for k in k_range:
            kmeans_temp = KMeans(n_clusters=k, random_state=run_idx, n_init=30)
            kmeans_temp.fit(latent_vectors_np)
            sse.append(kmeans_temp.inertia_)
            
        knee_k, kneedle_dists = find_kneedle(k_range, sse)
        
        sse_2nd_diff = [None] * len(k_range)
        for i in range(1, len(k_range) - 1):
            sse_2nd_diff[i] = sse[i+1] - 2 * sse[i] + sse[i-1]
            
        print(f"\n=================== Running Iteration {run_idx}/{search_range-1} ===================")
        print(f"{'K':<5}{'sse_val':<15}{'sse_diff':<15}{'knee_dist_val':<15}")
        print("-" * 50)
        for i, k in enumerate(k_range):
            sse_val = f"{sse[i]:.4f}"
            sse_diff = f"{sse_2nd_diff[i]:.4f}" if sse_2nd_diff[i] is not None else "N/A"
            knee_dist_val = f"{kneedle_dists[i]:.4f}"
            print(f"{k:<5}{sse_val:<15}{sse_diff:<15}{knee_dist_val:<15}")
        print("-" * 50)
        print(f"Final Chosen K: {knee_k}")
        
        run_dir = f'figure/run_{run_idx}'
        os.makedirs(run_dir, exist_ok=True)
        log_path = f'{run_dir}/kmeans_metrics.log'
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"Run {run_idx}/50 - Elbow Method Kneedle Distances\n")
            f.write("="*60 + "\n")
            f.write(f"{'K':<5}{'sse_val':<15}{'sse_diff':<15}{'knee_dist_val':<15}\n")
            f.write("-"*60 + "\n")
            for i, k in enumerate(k_range):
                sse_val = f"{sse[i]:.4f}"
                sse_diff = f"{sse_2nd_diff[i]:.4f}" if sse_2nd_diff[i] is not None else "N/A"
                knee_dist_val = f"{kneedle_dists[i]:.4f}"
                f.write(f"{k:<5}{sse_val:<15}{sse_diff:<15}{knee_dist_val:<15}\n")
            f.write("="*60 + "\n")
            f.write(f"Final Chosen K: {knee_k}\n")
            f.write("="*60 + "\n")
            
        plt.figure(figsize=(3.5, 2.5))
        plt.plot(k_range, sse, marker='o', color='blue', label='SSE')
        plt.axvline(x=5, color='red', linestyle='--', label='elbow')
        plt.title('Elbow Method')
        plt.xlabel('Number of Clusters K')
        plt.ylabel('SSE')
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(f'{run_dir}/elbow_method.tiff', format='tiff', dpi=300, pil_kwargs={"compression": "tiff_lzw"})
        plt.savefig(f'{run_dir}/elbow_method.eps', format='eps', dpi=300)
        plt.show(block=False)
        plt.pause(0.1)
        plt.close()

    print("\n" + "="*80)
    print("Elbow Method Analysis Completed!")
    print("="*80 + "\n")

# PCA & Cluster Sample Plotting per Fold
def plot_fold_pca_and_cluster_samples(round_num, latent_train, labels, kx_train, kmeans):
    """
    Generate and save PCA cluster distribution plot and waveform sample plots for each K-Means cluster in current fold.
    :param round_num: Fold index (0 to 4).
    :param latent_train: 2D latent representation array of training samples.
    :param labels: K-Means cluster assignment labels for training samples.
    :param kx_train: Raw input features (acoustic trend + age diff) of training samples.
    :param kmeans: Fitted KMeans clustering model.
    """
    fold_fig_dir = f'figure/fold_{round_num}'
    os.makedirs(fold_fig_dir, exist_ok=True)

    n_clusters = 5
    markers = ['o', 's', '^', 'd', '*']
    colors = ['blue', 'green', 'red', 'purple', 'orange']

    # 1. PCA transform (or use 2D latent coordinates directly)
    if latent_train.shape[1] > 2:
        pca = PCA(n_components=2)
        latent_pca = pca.fit_transform(latent_train)
        centroids_pca = pca.transform(kmeans.cluster_centers_)
    else:
        latent_pca = latent_train
        centroids_pca = kmeans.cluster_centers_

    # Plot PCA Cluster Scatter Plot
    plt.figure(figsize=(3.5, 3.0))
    legend_handles = []
    legend_labels = []

    centroid_handle = plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], marker='X', color='black',
                                  s=150, edgecolor='white', linewidth=1.5, label='Centroid', zorder=5)
    legend_handles.append(centroid_handle)
    legend_labels.append('Centroid')

    for cls in range(n_clusters):
        mask = (labels == cls)
        if np.sum(mask) == 0:
            continue
        size = np.sum(mask)
        cluster_points = latent_pca[mask]
        cluster_handle = plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                                     marker=markers[cls % len(markers)], color=colors[cls % len(colors)],
                                     label=f'Cluster {cls} (N={size})', alpha=0.7, rasterized=True)
        legend_handles.append(cluster_handle)
        legend_labels.append(f'Cluster {cls} (N={size})')

    plt.title(f"PCA Plot (Fold {round_num + 1})")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(legend_handles, legend_labels, fontsize=8)
    plt.tight_layout()

    plt.savefig(f'{fold_fig_dir}/pca_latent_clusters.tiff', format='tiff', dpi=300, pil_kwargs={"compression": "tiff_lzw"})
    plt.savefig(f'{fold_fig_dir}/pca_latent_clusters.eps', format='eps', dpi=300)
    plt.show(block=False)
    plt.pause(0.1)
    plt.close()

    # 2. Plot 5 closest sample waveforms for each cluster
    closest_sample_indices = {}
    centroids = kmeans.cluster_centers_

    plotted_y_vals = []
    plotted_x_vals = []

    for cls in range(n_clusters):
        cluster_indices = np.where(labels == cls)[0]
        if len(cluster_indices) == 0:
            continue
        cluster_latents = latent_train[cluster_indices]
        centroid = centroids[cls]
        distances = np.linalg.norm(cluster_latents - centroid, axis=1)
        closest_relative_indices = np.argsort(distances)[:5]
        closest_sample_indices[cls] = cluster_indices[closest_relative_indices]

        for idx in closest_sample_indices[cls]:
            plotted_y_vals.append(kx_train[idx, :4])
            plotted_x_vals.append(np.cumsum(kx_train[idx, 4:]))

    if len(plotted_y_vals) > 0:
        plotted_y_vals = np.array(plotted_y_vals)
        plotted_x_vals = np.array(plotted_x_vals)
        y_min, y_max = plotted_y_vals.min(), plotted_y_vals.max()
        x_min, x_max = plotted_x_vals.min(), plotted_x_vals.max()
        y_padding = max((y_max - y_min) * 0.05, 1.0)
        x_padding = max((x_max - x_min) * 0.05, 0.5)
        y_lim = (y_min - y_padding, y_max + y_padding)
        x_lim = (x_min - x_padding, x_max + x_padding)

        for cls in range(n_clusters):
            if cls not in closest_sample_indices:
                continue
            for rank, idx in enumerate(closest_sample_indices[cls]):
                plt.figure(figsize=(3.5, 2.5))
                y_vals = kx_train[idx, :4]
                x_vals = np.cumsum(kx_train[idx, 4:])

                plt.plot(x_vals, y_vals, marker='o', color=colors[cls % len(colors)], label='AC Mean Value Difference')
                plt.xlim(x_lim)
                plt.ylim(y_lim)
                plt.title(f"Fold {round_num + 1} - Cluster {cls} Sample {rank + 1}")
                plt.xlabel("Age (Shift)")
                plt.ylabel("AC Mean Value Difference")
                plt.legend(fontsize=8)
                plt.tight_layout()

                filename = f"{fold_fig_dir}/cluster_{cls}_sample_{rank + 1}"
                plt.savefig(f"{filename}.tiff", format='tiff', dpi=300, pil_kwargs={"compression": "tiff_lzw"})
                plt.savefig(f"{filename}.eps", format='eps', dpi=300)
                plt.show(block=False)
                plt.pause(0.1)
                plt.close()

# AE + KMeans Training per Fold
def train_ae_and_kmeans(train_data, device, round_num=0, run_elbow=False):
    """
    Train Autoencoder and fit K-Means clustering model (K=5) on a single fold's training dataset.
    :param train_data: Training dataset DataFrame for current fold.
    :param device: Compute device ('cpu' or 'cuda').
    :param round_num: Fold index (0 to 4).
    :param run_elbow: Whether to run 50-run Elbow Method analysis (set to True for Fold 0 only).
    :return: Tuple of (encoder, kmeans, data_mean, data_std).
    """
    if run_elbow:
        run_elbow_method_analysis(train_data, device)

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
    labels = kmeans.fit_predict(latent_train)

    # Plot PCA cluster scatter plot and representative waveform sample plots for this fold
    plot_fold_pca_and_cluster_samples(round_num, latent_train, labels, kx_train, kmeans)

    return encoder, kmeans, data_mean, data_std

def create_kmeans_for_fold(raw_data, kmeans, encoder, data_mean, data_std, device):
    """
    Annotate dataset records with cumulative cluster trend labels (avg_trend_labels_0..4) using trained AE and KMeans.
    :param raw_data: Dataset DataFrame to be annotated (train_data, val_data, or test_data).
    :param kmeans: Fitted KMeans model for current fold.
    :param encoder: Trained Autoencoder model for current fold.
    :param data_mean: Normalization feature mean.
    :param data_std: Normalization feature standard deviation.
    :param device: Compute device ('cpu' or 'cuda').
    """
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

# Dataset Definitions
class HearDataset_MAE(Dataset):
    """
    PyTorch Dataset for hearing regression (MAE) tasks, constructing sequence windows (prev_len).
    """
    def __init__(self, df, prev_len, mode):
        """
        Initialize HearDataset_MAE instance.
        :param df: Dataset DataFrame.
        :param prev_len: Time sequence window length (default is 3).
        :param mode: Model input mode ('1D' or 'RNN').
        """
        self.x, self.y = [], []
        self.prev_len = prev_len
        self.mode = mode
        self.process_data(df)
        self.x = np.array(self.x)
        self.y = np.array(self.y)

    def process_data(self, df):
        """Process DataFrame rows into input sequence matrix self.x and target matrix self.y."""
        side_cols = [
            'Age', 'Age_diff', 'Genotype', 'Gender',
            'AC025_trend', 'AC05_trend', 'AC1_trend', 'AC2_trend', 'AC4_trend', 'AC8_trend',
            'AC025_trend_label', 'AC05_trend_label', 'AC1_trend_label', 'AC2_trend_label', 'AC4_trend_label', 'AC8_trend_label',
            'AC025_std_in_time', 'AC05_std_in_time', 'AC1_std_in_time', 'AC2_std_in_time', 'AC4_std_in_time', 'AC8_std_in_time',
            'avg_trend_labels_0', 'avg_trend_labels_1', 'avg_trend_labels_2', 'avg_trend_labels_3', 'avg_trend_labels_4',
            'avg1', 'avg2', 'avg3', 'avg4', 'four_avg',
            'EVA', 'VAsize', 'mondini',
            'peak', 'peak_agg', 'peak_diff', 'big_peak', 'big_peak_agg', 'big_peak_diff',
            'recently_peak', 'recently_big_peak', 'recently_total',
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
        """
        Extract feature subarray for a specific time step index.
        :param row: Single DataFrame row.
        :param cols: List of feature column names.
        :param index: Time step index.
        :param istest: Whether to append test metadata (Age, Age_diff, occur).
        """
        subarr = []
        for col in cols:
            subarr.append(row[col][index])
        if istest:
            subarr.append(row['Age'][index])
            subarr.append(row['Age_diff'][index])
            subarr.append(row['occur'][index])
        return subarr

    def __len__(self):
        """Return total number of dataset samples."""
        return len(self.y)

    def __getitem__(self, idx):
        """Return (feature, label) tuple at specified sample index."""
        return self.x[idx], self.y[idx]
    
class HearDataset_Peak(Dataset):
    """
    PyTorch Dataset for hearing peak classification and occurrence prediction tasks.
    """
    def __init__(self, df, prev_len, mode):
        """
        Initialize HearDataset_Peak instance.
        :param df: Dataset DataFrame.
        :param prev_len: Time sequence window length (default is 3).
        :param mode: Model input mode ('1D' or 'RNN').
        """
        self.x, self.y = [], []
        self.prev_len = prev_len
        self.mode = mode
        self.process_data(df)
        self.x = np.array(self.x)
        self.y = np.array(self.y)

    def process_data(self, df):
        """Process DataFrame rows into input sequence matrix self.x and target matrix self.y."""
        side_cols = [
            'Age', 'Age_diff', 'Genotype', 'Gender',
            'AC025_trend', 'AC05_trend', 'AC1_trend', 'AC2_trend', 'AC4_trend', 'AC8_trend',
            'AC025_trend_label', 'AC05_trend_label', 'AC1_trend_label', 'AC2_trend_label', 'AC4_trend_label', 'AC8_trend_label',
            'AC025_std_in_time', 'AC05_std_in_time', 'AC1_std_in_time', 'AC2_std_in_time', 'AC4_std_in_time', 'AC8_std_in_time',
            'avg_trend_labels_0', 'avg_trend_labels_1', 'avg_trend_labels_2', 'avg_trend_labels_3', 'avg_trend_labels_4',
            'avg1', 'avg2', 'avg3', 'avg4', 'four_avg',
            'EVA', 'VAsize', 'mondini',
            'peak', 'peak_agg', 'peak_diff', 'big_peak', 'big_peak_agg', 'big_peak_diff',
            'recently_peak', 'recently_big_peak', 'recently_total',
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
        """
        Extract feature subarray for a specific time step index.
        :param row: Single DataFrame row.
        :param cols: List of feature column names.
        :param index: Time step index.
        :param istest: Whether to append test metadata (Age, Age_diff, occur).
        """
        subarr = []
        for col in cols:
            subarr.append(row[col][index])
        if istest:
            subarr.append(row['Age'][index])
            subarr.append(row['Age_diff'][index])
            subarr.append(row['occur'][index])
        return subarr

    def __len__(self):
        """Return total number of dataset samples."""
        return len(self.y)

    def __getitem__(self, idx):
        """Return (feature, label) tuple at specified sample index."""
        return self.x[idx], self.y[idx]

# Helper Baseline Models
class AllPositive:
    """Baseline classifier that predicts class 1 with probability 1.0 for all samples."""
    def fit(self, X, y):
        """No-op fit method for compatibility."""
        pass
    def predict_proba(self, X):
        """Return N x 2 array of ones."""
        return np.ones((X.shape[0], 2)) 

class AllNegative:
    """Baseline classifier that predicts class 1 with probability 0.0 for all samples."""
    def fit(self, X, y):
        """No-op fit method for compatibility."""
        pass
    def predict_proba(self, X):
        """Return N x 2 array of zeros."""
        return np.zeros((X.shape[0], 2)) 

# Feature Extraction Helpers for Scikit-Learn Models
def prepare_data(dataset, id):
    """
    Flatten PyTorch Dataset sequences into feature matrix X and target vector y for scikit-learn regression model (frequency id).
    :param dataset: HearDataset_MAE instance.
    :param id: Frequency channel index (0 to 5).
    :return: Tuple of (X, y) feature matrix and target vector.
    """
    sz = dataset.x.shape[0] 
    X = dataset.x.reshape(sz, -1)
    diff = dataset.y[:, 1]
    X = np.hstack((X, diff[:, None]))
    y = dataset.y[:, 4 + id]
    return X, y

def prepare_data_peak(dataset, id):
    """
    Flatten PyTorch Dataset sequences into feature matrix X for auxiliary Peak classifier model.
    :param dataset: HearDataset_Peak instance.
    :param id: Frequency channel index.
    :return: Feature matrix X.
    """
    sz = dataset.x.shape[0] 
    X = dataset.x.reshape(sz, -1)
    diff = np.full((len(X),), 0.5)
    X = np.hstack((X, diff[:, None]))
    return X


# ==============================================================================
# MAIN EXPERIMENT PIPELINE
# ==============================================================================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using compute device: {device}")

    # Load initial acoustic dataset
    df_s2 = load_s2_table('S2 Table.csv')
    df_new, raw_data = process_acoustic_data(df_s2)

    # Global 5-Fold Cross Validation Split (72% Train / 8% Val / 20% Test)
    # Using seed = 1 as originally specified
    folder_data = k_folder_split_data(raw_data, num_seed=1, k=5, val=True)

    print("\n" + "="*80)
    print("Preprocessing 5 Folds: Training Autoencoder & KMeans once per fold...")
    print("="*80)

    preprocessed_folds = []

    for round_num, (train_data, val_data, test_data) in enumerate(folder_data):
        print(f"--- Processing Fold {round_num + 1}/5 ---")
        logging.info(f"Processing fold {round_num}")

        train_data = train_data.copy()
        val_data = val_data.copy()
        test_data = test_data.copy()

        # Run 50-run elbow analysis ONLY on Fold 0
        run_elbow = (round_num == 0)
        encoder_fold, kmeans_fold, mean_fold, std_fold = train_ae_and_kmeans(train_data, device, round_num=round_num, run_elbow=run_elbow)

        # Annotate cluster labels on all fold splits
        create_kmeans_for_fold(train_data, kmeans_fold, encoder_fold, mean_fold, std_fold, device)
        create_kmeans_for_fold(val_data, kmeans_fold, encoder_fold, mean_fold, std_fold, device)
        create_kmeans_for_fold(test_data, kmeans_fold, encoder_fold, mean_fold, std_fold, device)

        # Save model artifacts for fold
        save_dir = f"model_weight/saved_models/fold_{round_num}"
        save_pytorch_model(encoder_fold, f"{save_dir}/autoencoder.pth")
        save_model_artifact(kmeans_fold, f"{save_dir}/kmeans.pkl")
        save_numpy_array(mean_fold.cpu().numpy(), f"{save_dir}/scaler_mean.npy")
        save_numpy_array(std_fold.cpu().numpy(), f"{save_dir}/scaler_std.npy")

        # Build PyTorch Dataset instances for this fold
        train_mae = HearDataset_MAE(train_data, 3, 'RNN')
        val_mae = HearDataset_MAE(val_data, 3, 'RNN')
        test_mae = HearDataset_MAE(test_data, 3, 'RNN')

        train_peak = HearDataset_Peak(train_data, 3, 'RNN')
        val_peak = HearDataset_Peak(val_data, 3, 'RNN')
        test_peak = HearDataset_Peak(test_data, 3, 'RNN')

        preprocessed_folds.append({
            'round_num': round_num,
            'save_dir': save_dir,
            'train_data': train_data, 'val_data': val_data, 'test_data': test_data,
            'train_mae': train_mae, 'val_mae': val_mae, 'test_mae': test_mae,
            'train_peak': train_peak, 'val_peak': val_peak, 'test_peak': test_peak,
        })

    all_experiment_results = {}

    # --------------------------------------------------------------------------
    # Experiment 1: Proposed Model (ElasticNet + CatBoost) & Naive Baselines
    # --------------------------------------------------------------------------
    print("\n" + "="*80)
    print("Running Experiment 1: Proposed Model (ElasticNet + CatBoost) & Naive Baselines")
    print("="*80)

    for fold in preprocessed_folds:
        round_num = fold['round_num']
        save_dir = fold['save_dir']
        train_mae, val_mae, test_mae = fold['train_mae'], fold['val_mae'], fold['test_mae']
        train_peak, val_peak, test_peak = fold['train_peak'], fold['val_peak'], fold['test_peak']

        models = {'elasticnet': [ElasticNetCV(cv=5, max_iter=5000, tol=1e-3) for _ in range(6)]}
        clfs = {
            'catboost': cb.CatBoostClassifier(iterations=100, learning_rate=0.01, eval_metric='AUC', silent=True),
            'all_positive': AllPositive(),
            'all_negative': AllNegative()
        }

        for id in range(6):
            X_train, y_train = prepare_data(train_mae, id)
            X_test, y_test = prepare_data(test_mae, id)
            for model_name, model in models.items():
                model[id].fit(X_train, y_train)

        for model_name, model in models.items():
            X_train, y_train = train_peak[:][0], train_peak[:][1][:, -1]
            X_val, y_val = val_peak[:][0], val_peak[:][1][:, -1]
            X_test, y_test = test_peak[:][0], test_peak[:][1][:, -1]

            X_train = X_train.reshape(X_train.shape[0], -1)
            X_val = X_val.reshape(X_val.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)
            
            y_train = np.nan_to_num(y_train).astype(int)
            y_val = np.nan_to_num(y_val).astype(int)
            y_test = np.nan_to_num(y_test).astype(int)

            for id in range(6):
                X_train_peak = prepare_data_peak(train_peak, id)
                y_pred = model[id].predict(X_train_peak)
                X_train = np.hstack((X_train, y_pred[:, None]))
                
                X_val_peak = prepare_data_peak(val_peak, id)
                y_pred = model[id].predict(X_val_peak)
                X_val = np.hstack((X_val, y_pred[:, None]))

                X_test_peak = prepare_data_peak(test_peak, id)
                y_pred = model[id].predict(X_test_peak)
                X_test = np.hstack((X_test, y_pred[:, None]))

            for clf_name, clf in clfs.items():
                clf.fit(X_train, y_train)
                y_scores_train = clf.predict_proba(X_train)[:, 1]
                y_scores_test = clf.predict_proba(X_test)[:, 1]

                if clf_name == 'all_negative':
                    optimal_threshold = 0.5
                else:
                    fpr, tpr, thresholds = roc_curve(y_train, y_scores_train)
                    optimal_idx = np.argmax(tpr - fpr)
                    optimal_threshold = thresholds[optimal_idx]

                y_pred_test = (y_scores_test >= optimal_threshold).astype(int)

                precision = precision_score(y_test, y_pred_test, zero_division=0)
                recall = recall_score(y_test, y_pred_test, zero_division=0)
                f1 = f1_score(y_test, y_pred_test, zero_division=0)
                try:
                    roc_auc = roc_auc_score(y_test, y_scores_test)
                except ValueError:
                    roc_auc = 0.5
                accuracy = accuracy_score(y_test, y_pred_test)

                precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_scores_test)
                pr_auc = auc(recall_vals[::-1], precision_vals[::-1])

                tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test, labels=[0, 1]).ravel()
                sensitivity = recall
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                balanced_accuracy = (sensitivity + specificity) / 2

                key = f"{model_name}_{clf_name}"
                if key not in all_experiment_results:
                    all_experiment_results[key] = {
                        "auc": [], "pr_auc": [], "f1": [], "precision": [], "recall": [], "specificity": [], "balanced_accuracy": [], "accuracy": [], "optimal_threshold": []
                    }
                res = all_experiment_results[key]
                res["auc"].append(roc_auc)
                res["pr_auc"].append(pr_auc)
                res["f1"].append(f1)
                res["precision"].append(precision)
                res["recall"].append(recall)
                res["specificity"].append(specificity)
                res["balanced_accuracy"].append(balanced_accuracy)
                res["accuracy"].append(accuracy)
                res["optimal_threshold"].append(optimal_threshold)

                print(f"  [Fold {round_num + 1}/5] {key:<35} | AUC: {roc_auc:.4f} | F1: {f1:.4f} | Recall: {recall:.4f} | Spec: {specificity:.4f} | Acc: {accuracy:.4f}")

    # --------------------------------------------------------------------------
    # Experiment 2: Deep Learning Regressors + CatBoost Comparative Baselines
    # --------------------------------------------------------------------------
    print("\n" + "="*80)
    print("Running Experiment 2: Deep Learning Regressors + CatBoost Comparative Baselines")
    print("="*80)

    feature_len = 44
    ac_offset = 4
    num_epochs = 20
    criterion_choices = {'L1Loss': nn.L1Loss()}

    for criterion_name, criterion in criterion_choices.items():
        for fold in preprocessed_folds:
            round_num = fold['round_num']
            save_dir = fold['save_dir']
            train_mae, val_mae, test_mae = fold['train_mae'], fold['val_mae'], fold['test_mae']
            train_peak, val_peak, test_peak = fold['train_peak'], fold['val_peak'], fold['test_peak']

            train_loader = DataLoader(dataset=train_mae, batch_size=256, shuffle=True)
            val_loader = DataLoader(dataset=val_mae, batch_size=256, shuffle=False)
            test_loader = DataLoader(dataset=test_mae, batch_size=256, shuffle=False)

            dl_models = {
                f'{criterion_name}_transformer': Transformer_peaks_Model(feature_len, 16, 1, 4).to(device),
                f'{criterion_name}_mlp': MLP_peaks_Model(feature_len, 3, 128).to(device),
                f'{criterion_name}_attention': AModel(feature_len, 64, 1, 16).to(device),
                f'{criterion_name}_lstm': LSTM_peaks_Model(feature_len, 128, 1).to(device),
            }

            for model_name, model in dl_models.items():
                optimizer = optim.AdamW(model.parameters(), lr=0.001)
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

                best_val_loss = float('inf')
                for epoch in range(num_epochs):
                    train_loss = train_epoch(model, optimizer, criterion, train_loader, ac_offset, device)
                    val_loss = evaluate(model, criterion, val_loader, ac_offset, device)
                    scheduler.step()

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        os.makedirs('model_weight', exist_ok=True)
                        torch.save(model.state_dict(), 'model_weight/best_model.pth')

                best_model = model.to(device)
                best_model.load_state_dict(torch.load('model_weight/best_model.pth'))
                
                a, b = cal(best_model, test_loader, ac_offset, device)
                mae_per_freq = np.mean(abs(a - b), axis=0)
                overall_mae = np.mean(mae_per_freq)

                X_train, y_train = train_peak[:][0], train_peak[:][1][:, -1]
                X_test, y_test = test_peak[:][0], test_peak[:][1][:, -1]
                X_train = X_train.reshape(X_train.shape[0], -1)
                X_test = X_test.reshape(X_test.shape[0], -1)

                X_train_guide = torch.tensor(train_peak[:][0]).float().to(device)
                y_train_guide = predict(best_model, X_train_guide, device).cpu().detach().numpy()           
                X_train = np.hstack((X_train, y_train_guide))
                
                X_test_guide = torch.tensor(test_peak[:][0]).float().to(device)
                y_test_guide = predict(best_model, X_test_guide, device).cpu().detach().numpy()
                X_test = np.hstack((X_test, y_test_guide))
                
                clf = cb.CatBoostClassifier(iterations=100, learning_rate=0.01, depth=6, silent=True)
                clf.fit(X_train, y_train)

                y_scores_train = clf.predict_proba(X_train)[:, 1]  
                y_scores_test = clf.predict_proba(X_test)[:, 1] 

                fpr, tpr, thresholds = roc_curve(y_train, y_scores_train)
                optimal_idx = np.argmax(tpr - fpr)
                optimal_threshold = thresholds[optimal_idx]

                y_pred_test = (y_scores_test >= optimal_threshold).astype(int)

                precision = precision_score(y_test, y_pred_test, zero_division=0)
                recall = recall_score(y_test, y_pred_test, zero_division=0)
                f1 = f1_score(y_test, y_pred_test, zero_division=0)
                try:
                    roc_auc = roc_auc_score(y_test, y_scores_test)
                except ValueError:
                    roc_auc = 0.5
                accuracy = accuracy_score(y_test, y_pred_test)

                precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_scores_test)
                pr_auc = auc(recall_vals[::-1], precision_vals[::-1])

                tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test, labels=[0, 1]).ravel()
                sensitivity = recall
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                balanced_accuracy = (sensitivity + specificity) / 2

                key = f"{model_name}_CAT_R"
                if key not in all_experiment_results:
                    all_experiment_results[key] = {
                        "auc": [], "pr_auc": [], "f1": [], "precision": [], "recall": [], "specificity": [], "balanced_accuracy": [], "accuracy": [], "optimal_threshold": [],
                        "reg_mae": [], "reg_mae_individual": []
                    }
                res = all_experiment_results[key]
                res["auc"].append(roc_auc)
                res["pr_auc"].append(pr_auc)
                res["f1"].append(f1)
                res["precision"].append(precision)
                res["recall"].append(recall)
                res["specificity"].append(specificity)
                res["balanced_accuracy"].append(balanced_accuracy)
                res["accuracy"].append(accuracy)
                res["optimal_threshold"].append(optimal_threshold)
                res["reg_mae"].append(overall_mae)
                res["reg_mae_individual"].append(mae_per_freq.tolist())

                print(f"  [Fold {round_num + 1}/5] {key:<35} | Reg MAE: {overall_mae:.4f} | AUC: {roc_auc:.4f} | F1: {f1:.4f} | Recall: {recall:.4f} | Spec: {specificity:.4f} | Acc: {accuracy:.4f}")

    # --------------------------------------------------------------------------
    # Experiment 3: Standalone ElasticNet Hearing Threshold Regression Baseline
    # --------------------------------------------------------------------------
    print("\n" + "="*80)
    print("Running Experiment 3: Standalone ElasticNet Hearing Threshold Regression Baseline")
    print("="*80)

    for fold in preprocessed_folds:
        round_num = fold['round_num']
        save_dir = fold['save_dir']
        train_mae, test_mae = fold['train_mae'], fold['test_mae']

        models = {'elasticnet': [ElasticNetCV(cv=3, max_iter=5000, tol=1e-3) for _ in range(6)]}
        individual_maes = {model: [] for model in models.keys()}

        for id in range(6):
            X_train, y_train = prepare_data(train_mae, id)
            X_test, y_test = prepare_data(test_mae, id)
            
            for model_name, model in models.items():
                model[id].fit(X_train, y_train)
                y_pred = model[id].predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                individual_maes[model_name].append(mae)

        for model_name, model in models.items():
            key = f"{model_name}_regression"
            if key not in all_experiment_results:
                all_experiment_results[key] = {
                    "reg_mae": [],
                    "reg_mae_individual": []
                }
            mean_mae = np.mean(individual_maes[model_name])
            all_experiment_results[key]["reg_mae"].append(mean_mae)
            all_experiment_results[key]["reg_mae_individual"].append(individual_maes[model_name])

            print(f"  [Fold {round_num + 1}/5] {key:<35} | Reg MAE: {mean_mae:.4f}")

    # ==============================================================================
    # PRINT CLEAN SUMMARY TABLE OF ALL EXPERIMENTS
    # ==============================================================================
    def fmt_metric(lst):
        """
        Format evaluation metric list into 'mean ± std' string formatted to 4 decimal places.
        :param lst: Metric value list across 5 CV folds.
        :return: Formatted string or 'N/A' if empty or all NaN.
        """
        if lst is None or (isinstance(lst, list) and len(lst) == 0):
            return "N/A"
        arr = np.array(lst, dtype=float)
        valid_arr = arr[~np.isnan(arr)]
        if len(valid_arr) == 0:
            return "N/A"
        return f"{np.mean(valid_arr):.4f}±{np.std(valid_arr):.4f}"

    print("\n" + "=" * 202)
    print(f"{'FINAL EVALUATION SUMMARY (Averages across 5 Folds with mean ± std)':^202}")
    print("=" * 202)
    print(f"{'Comparison Target':<40} | {'Reg. MAE':<15} | {'AUC':<15} | {'PR-AUC':<15} | {'F1':<15} | {'Recall':<15} | {'Spec.':<15} | {'Acc':<15} | {'Precision':<15} | {'Threshold':<15}")
    print("-" * 202)

    for name, metrics in all_experiment_results.items():
        mae_val = fmt_metric(metrics.get("reg_mae"))
        auc_val = fmt_metric(metrics.get("auc"))
        pr_auc_val = fmt_metric(metrics.get("pr_auc"))
        f1_val = fmt_metric(metrics.get("f1"))
        rec_val = fmt_metric(metrics.get("recall"))
        spec_val = fmt_metric(metrics.get("specificity"))
        acc_val = fmt_metric(metrics.get("accuracy"))
        prec_val = fmt_metric(metrics.get("precision"))
        thresh_val = fmt_metric(metrics.get("optimal_threshold"))
        
        print(f"{name:<40} | {mae_val:<15} | {auc_val:<15} | {pr_auc_val:<15} | {f1_val:<15} | {rec_val:<15} | {spec_val:<15} | {acc_val:<15} | {prec_val:<15} | {thresh_val:<15}")

    print("=" * 202)
    print("\nIndividual Frequency MAEs (if applicable):")
    for name, metrics in all_experiment_results.items():
        if "reg_mae_individual" in metrics and len(metrics["reg_mae_individual"]) > 0:
            mean_individual = np.mean(metrics["reg_mae_individual"], axis=0)
            formatted_mae = ", ".join([f"{v:.4f}" for v in mean_individual])
            print(f"  {name:<40}: [{formatted_mae}]")
    print("=" * 202)
    sys.stdout.flush()
