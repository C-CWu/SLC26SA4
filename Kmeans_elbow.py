from utils import *
from dataset import *
from models import *
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, PrecisionRecallDisplay
import logging
import os
import matplotlib.pyplot as plt

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

def find_kneedle(k_values, sse_values):
    k_arr = np.array(list(k_values))
    sse_arr = np.array(sse_values)
    k_min, k_max = k_arr.min(), k_arr.max()
    sse_min, sse_max = sse_arr.min(), sse_arr.max()
    if sse_max == sse_min:
        return k_arr[0], np.zeros_like(k_arr)
    x_norm = (k_arr - k_min) / (k_max - k_min)
    y_norm = (sse_arr - sse_min) / (sse_max - sse_min)
    # Distance to the line connecting (0, 1) and (1, 0)
    distances = (1 - x_norm) - y_norm
    knee_idx = np.argmax(distances)
    return k_arr[knee_idx], distances

def calculate_gap_statistic(data, k_range, B=5, random_state=0):
    n_samples, n_features = data.shape
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    
    ref_log_w = np.zeros((len(k_range), B))
    for b in range(B):
        ref_data = np.random.uniform(data_min, data_max, size=(n_samples, n_features))
        for i, k in enumerate(k_range):
            km = KMeans(n_clusters=k, random_state=random_state + b, n_init=10)
            km.fit(ref_data)
            ref_log_w[i, b] = np.log(km.inertia_)
            
    expected_log_w = np.mean(ref_log_w, axis=1)
    sd_k = np.std(ref_log_w, axis=1)
    s_k = sd_k * np.sqrt(1 + 1.0 / B)
    return expected_log_w, s_k


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

from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim


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

class CNN_Autoencoder(nn.Module):
    def __init__(self):
        super(CNN_Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(2, 2)),
            nn.Mish(),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(2, 1)),  
            nn.Mish(),
            nn.Flatten(),  
            nn.Linear(8 * 2 * 1, 2)  
        )

        self.decoder = nn.Sequential(
            nn.Linear(2, 8 * 2 * 1),  
            nn.Mish(),
            nn.Unflatten(1, (8, 2, 1)),  
            nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=(2, 1)),  
            nn.Mish(),
            nn.ConvTranspose2d(in_channels=4, out_channels=1, kernel_size=(2, 2)),  
        )

    def forward(self, x):
        x = x.unsqueeze(1)  
        latent = self.encoder(x)
        x = self.decoder(latent)
        x = x.squeeze(1)  
        return x, latent


X_train_flattened = torch.tensor(kx).float()
data_mean = X_train_flattened.mean(dim=0, keepdim=True)
data_std = X_train_flattened.std(dim=0, keepdim=True)
X_train_normalized = (X_train_flattened - data_mean) / data_std

dataset = TensorDataset(X_train_normalized, X_train_normalized)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset, batch_size=32, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.HuberLoss()
for run_idx in range(1, 51):
    print(f"\n=================== Running Iteration {run_idx}/50 ===================")
    
    # Set random seeds for this run to guarantee reproducibility
    import random
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
        
        # Print loss only every 100 epochs or when finished to avoid console clutter
        if (epoch + 1) % 100 == 0 or loss.item() < 0.061:
            print(f'Run {run_idx}/50 - Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        if loss.item() < 0.061:
            break
            
    # Set up directory for the current run
    run_dir = f'figure/run_{run_idx}'
    pca_dir = f'{run_dir}/pca'
    os.makedirs(pca_dir, exist_ok=True)
    
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    
    latent_vectors = []
    encoder.eval()  
    
    with torch.no_grad():
        for data, _ in val_loader:
            data = data.to(device)
            _, latent = encoder(data)
            latent_vectors.append(latent.cpu())
            
    latent_vectors = torch.cat(latent_vectors, dim=0)
    
    n_clusters = 5  
    kmeans = KMeans(n_clusters=n_clusters, random_state=run_idx)
    kmeans.fit(latent_vectors)
    labels = kmeans.predict(latent_vectors)
    
    pca = PCA(n_components=2)
    latent_vectors_pca = pca.fit_transform(latent_vectors.numpy())
    
    markers = ['o', 's', '^', 'd', '*']  
    colors = ['blue', 'green', 'red', 'purple', 'orange']  
    
    plt.figure(figsize=(5.25, 4.5))
    
    # Prepare lists for custom legend ordering
    legend_handles = []
    legend_labels = []
    
    # 1. Compute and Plot Centroids in PCA-transformed latent space
    centroids_x = []
    centroids_y = []
    for cls in range(n_clusters):
        mask = labels == cls
        centroid = latent_vectors_pca[mask].mean(axis=0)
        centroids_x.append(centroid[0])
        centroids_y.append(centroid[1])
        
    centroid_handle = plt.scatter(centroids_x, centroids_y, marker='X', color='black', 
                                  s=150, edgecolor='white', linewidth=1.5, label='Centroid', zorder=5)
    legend_handles.append(centroid_handle)
    legend_labels.append('Centroid')
    
    # 2. Plot Cluster Points in PCA-transformed latent space
    for cls in np.unique(labels):
        mask = labels == cls
        size = np.sum(mask)
        cluster_points = latent_vectors_pca[mask]
        
        cluster_handle = plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                                     marker=markers[cls], color=colors[cls], 
                                     label=f'Cluster {cls} (N={size})', alpha=0.7,
                                     rasterized=True)
        legend_handles.append(cluster_handle)
        legend_labels.append(f'Cluster {cls} (N={size})')
        
    plt.title("PCA plot")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    
    # Set ordered legend: Centroid first
    plt.legend(legend_handles, legend_labels)
    plt.tight_layout()
    
    # Save the PCA figures to figure/run_{run_idx}/pca folder
    plt.savefig(f'{pca_dir}/pca_latent_clusters.tiff', format='tiff', dpi=300, pil_kwargs={"compression": "tiff_lzw"})
    plt.savefig(f'{pca_dir}/pca_latent_clusters.eps', format='eps', dpi=300)
    plt.close()
    
    # Collect X and Y values only from the 25 samples to be plotted (closest to centroids)
    plotted_y_vals = []
    plotted_x_vals = []
    
    # Calculate closest indices for each cluster
    closest_sample_indices = {}
    latent_vectors_np = latent_vectors.numpy()
    centroids = kmeans.cluster_centers_
    
    for cls in range(n_clusters):
        cluster_indices = np.where(labels == cls)[0]
        cluster_latents = latent_vectors_np[cluster_indices]
        centroid = centroids[cls]
        distances = np.linalg.norm(cluster_latents - centroid, axis=1)
        closest_relative_indices = np.argsort(distances)[:5]
        closest_sample_indices[cls] = cluster_indices[closest_relative_indices]
        
        for idx in closest_sample_indices[cls]:
            plotted_y_vals.append(kx[idx, :4])
            plotted_x_vals.append(np.cumsum(kx[idx, 4:]))
            
    plotted_y_vals = np.array(plotted_y_vals)
    plotted_x_vals = np.array(plotted_x_vals)
    
    y_min, y_max = plotted_y_vals.min(), plotted_y_vals.max()
    x_min, x_max = plotted_x_vals.min(), plotted_x_vals.max()
    
    # Add margin/padding to axes limits
    y_padding = max((y_max - y_min) * 0.05, 1.0)
    x_padding = max((x_max - x_min) * 0.05, 0.5)
    
    y_lim = (y_min - y_padding, y_max + y_padding)
    x_lim = (x_min - x_padding, x_max + x_padding)
    
    # Generate 5 raw input line plots for each cluster (total 25 plots)
    for cls in range(n_clusters):
        for rank, idx in enumerate(closest_sample_indices[cls]):
            plt.figure(figsize=(3.5, 2.5))
            
            y_vals = kx[idx, :4]
            x_vals = np.cumsum(kx[idx, 4:])
            
            plt.plot(x_vals, y_vals, marker='o', color=colors[cls], label='AC Mean Value Difference')
            
            # Enforce identical axes ranges
            plt.xlim(x_lim)
            plt.ylim(y_lim)
            
            plt.title(f"A Sample from Cluster {cls}")
            plt.xlabel("Age (Shift)")
            plt.ylabel("dB")
            plt.legend()
            plt.tight_layout()
            
            # Save to figure/run_{run_idx}/pca folder
            filename = f"{pca_dir}/cluster_{cls}_sample_{rank + 1}"
            plt.savefig(f"{filename}.tiff", format='tiff', dpi=300, pil_kwargs={"compression": "tiff_lzw"})
            plt.savefig(f"{filename}.eps", format='eps', dpi=300)
            plt.close()
            
    # Compute SSE, Silhouette Score, Calinski-Harabasz Index, and Davies-Bouldin Index for K = 2 to 10
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    
    k_range = range(2, 11)
    sse = []
    silhouette_scores = []
    ch_scores = []
    db_scores = []
    
    # Convert latent vectors to numpy array for metric calculation
    latent_vectors_np = latent_vectors.numpy()
    
    for k in k_range:
        kmeans_temp = KMeans(n_clusters=k, random_state=run_idx, n_init=30)
        kmeans_temp.fit(latent_vectors_np)
        labels_temp = kmeans_temp.labels_
        
        sse.append(kmeans_temp.inertia_)
        silhouette_scores.append(silhouette_score(latent_vectors_np, labels_temp))
        ch_scores.append(calinski_harabasz_score(latent_vectors_np, labels_temp))
        db_scores.append(davies_bouldin_score(latent_vectors_np, labels_temp))
        
    # Calculate Kneedle elbow point
    knee_k, kneedle_dists = find_kneedle(k_range, sse)
    
    # Calculate Gap Statistic expected log W and s_k
    expected_log_w, s_k = calculate_gap_statistic(latent_vectors_np, k_range, B=5, random_state=run_idx)
    gaps = expected_log_w - np.log(sse)
    
    # Determine optimal K from Gap Statistic: smallest K where Gap(K) >= Gap(K+1) - s_{K+1}
    gap_opt_k = None
    for i in range(len(k_range) - 1):
        if gaps[i] >= gaps[i+1] - s_k[i+1]:
            gap_opt_k = k_range[i]
            break
    if gap_opt_k is None:
        gap_opt_k = k_range[-1]
        
    # Calculate second-order differences (2nd Diff) for K = 3 to 9
    sse_2nd_diff = [None] * len(k_range)
    sil_2nd_diff = [None] * len(k_range)
    ch_2nd_diff = [None] * len(k_range)
    db_2nd_diff = [None] * len(k_range)
    
    for i in range(1, len(k_range) - 1):
        sse_2nd_diff[i] = sse[i+1] - 2 * sse[i] + sse[i-1]
        sil_2nd_diff[i] = silhouette_scores[i+1] - 2 * silhouette_scores[i] + silhouette_scores[i-1]
        ch_2nd_diff[i] = ch_scores[i+1] - 2 * ch_scores[i] + ch_scores[i-1]
        db_2nd_diff[i] = db_scores[i+1] - 2 * db_scores[i] + db_scores[i-1]
        
    # Save the metrics to a log file inside the run folder
    log_path = f'{run_dir}/kmeans_metrics.log'
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("K-means Clustering Metrics & Second-Order Differences for K = 2 to 10\n")
        f.write("="*220 + "\n")
        f.write(f"{'K':<5}{'SSE':<15}{'SSE 2nd Diff':<18}{'Silhouette':<15}{'Sil 2nd Diff':<18}{'Calinski-Harabasz':<20}{'CH 2nd Diff':<18}{'Davies-Bouldin':<18}{'DBI 2nd Diff':<18}{'Kneedle Dist':<18}{'Gap Value':<15}{'s_k':<12}\n")
        f.write("-"*220 + "\n")
        for i, k in enumerate(k_range):
            sse_val = f"{sse[i]:.4f}"
            sse_diff = f"{sse_2nd_diff[i]:.4f}" if sse_2nd_diff[i] is not None else "N/A"
            sil_val = f"{silhouette_scores[i]:.4f}"
            sil_diff = f"{sil_2nd_diff[i]:.4f}" if sil_2nd_diff[i] is not None else "N/A"
            ch_val = f"{ch_scores[i]:.4f}"
            ch_diff = f"{ch_2nd_diff[i]:.4f}" if ch_2nd_diff[i] is not None else "N/A"
            db_val = f"{db_scores[i]:.4f}"
            db_diff = f"{db_2nd_diff[i]:.4f}" if db_2nd_diff[i] is not None else "N/A"
            knee_dist_val = f"{kneedle_dists[i]:.4f}"
            gap_val = f"{gaps[i]:.4f}"
            sk_val = f"{s_k[i]:.4f}"
            
            f.write(f"{k:<5}{sse_val:<15}{sse_diff:<18}{sil_val:<15}{sil_diff:<18}{ch_val:<20}{ch_diff:<18}{db_val:<18}{db_diff:<18}{knee_dist_val:<18}{gap_val:<15}{sk_val:<12}\n")
            
        f.write("="*220 + "\n")
        f.write("Automated Selection Algorithms Recommendations:\n")
        f.write(f"- Kneedle Algorithm (Elbow detector) recommends K = {knee_k}\n")
        f.write(f"- Gap Statistic Algorithm recommends K = {gap_opt_k}\n")
        f.write("="*220 + "\n")
        
    print(f"Run {run_idx}/50 completed. Metrics written to: {log_path}")
    print(f"  [Recommendations] Kneedle: K={knee_k} | Gap Statistic: K={gap_opt_k}")
    
    # Plot the elbow curve
    plt.figure(figsize=(3.5, 2.5))
    plt.plot(k_range, sse, marker='o', color='blue', label='SSE')
    plt.axvline(x=5, color='red', linestyle='--', label='elbow')
    
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters K')
    plt.ylabel('SSE')
    plt.legend()
    plt.tight_layout()
    
    # Save the elbow curve figures
    plt.savefig(f'{run_dir}/elbow_method.tiff', format='tiff', dpi=300, pil_kwargs={"compression": "tiff_lzw"})
    plt.savefig(f'{run_dir}/elbow_method.eps', format='eps', dpi=300)
    plt.close()
