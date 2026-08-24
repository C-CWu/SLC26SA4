import re
from scipy.stats import skew
import pandas as pd
import matplotlib.pyplot as plt
import numpy  as np
import random
import math
from sklearn.model_selection import train_test_split, KFold
import torch
from scipy.stats import kurtosis, entropy


def process_image_data(file_path: str, sheet_name: str) -> (dict, dict):
    """
    Processes image data from an Excel file and creates dictionaries mapping IDs to 'VA size(mm)' and 'mondini'.
    
    Parameters:
    file_path (str): The path to the Excel file.
    sheet_name (str): The name of the sheet in the Excel file containing the image data.
    
    Returns:
    tuple: A tuple containing two dictionaries:
        - ID2VAsize: A dictionary mapping IDs to 'VA size(mm)'.
        - ID2mondini: A dictionary mapping IDs to 'mondini'.
    """
    image_data = pd.read_excel(file_path, sheet_name=sheet_name)
    image_data = image_data[2:]
    
    # Fill NaN values in 'VA size(mm)' and 'mondini' columns with -1
    image_data['VA size(mm)'] = image_data['VA size(mm)'].fillna(-1).infer_objects()
    image_data['mondini'] = image_data['mondini'].fillna(-1).infer_objects()
    image_data['EVA'] = image_data['EVA'].fillna(-1).infer_objects()
        
    # dictionaries to map IDs to 'VA size(mm)' and 'mondini'
    ID2VAsize = {}
    ID2mondini = {}
    ID2EVA = {}
    
    for _, row in image_data.iterrows():
        # Create a unique key combining 'ID' and '左右耳' (left or right ear) in lowercase
        key = row['ID'] + '_' + row['左右耳'].lower()
        
        # Map the key to the corresponding 'VA size(mm)' and 'mondini' values
        ID2VAsize[key] = row['VA size(mm)']
        ID2mondini[key] = row['mondini']
        ID2EVA[key] = row['EVA']
    
    return ID2VAsize, ID2mondini, ID2EVA


def load_s2_table(file_path: str) -> pd.DataFrame:
    """
    Reads S3 Table.csv, maps categorical variables to the specified numeric values,
    reconstructs Left/Right ears, and handles missing values via interpolation/ffill/bfill.
    """
    df_raw = pd.read_csv(file_path)
    df_raw.columns = df_raw.columns.str.strip()
    
    # Drop rows that don't have valid ID or Age
    df_raw = df_raw.dropna(subset=['No.', 'Age at exam'])
    
    # Map Gender: Female (F) -> 0, Male (M) -> 1
    df_raw['Gender'] = df_raw['Gender'].astype(str).str.strip().str.upper()
    gender_map = {'F': 0, 'M': 1, 'FEMALE': 0, 'MALE': 1}
    df_raw['gender_num'] = df_raw['Gender'].map(gender_map).fillna(0)
    
    # Map Genotype: Single -> 2, Two LoF -> 3, One non-LoF -> 1
    genotype_col = [col for col in df_raw.columns if col.startswith('Genotype')][0]
    df_raw[genotype_col] = df_raw[genotype_col].astype(str).str.strip()
    genotype_map = {'Single': 2, 'Two LoF': 3, 'One non-LoF': 1}
    df_raw['genotype_num'] = df_raw[genotype_col].map(genotype_map).fillna(1)
    
    # Parse EVA, Mondini, VAsize
    df_raw['EVA_R_num'] = df_raw['EVA_R'].astype(str).str.strip().str.upper().map({'Y': 1.0, 'N': 0.0}).fillna(-1.0)
    df_raw['EVA_L_num'] = df_raw['EVA_L'].astype(str).str.strip().str.upper().map({'Y': 1.0, 'N': 0.0}).fillna(-1.0)
    
    df_raw['Mondini_R_num'] = df_raw['Mondini_R'].astype(str).str.strip().str.upper().map({'Y': 1.0, 'N': 0.0}).fillna(-1.0)
    df_raw['Mondini_L_num'] = df_raw['Mondini_L'].astype(str).str.strip().str.upper().map({'Y': 1.0, 'N': 0.0}).fillna(-1.0)
    
    df_raw['VA_size_R_num'] = pd.to_numeric(df_raw['VA size_R'], errors='coerce').fillna(-1.0)
    df_raw['VA_size_L_num'] = pd.to_numeric(df_raw['VA size_L'], errors='coerce').fillna(-1.0)
    
    # Reconstruct left and right ears
    records = []
    for _, row in df_raw.iterrows():
        patient_id = str(row['No.']).strip()
        gender = row['gender_num']
        genotype = row['genotype_num']
        age = pd.to_numeric(row['Age at exam'], errors='coerce')
        if pd.isna(age):
            continue
        
        # Right Ear
        r_record = {
            'No': f"{patient_id}_r",
            'Gender': gender,
            'Genotype': genotype,
            'Age': age,
            'AC025': pd.to_numeric(row['R_AC0.25k'], errors='coerce'),
            'AC05': pd.to_numeric(row['R_AC0.5k'], errors='coerce'),
            'AC1': pd.to_numeric(row['R_AC1k'], errors='coerce'),
            'AC2': pd.to_numeric(row['R_AC2k'], errors='coerce'),
            'AC4': pd.to_numeric(row['R_AC4k'], errors='coerce'),
            'AC8': pd.to_numeric(row['R_AC8k'], errors='coerce'),
            'EVA': row['EVA_R_num'],
            'VAsize': row['VA_size_R_num'],
            'mondini': row['Mondini_R_num']
        }
        records.append(r_record)
        
        # Left Ear
        l_record = {
            'No': f"{patient_id}_l",
            'Gender': gender,
            'Genotype': genotype,
            'Age': age,
            'AC025': pd.to_numeric(row['L_AC0.25k'], errors='coerce'),
            'AC05': pd.to_numeric(row['L_AC0.5k'], errors='coerce'),
            'AC1': pd.to_numeric(row['L_AC1k'], errors='coerce'),
            'AC2': pd.to_numeric(row['L_AC2k'], errors='coerce'),
            'AC4': pd.to_numeric(row['L_AC4k'], errors='coerce'),
            'AC8': pd.to_numeric(row['L_AC8k'], errors='coerce'),
            'EVA': row['EVA_L_num'],
            'VAsize': row['VA_size_L_num'],
            'mondini': row['Mondini_L_num']
        }
        records.append(l_record)
        
    df_s3 = pd.DataFrame(records)
    
    # Sort by No and Age
    df_s3 = df_s3.sort_values(by=['No', 'Age']).reset_index(drop=True)
    
    # Fill missing acoustic values per ear group
    ac_cols = ['AC025', 'AC05', 'AC1', 'AC2', 'AC4', 'AC8']
    
    for name, group in df_s3.groupby('No'):
        group_sorted = group.sort_values('Age')
        group_sorted[ac_cols] = group_sorted[ac_cols].interpolate(method='linear', limit_direction='both')
        group_sorted[ac_cols] = group_sorted[ac_cols].ffill().bfill()
        df_s3.loc[group_sorted.index, ac_cols] = group_sorted[ac_cols]
    
    # Fill any remaining NaNs (if an ear has NO valid measurements at all) with column mean or standard fallback
    for col in ac_cols:
        if df_s3[col].isna().any():
            mean_val = df_s3[col].mean()
            if pd.isna(mean_val):
                mean_val = 70.0  # Safe default if column is entirely empty
            df_s3[col] = df_s3[col].fillna(mean_val)
            
    return df_s3


def process_acoustic_data(df: pd.DataFrame, ID2VAsize: dict = None, ID2mondini: dict = None, ID2EVA: dict = None) -> (pd.DataFrame, pd.DataFrame):
    """
    Processes acoustic data from a CSV file, computes various trends and percentages, and maps additional data.
    
    Parameters:
    file_path (str): The path to the CSV file.
    ID2VAsize (dict): A dictionary mapping IDs to 'VA size(mm)'.
    ID2mondini (dict): A dictionary mapping IDs to 'mondini'.
    
    Returns:
    tuple: A tuple containing two DataFrames:
        - data: A DataFrame with grouped and aggregated data.
        - df_new: The modified DataFrame with additional computed columns.
    """
    def determine_trend_category(x):
        if x <= -30:
            return 0
        elif -30 < x <= -15:
            return 1
        elif -15 < x < 0:
            return 2
        elif x == 0:
            return 3
        elif 0 < x < 15:
            return 4
        elif 15 <= x < 30:
            return 5
        elif x >= 30:
            return 6
        else:
            return None
    df = df.groupby("No").filter(lambda x: len(x) >= 4).reset_index(drop=True)
    
    df['Age_diff'] = df.groupby('No')['Age'].diff().fillna(0)
    
    df_new = df.copy()
    
    for col in ['AC025', 'AC05', 'AC1', 'AC2', 'AC4', 'AC8']:
        df_new[col] = df_new[col]

    df_new['four_avg'] = df_new[['AC05', 'AC1', 'AC2', 'AC4']].mean(axis=1)
    
    for col in ['AC025', 'AC05', 'AC1', 'AC2', 'AC4', 'AC8']:
        df_new[f'{col}_trend'] = df_new.groupby('No')[col].diff().fillna(0)
    
    for col in ['AC025_trend', 'AC05_trend', 'AC1_trend', 'AC2_trend', 'AC4_trend', 'AC8_trend']:
        df_new[f'{col}_label'] = df_new[col].apply(determine_trend_category)
    
    for col in ['AC025', 'AC05', 'AC1', 'AC2', 'AC4', 'AC8']:
        df_new[f'{col}_pct'] = df_new.groupby('No')[col].pct_change().fillna(0)
    
    for col in ['AC025', 'AC05', 'AC1', 'AC2', 'AC4', 'AC8']:
        df_new[f'{col}_slope'] = (df_new[f'{col}_trend'] / ((df_new['Age_diff'] // 0.5 + 1)) * 0.5).fillna(0)
    
    for col in ['AC025_trend', 'AC05_trend', 'AC1_trend', 'AC2_trend', 'AC4_trend', 'AC8_trend']:
        df_new[f'{col}_diff'] = df_new.groupby('No')[col].diff().fillna(0)
    
    df_new['avg1'] = df_new[['AC025', 'AC05', 'AC1']].mean(axis=1)
    df_new['avg2'] = df_new[['AC05', 'AC1', 'AC2']].mean(axis=1)
    df_new['avg3'] = df_new[['AC1', 'AC2', 'AC4']].mean(axis=1)
    df_new['avg4'] = df_new[['AC2', 'AC4', 'AC8']].mean(axis=1)

    # Map additional data from dictionaries
    if ID2VAsize is not None and ID2mondini is not None and ID2EVA is not None:
        df_new['VAsize'] = df_new['No'].map(lambda x: ID2VAsize[x.split('#')[0]])
        df_new['mondini'] = df_new['No'].map(lambda x: ID2mondini[x.split('#')[0]])
        df_new['EVA'] = df_new['No'].map(lambda x: ID2EVA[x.split('#')[0]])
    # df_new = df_new[df_new['VAsize'] != -1]
    # df_new = df_new[df_new['mondini'] != -1]

    create_peak_col(df_new)
    create_bigpeak_col(df_new)

    create_peak_agg(df_new)
    create_bigpeak_agg(df_new)

    create_peak_diff(df_new)
    create_bigpeak_diff(df_new)

    create_occur_col(df_new)
    create_recently_peak(df_new)
    calculate_std_within_time(df_new, 0.5)
    calculate_features_within_time(df_new, 0.5)

    # --- Foolproof safety net for NaNs and Infs ---
    df_new = df_new.replace([np.inf, -np.inf], np.nan)
    numeric_cols = df_new.select_dtypes(include=[np.number]).columns
    df_new[numeric_cols] = df_new[numeric_cols].fillna(0.0)
    # -----------------------------------------------

    data = df_new.groupby('No').agg({
        'Age': list,
        'Age_diff': list,
        'Genotype' : list,
        'Gender' : list,
        'AC025': list,
        'AC05': list,
        'AC1': list,
        'AC2': list,
        'AC4': list,
        'AC8': list,
        'AC025_trend': list,
        'AC05_trend': list,
        'AC1_trend': list,
        'AC2_trend': list,
        'AC4_trend': list,
        'AC8_trend': list,
        'AC025_trend_label': list,
        'AC05_trend_label': list,
        'AC1_trend_label': list,
        'AC2_trend_label': list,
        'AC4_trend_label': list,
        'AC8_trend_label': list,
        'AC025_pct': list,
        'AC05_pct': list,
        'AC1_pct': list,
        'AC2_pct': list,
        'AC4_pct': list,
        'AC8_pct': list,
        'AC025_std_in_time': list,
        'AC05_std_in_time': list,
        'AC1_std_in_time': list,
        'AC2_std_in_time': list,
        'AC4_std_in_time': list,
        'AC8_std_in_time': list,
        'AC025_VAR_in_time': list,
        'AC05_VAR_in_time': list,
        'AC1_VAR_in_time': list,
        'AC2_VAR_in_time': list,
        'AC4_VAR_in_time': list,
        'AC8_VAR_in_time': list,
        'AC025_trend_diff': list,
        'AC05_trend_diff': list,
        'AC1_trend_diff': list,
        'AC2_trend_diff': list,
        'AC4_trend_diff': list,
        'AC8_trend_diff': list,
        'avg1': list,
        'avg2': list,
        'avg3': list,
        'avg4': list,
        'four_avg': list,
        'EVA'  : list,
        'VAsize'  : list,
        'mondini' : list,
        'peak': list,
        'peak_agg' : list,
        'peak_diff': list,
        'big_peak' : list,
        'big_peak_agg' : list,
        'big_peak_diff': list,
        'recently_peak' : list,
        'recently_big_peak' : list,
        'recently_total' : list,
        'occur' : list,
    }).reset_index()
    
    return df_new, data

def split_data(data: pd.DataFrame, num_seed: int, only_test: bool) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Splits the data into training, validation, and test sets based on unique IDs.

    Parameters:
    data (pd.DataFrame): The DataFrame containing the data to be split.
    num_seed (int): The seed for random number generator to ensure reproducibility.
    
    Returns:
    tuple: A tuple containing three DataFrames:
        - train_data: The training data.
        - val_data: The validation data.
        - test_data: The test data.
    """
    unique_ids = set()
    for item in data['No'].unique():
        parts = item.split("_")
        unique_ids.add(parts[0])
    
    unique_ids = list(unique_ids)
    unique_ids.sort()
    if only_test:
        train_ids, test_ids = train_test_split(unique_ids, train_size=0.8, random_state=num_seed)

        train_ids = [id_ + "_r" for id_ in train_ids] + [id_ + "_l" for id_ in train_ids]
        test_ids = [id_ + "_r" for id_ in test_ids] + [id_ + "_l" for id_ in test_ids]

        train_data = data[data['No'].isin(train_ids)]
        test_data = data[data['No'].isin(test_ids)]      

        return train_data, test_data
    else:
        train_ids, temp_ids = train_test_split(unique_ids, train_size=0.8, random_state=num_seed)
        test_ids, val_ids  = train_test_split(temp_ids, test_size=0.5, random_state=num_seed)

        train_ids = [id_ + "_r" for id_ in train_ids] + [id_ + "_l" for id_ in train_ids]
        val_ids = [id_ + "_r" for id_ in val_ids] + [id_ + "_l" for id_ in val_ids]
        test_ids = [id_ + "_r" for id_ in test_ids] + [id_ + "_l" for id_ in test_ids]
        
        train_data = data[data['No'].isin(train_ids)]
        val_data = data[data['No'].isin(val_ids)]
        test_data = data[data['No'].isin(test_ids)]
        
        return train_data, val_data, test_data

def create_peak_col(side_df, times = 2):
    side_df['peak'] = 0
    grouped = side_df.groupby("No")
    for idx, group in grouped:
        for i in range(1, len(group)):
            if group.iloc[i]['Age'] - group.iloc[i - 1]['Age'] > 1:
                continue
            cnt = 0
            for ac in ['AC025', 'AC05', 'AC1', 'AC2', 'AC4', 'AC8']:
                if group.iloc[i][ac] - group.iloc[i - 1][ac] >= 15:
                    cnt += 1
            if cnt >= times:
                side_df.loc[group.index[i], 'peak'] = 1
            else:
                side_df.loc[group.index[i], 'peak'] = 0

def create_occur_col(side_df, time_interval = 0.5):
    side_df['occur'] = 0
    grouped = side_df.groupby("No")
    for idx, group in grouped:
        j = 0
        for i in range(len(group)):
            while group.iloc[i]['Age'] - group.iloc[j]['Age'] > time_interval:
                j += 1
            if group.iloc[i]['peak'] == 1 or group.iloc[i]['big_peak'] == 1:
                while j < i:
                    side_df.loc[group.index[j], 'occur'] = 1
                    j += 1

def create_bigpeak_agg(side_df):
    side_df['big_peak_agg'] = 0
    grouped = side_df.groupby("No")
    for _, group in grouped:
        peak_agg = 0
        for i in range(len(group)):
            peak_agg += group.iloc[i]['big_peak']
            side_df.loc[group.index[i], 'big_peak_agg'] = peak_agg

def create_peak_agg(side_df):
    side_df['peak_agg'] = 0
    grouped = side_df.groupby("No")
    for _, group in grouped:
        peak_agg = 0
        for i in range(len(group)):
            peak_agg += group.iloc[i]['peak']
            side_df.loc[group.index[i], 'peak_agg'] = peak_agg

def create_bigpeak_col(side_df):
    side_df['big_peak'] = 0
    grouped = side_df.groupby("No")
    for _, group in grouped:
        for i in range(1, len(group)):
            if group.iloc[i]['Age'] - group.iloc[i - 1]['Age'] > 1:
                continue
            for ac in ['avg1', 'avg2', 'avg3', 'avg4']:
                if group.iloc[i][ac] - group.iloc[i - 1][ac] >= 30:
                    side_df.loc[group.index[i], 'big_peak'] = 1
                    break

def create_peak_diff(side_df):
    side_df['peak_diff'] = -1.0  
    grouped = side_df.groupby("No")
    for idx, group in grouped:
        prev = -1
        for i in range(len(group)):
            side_df.loc[group.index[i], 'peak_diff'] = float(group.iloc[i]['Age'] - prev) if prev != -1 else -1.0
            if group.iloc[i]['peak'] == 1:
                prev = group.iloc[i]['Age']

def create_bigpeak_diff(side_df):
    side_df['big_peak_diff'] = -1.0  
    grouped = side_df.groupby("No")
    for idx, group in grouped:
        prev = -1
        for i in range(len(group)):
            side_df.loc[group.index[i], 'big_peak_diff'] = float(group.iloc[i]['Age'] - prev) if prev != -1 else -1.0
            if group.iloc[i]['big_peak'] == 1:
                prev = group.iloc[i]['Age']

def create_recently_peak(side_df, time = 0.5):
    side_df['recently_peak'] = 0
    side_df['recently_big_peak'] = 0

    grouped = side_df.groupby("No")

    def calculate_recently_peak(group):
        recently_peak = np.zeros(len(group))
        recently_big_peak = np.zeros(len(group))

        for i in range(len(group)):
            valid_indices = (group['Age'] >= group['Age'].iloc[i] - time) & (group['Age'] <= group['Age'].iloc[i])
            recently_peak[i] = group.loc[valid_indices, 'peak'].sum()
            recently_big_peak[i] = group.loc[valid_indices, 'big_peak'].sum()

        return pd.Series(recently_peak, index=group.index), pd.Series(recently_big_peak, index=group.index)

    results = grouped.apply(calculate_recently_peak)
    side_df['recently_peak'] = pd.concat([res[0] for res in results]).sort_index()
    side_df['recently_big_peak'] = pd.concat([res[1] for res in results]).sort_index()

    side_df['recently_total'] = side_df['recently_peak'] + side_df['recently_big_peak']

    return side_df

def calculate_std_within_time(side_df, time=0.5):
    for col in  ['AC025', 'AC05', 'AC1', 'AC2', 'AC4', 'AC8']:
        #fix for new pandas# side_df[f'{col}_std_in_time'] = 0
        side_df[f'{col}_std_in_time'] = np.nan

        grouped = side_df.groupby("No")

        def calculate_std(group, col):
            std_values = np.zeros(len(group))
            
            for i in range(len(group)):
                # Creating a mask to filter the data points within the specified time window
                valid_indices = (group['Age'] >= group['Age'].iloc[i] - time) & (group['Age'] <= group['Age'].iloc[i])
                valid_data = group.loc[valid_indices, col]
                if len(valid_data) > 1:
                    std_values[i] = valid_data.std()
                else:
                    std_values[i] = 0  # Set std to 0 if not enough data points

            return pd.Series(std_values, index=group.index)

        for name, group in grouped:
            side_df.loc[group.index, f'{col}_std_in_time'] = calculate_std(group, col)
    return side_df

def calculate_features_within_time(side_df, time=1):
    features = ['VAR']
    for col in  ['AC025', 'AC05', 'AC1', 'AC2', 'AC4', 'AC8']:
        for feature in features:
            side_df[f'{col}_{feature}_in_time'] = np.nan

        grouped = side_df.groupby("No")

        def calculate_feature(group, col, feature):
            values = np.zeros(len(group))
            
            for i in range(len(group)):
                valid_indices = (group['Age'] >= group['Age'].iloc[i] - time) & (group['Age'] <= group['Age'].iloc[i])
                valid_data = group.loc[valid_indices, col]
                
                if len(valid_data) > 1:
                    if feature == 'VAR':
                        values[i] = np.var(valid_data)
                else:
                    values[i] = 0

            return pd.Series(values, index=group.index)

        for name, group in grouped:
            for feature in features:
                side_df.loc[group.index, f'{col}_{feature}_in_time'] = calculate_feature(group, col, feature)

    return side_df

def plot_ac_values(ax, data, title):
    columns = ['AC025', 'AC05', 'AC1', 'AC2', 'AC4', 'AC8']
    peak_ages = data.loc[data['peak'] == 1, 'Age']
    big_peak_ages = data.loc[data['big_peak'] == 1, 'Age']
    for age in peak_ages:
        ax.axvline(x=age, color='r', linestyle='--')
    
    for age in big_peak_ages:
        ax.axvline(x=age, color='b', linestyle='--')
    
    for col in columns:
        ax.plot(data['Age'], data[col], label=col, marker='o')
    ax.set_xlabel('Age')
    ax.set_ylabel('AC Values')
    ax.set_title(title)
    ax.legend()

def plot_id(df, id):
    id_l = df[df["No"] == f"{id}_l"]
    id_r = df[df["No"] == f"{id}_r"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    plot_ac_values(ax1, id_l, f'AC Values vs Age for No {id_l["No"].values[0]} (Left)')
    plot_ac_values(ax2, id_r, f'AC Values vs Age for No {id_r["No"].values[0]} (Right)')
    plt.tight_layout()
    plt.show()

def plot_all_No(df, No_set):
    for id in No_set:
        print(id)
        plot_id(df, id)

def augment_data(df, train_data, std, edge):
    def generate_normal_change(mean=0, std=std, min_val=-edge, max_val=edge, step=5):
        value = np.random.normal(mean, std)
        value = round(value / step) * step
        return np.clip(value, min_val, max_val)
    ac_columns = ['AC025', 'AC05', 'AC1', 'AC2', 'AC4', 'AC8']
    # Extract the training data subset from the original dataframe
    train_data_subset = df[df['No'].isin(train_data['No'])].reset_index(drop=True)

    # Initialize the list for augmented training data
    augmented_train_data = []

    # Perform data augmentation for each group of 'No'
    for id, group in train_data_subset.groupby('No'):
        augmented_group = []
        for i, row in group.iterrows():
            augmented_row = []
            augmented_row.append(f"{row['No']}#fake")  # Append fake identifier to 'No'
            augmented_row.append(row['Gender'])
            augmented_row.append(row['Genotype'])
            augmented_row.append(row['Age'])
            for col in ac_columns:
                # change = random.choice([-15, -10, -5, 0, 5, 10, 15])  # Randomly choose a change value
                change = generate_normal_change()
                augmented_row.append(max(min(row[col] + change, 125.0), 5.0))
                # augmented_row.append(row[col] + change)
            augmented_group.append(augmented_row)
        augmented_train_data.append(augmented_group)

    augmented_train_df = pd.DataFrame(
        [item for sublist in augmented_train_data for item in sublist], 
        columns=['No', 'Gender', 'Genotype', 'Age'] + ac_columns
    )
    

    return augmented_train_df

def resample(X_train, y_train):
    count_0 = np.sum(y_train == 0)
    count_1 = np.sum(y_train == 1)
    

    if count_0 > count_1:
        drop_rate = 1 - (count_1 / count_0)
    else:
        drop_rate = 0 

    X_train_1 = X_train[y_train == 1]
    y_train_1 = y_train[y_train == 1]
    
    mask = y_train == 0
    X_train_0 = X_train[mask]
    y_train_0 = y_train[mask]
    
    drop_mask = np.random.rand(len(y_train_0)) > drop_rate
    X_train_0_resampled = X_train_0[drop_mask]
    y_train_0_resampled = y_train_0[drop_mask]
    
    X_train_resampled = np.concatenate((X_train_1, X_train_0_resampled), axis=0)
    y_train_resampled = np.concatenate((y_train_1, y_train_0_resampled), axis=0)
    
    return X_train_resampled, y_train_resampled


def parse_mae_line(line, model_name):
    """
    Parses a line from the log file to extract MAE values and mean.

    Parameters:
    line (str): A line from the log file.
    model_name (str): The name of the model.

    Returns:
    list: A list containing the model name, MAE values, and the mean MAE.
    """
    mae_data = re.search(r'MAE = \[(.*?)\], mean: (\d+\.\d+)', line)
    if mae_data:
        mae_values = list(map(float, re.split(r',\s*|\s+', mae_data.group(1).lstrip().rstrip())))
        mean_value = float(mae_data.group(2))
        return [model_name] + mae_values + [mean_value]
    return None

def read_log_file(file_path, model_name_pattern):
    """
    Reads a log file and processes each line to extract MAE data.

    Parameters:
    file_path (str): The path to the log file.
    model_name_pattern (dict): A dictionary mapping substrings in the log lines to model names.

    Returns:
    list: A list of lists, each containing the model name, MAE values, and the mean MAE.
    """
    data = []
    with open(file_path, 'r') as log_file:
        lines = log_file.readlines()
        for line in lines:
            line = line.strip()
            for model_name in model_name_pattern.keys():
                if model_name in line:
                    parsed_data = parse_mae_line(line, model_name_pattern[model_name])
                    if parsed_data:
                        data.append(parsed_data)
                    break
    return data

def calculate_mean_std(dataframe, columns):
    """
    Calculates the mean and standard deviation for specified columns in a DataFrame.

    Parameters:
    dataframe (pd.DataFrame): The DataFrame containing the data.
    columns (list): A list of column names to calculate mean and standard deviation for.

    Returns:
    pd.Series: A Series with mean ± standard deviation for each specified column.
    """
    mean = dataframe[columns].mean()
    std_dev = dataframe[columns].std()
    return mean.apply(lambda x: f"{x:.2f}") + " ± " + std_dev.apply(lambda x: f"{x:.2f}")

def process_log_files(log_files):
    """
    Processes multiple log files to extract and summarize MAE data.

    Parameters:
    log_files (dict): A dictionary mapping log file paths to model name patterns.

    Returns:
    pd.DataFrame: A DataFrame containing the summarized MAE data for each model.
    """
    all_data = []
    for log_file, pattern in log_files.items():
        all_data.extend(read_log_file(log_file, pattern))

    columns = ['Model', 'MAE1', 'MAE2', 'MAE3', 'MAE4', 'MAE5', 'MAE6', 'Mean']
    df = pd.DataFrame(all_data, columns=columns)

    for col in columns[1:]:
        df[col] = df[col].astype(float)

    return df

def train_epoch(model, optimizer, criterion, train_loader, ac_offset ,device = 'cuda'):
    model.train()
    train_loss = 0.0
    for X, Y in train_loader:
        X, Y = X.float().to(device), Y.float().to(device)
        optimizer.zero_grad()
        outputs = model(X, Y[:, 0], Y[:, 1])
        loss = criterion(outputs, Y[:, ac_offset : ac_offset + 6])
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss /= len(train_loader)
    return train_loss

# Evaluation function
def evaluate(model, criterion, eval_loader, ac_offset, device = 'cuda'):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in eval_loader:
            inputs, targets = inputs.float().to(device), targets.float().to(device)            
            outputs = model(inputs, targets[:, 0], targets[:, 1])
            loss = criterion(outputs, targets[:, ac_offset : ac_offset + 6])
            val_loss += loss.item()
    val_loss /= len(eval_loader)
    return val_loss

def predict(model, data, device = 'cuda'):
    model.eval()
    age = data[:, -1, 0].float().to(device)
    step = torch.tensor(np.full(data.shape[0], 0.5)).float().to(device)
    with torch.no_grad():
        outputs = model(data, age, step)
    return outputs


def k_folder_split_data(data: pd.DataFrame, num_seed: int, k = 5, val = False) -> list:
    unique_ids = set()
    for item in data['No'].unique():
        parts = item.split("_")
        unique_ids.add(parts[0])
    
    unique_ids = list(unique_ids)
    unique_ids.sort()
    
    kf = KFold(n_splits=k, shuffle=True, random_state=num_seed)
    
    fold_data = []
    for train_val_index, test_index in kf.split(unique_ids):
        train_val_ids = [unique_ids[i] for i in train_val_index]
        test_ids = [unique_ids[i] for i in test_index]
        
        if val:
            # Split patient IDs before reconstructing left/right ear suffixes to prevent leakage
            train_ids, val_ids = train_test_split(train_val_ids, test_size=0.1, random_state=num_seed)
            train_ids = [id_ + "_r" for id_ in train_ids] + [id_ + "_l" for id_ in train_ids]
            val_ids = [id_ + "_r" for id_ in val_ids] + [id_ + "_l" for id_ in val_ids]
            test_ids = [id_ + "_r" for id_ in test_ids] + [id_ + "_l" for id_ in test_ids]
            
            train_data = data[data['No'].isin(train_ids)]
            val_data = data[data['No'].isin(val_ids)]
            test_data = data[data['No'].isin(test_ids)]
            fold_data.append((train_data, val_data, test_data))
        else:
            train_val_ids = [id_ + "_r" for id_ in train_val_ids] + [id_ + "_l" for id_ in train_val_ids]
            test_ids = [id_ + "_r" for id_ in test_ids] + [id_ + "_l" for id_ in test_ids]
            
            train_data = data[data['No'].isin(train_val_ids)]
            test_data = data[data['No'].isin(test_ids)]
            fold_data.append((train_data, test_data))
        
    return fold_data