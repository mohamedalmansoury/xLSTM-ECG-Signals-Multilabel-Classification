import os
import ast
import numpy as np
import pandas as pd
import wfdb
from typing import Tuple, List, Dict


SUPERCLASS_NAMES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']


def load_ptbxl_metadata(data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ptbxl = pd.read_csv(os.path.join(data_path, 'ptbxl_database.csv'))
    scp_statements = pd.read_csv(os.path.join(data_path, 'scp_statements.csv'))
    scp_statements.set_index('Unnamed: 0', inplace=True)
    
    return ptbxl, scp_statements


def extract_labels(code_string: str, scp_statements: pd.DataFrame) -> Dict[str, int]:
    parsed_dict = ast.literal_eval(code_string)
    result_dict = {name: 0 for name in SUPERCLASS_NAMES}
    
    for code in parsed_dict.keys():
        if code in scp_statements.index:
            superclass_name = scp_statements.loc[code].diagnostic_class
            if superclass_name in SUPERCLASS_NAMES:
                result_dict[superclass_name] = 1 if parsed_dict[code] > 0.0 else 0
    
    return result_dict


def load_ptbxl_signal(record_path: str, sampling_rate: int = 100) -> np.ndarray:
    signal, _ = wfdb.rdsamp(record_path)
    return signal.astype(np.float32)


def prepare_ptbxl_dataset(data_path: str, sampling_rate: int = 100) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ptbxl, scp_statements = load_ptbxl_metadata(data_path)
    
    expanded_codes = ptbxl['scp_codes'].apply(
        lambda x: extract_labels(x, scp_statements)
    )
    expanded_codes = pd.json_normalize(expanded_codes)
    ptbxl = pd.concat([ptbxl, expanded_codes], axis=1)
    
    non_zero_mask = (ptbxl[SUPERCLASS_NAMES].sum(axis=1) > 0)
    ptbxl = ptbxl[non_zero_mask].reset_index(drop=True)
    
    ptbxl['filename_full'] = ptbxl['filename_lr'] if sampling_rate == 100 else ptbxl['filename_hr']
    ptbxl['filename_full'] = ptbxl['filename_full'].apply(
        lambda x: os.path.join(data_path, x)
    )
    
    return ptbxl, scp_statements


def load_wfdb_record(dat_file: str, hea_file: str = None) -> np.ndarray:
    record_name = dat_file.replace('.dat', '')
    signal, fields = wfdb.rdsamp(record_name)
    return signal.astype(np.float32)


def stratified_split(df: pd.DataFrame, 
                     label_columns: List[str],
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.15,
                     random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split
    
    df['strat_key'] = df[label_columns].apply(lambda x: ''.join(x.astype(str)), axis=1)
    
    train_df, temp_df = train_test_split(
        df, 
        test_size=1-train_ratio, 
        stratify=df['strat_key'],
        random_state=random_state
    )
    
    val_ratio_adjusted = val_ratio / (1 - train_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=1-val_ratio_adjusted,
        stratify=temp_df['strat_key'],
        random_state=random_state
    )
    
    train_df = train_df.drop('strat_key', axis=1)
    val_df = val_df.drop('strat_key', axis=1)
    test_df = test_df.drop('strat_key', axis=1)
    
    return train_df, val_df, test_df


def batch_load_signals(record_paths: List[str]) -> np.ndarray:
    signals = []
    for path in record_paths:
        signal, _ = wfdb.rdsamp(path)
        signals.append(signal)
    return np.array(signals, dtype=np.float32)


class ECGDataset:
    
    def __init__(self, df: pd.DataFrame, label_columns: List[str], 
                 data_path: str = None):
        self.df = df.reset_index(drop=True)
        self.label_columns = label_columns
        self.data_path = data_path
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        row = self.df.iloc[idx]
        
        if 'filename_full' in row:
            signal_path = row['filename_full'].replace('.hea', '')
        else:
            signal_path = os.path.join(self.data_path, row['filename_lr'])
        
        signal, _ = wfdb.rdsamp(signal_path)
        
        age = row['age']
        sex = 1 if row['sex'] == 1 else 0
        metadata = np.array([age, sex], dtype=np.float32)
        
        labels = row[self.label_columns].values.astype(np.float32)
        
        return signal.astype(np.float32), metadata, labels
