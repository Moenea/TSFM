import os
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class UnivariateDatasetBenchmark(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv', scale=True, nonautoregressive=False, test_flag='T', subset_rand_ratio=1.0):
        self.seq_len = size[0]
        self.input_token_len = size[1]
        self.output_token_len = size[2]
        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.root_path = root_path
        self.data_path = data_path
        self.data_type = data_path.split('.')[0]
        self.scale = scale
        self.nonautoregressive = nonautoregressive
        self.subset_rand_ratio = subset_rand_ratio
        if self.set_type == 0:
            self.internal = int(1 // self.subset_rand_ratio)
        else:
            self.internal = 1
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        dataset_file_path = os.path.join(self.root_path, self.data_path)
        if dataset_file_path.endswith('.csv'):
            df_raw = pd.read_csv(dataset_file_path)
        elif dataset_file_path.endswith('.txt'):
            df_raw = []
            with open(dataset_file_path, "r", encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip('\n').split(',')
                    data_line = np.stack([float(i) for i in line])
                    df_raw.append(data_line)
            df_raw = np.stack(df_raw, 0)
            df_raw = pd.DataFrame(df_raw)
        elif dataset_file_path.endswith('.npz'):
            data = np.load(dataset_file_path, allow_pickle=True)
            data = data['data'][:, :, 0]
            df_raw = pd.DataFrame(data)
        elif dataset_file_path.endswith('.npy'):
            data = np.load(dataset_file_path)
            df_raw = pd.DataFrame(data)
        else:
            raise ValueError('Unknown data format: {}'.format(dataset_file_path))

        if self.data_type == 'ETTh' or self.data_type == 'ETTh1' or self.data_type == 'ETTh2':
            border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        elif self.data_type == 'ETTm' or self.data_type == 'ETTm1' or self.data_type == 'ETTm2':
            border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
            border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        else:
            data_len = len(df_raw)
            num_train = int(data_len * 0.7)
            num_test = int(data_len * 0.2)
            num_vali = data_len - num_train - num_test
            border1s = [0, num_train - self.seq_len, data_len - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, data_len]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if isinstance(df_raw[df_raw.columns[0]][2], str):
            data = df_raw[df_raw.columns[1:]].values
        else:
            data = df_raw.values

        if self.scale:
            train_data = data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        
        self.n_var = self.data_x.shape[-1]
        self.n_timepoint =  len(self.data_x) - self.seq_len - self.output_token_len + 1
        
    def __getitem__(self, index):
        feat_id = index // self.n_timepoint
        s_begin = index % self.n_timepoint
        s_end = s_begin + self.seq_len

        if not self.nonautoregressive:
            r_begin = s_begin + self.input_token_len
            r_end = s_end + self.output_token_len

            seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
            seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
            seq_y = torch.tensor(seq_y)
            seq_y = seq_y.unfold(dimension=0, size=self.output_token_len,
                                 step=self.input_token_len).permute(0, 2, 1)
            seq_y = seq_y.reshape(seq_y.shape[0] * seq_y.shape[1], -1)
        else:
            r_begin = s_end
            r_end = r_begin + self.output_token_len
            seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
            seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))
                
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if self.set_type == 0:
            return max(int(self.n_var * self.n_timepoint * self.subset_rand_ratio), 1)
        else:
            return int(self.n_var * self.n_timepoint)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class MultivariateDatasetBenchmark(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv', scale=True, nonautoregressive=False, test_flag='T', subset_rand_ratio=1.0):
        self.seq_len = size[0]
        self.input_token_len = size[1]
        self.output_token_len = size[2]
        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.root_path = root_path
        self.data_path = data_path
        self.data_type = data_path.split('.')[0]
        self.scale = scale
        self.nonautoregressive = nonautoregressive
        self.subset_rand_ratio = subset_rand_ratio
        if self.set_type == 0:
            self.internal = int(1 // self.subset_rand_ratio)
        else:
            self.internal = 1
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        dataset_file_path = os.path.join(self.root_path, self.data_path)
        if dataset_file_path.endswith('.csv'):
            df_raw = pd.read_csv(dataset_file_path)
        elif dataset_file_path.endswith('.txt'):
            df_raw = []
            with open(dataset_file_path, "r", encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip('\n').split(',')
                    data_line = np.stack([float(i) for i in line])
                    df_raw.append(data_line)
            df_raw = np.stack(df_raw, 0)
            df_raw = pd.DataFrame(df_raw)
        elif dataset_file_path.endswith('.npz'):
            data = np.load(dataset_file_path, allow_pickle=True)
            data = data['data'][:, :, 0]
            df_raw = pd.DataFrame(data)
        elif dataset_file_path.endswith('.npy'):
            data = np.load(dataset_file_path)
            df_raw = pd.DataFrame(data)
        else:
            raise ValueError('Unknown data format: {}'.format(dataset_file_path))

        if self.data_type == 'ETTh' or self.data_type == 'ETTh1' or self.data_type == 'ETTh2':
            border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        elif self.data_type == 'ETTm' or self.data_type == 'ETTm1' or self.data_type == 'ETTm2':
            border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
            border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        else:
            data_len = len(df_raw)
            num_train = int(data_len * 0.7)
            num_test = int(data_len * 0.2)
            num_vali = data_len - num_train - num_test
            border1s = [0, num_train - self.seq_len, data_len - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, data_len]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if isinstance(df_raw[df_raw.columns[0]][2], str):
            data = df_raw[df_raw.columns[1:]].values
        else:
            data = df_raw.values

        if self.scale:
            train_data = data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        
        self.n_var = self.data_x.shape[-1]
        self.n_timepoint =  len(self.data_x) - self.seq_len - self.output_token_len + 1
        
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len

        if not self.nonautoregressive:
            r_begin = s_begin + self.input_token_len
            r_end = s_end + self.output_token_len
            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
            seq_y = torch.tensor(seq_y)
            seq_y = seq_y.unfold(dimension=0, size=self.output_token_len,
                                 step=self.input_token_len).permute(0, 2, 1)
            seq_y = seq_y.reshape(seq_y.shape[0] * seq_y.shape[1], -1)
        else:
            r_begin = s_end
            r_end = r_begin + self.output_token_len
            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))
            
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if self.set_type == 0:
            return max(int(self.n_timepoint * self.subset_rand_ratio), 1)
        else:
            return self.n_timepoint

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class MultivariateDatasetYAMLSplit(MultivariateDatasetBenchmark):
    """
    Forecasting dataset that loads train/val/test from SEPARATE CSV files
    declared in a YAML split file. Inherits patch-based unfold logic from
    :class:`MultivariateDatasetBenchmark`.

    YAML format:
        target: <column name>      # required for features='S'; else optional doc
        train:
          - relative/path/to/train1.csv
        val:
          - relative/path/to/val1.csv
        test:
          - relative/path/to/test1.csv

    Paths inside the YAML are resolved relative to ``root_path``.

    Three task modes (selected via the ``features`` argument, normally
    forwarded from CLI ``--features``):

      * ``features='M'``  — load all columns. Used for M (M->M) and Co modes.
                            For Co the model also needs ``--covariate``.
      * ``features='MS'`` — load all columns; identical data shape as 'M'.
                            Distinction is in downstream loss/metric extraction.
      * ``features='S'``  — load ONLY the target column. Used for U mode
                            (univariate self-prediction). Requires the YAML
                            to declare ``target``.

    Design notes:
      * Each CSV file is kept as a separate array element so sliding windows
        never cross file boundaries (mirrors MyTimeXer's Dataset_Custom).
      * StandardScaler is fit on the concatenation of all train files
        (post-slicing — so 'S' mode fits a 1D scaler).
    """

    def __init__(self, root_path, flag='train', size=None, data_path='splits.yaml',
                 scale=True, nonautoregressive=False, test_flag='T',
                 subset_rand_ratio=1.0, split_file=None, features='M', target=None):
        self.split_file = split_file if split_file else os.path.join(root_path, data_path)
        self.features = features
        self.target_override = target
        super().__init__(
            root_path=root_path, flag=flag, size=size, data_path=data_path,
            scale=scale, nonautoregressive=nonautoregressive,
            test_flag=test_flag, subset_rand_ratio=subset_rand_ratio,
        )

    def _load_csv_df(self, rel_path):
        """Load a CSV, drop a leading string (timestamp) column if present."""
        full = os.path.join(self.root_path, rel_path)
        df = pd.read_csv(full)
        if len(df) > 0 and isinstance(df[df.columns[0]].iloc[0], str):
            df = df[df.columns[1:]]
        return df

    def _slice_target(self, df):
        """Slice DataFrame to the target column when features='S'."""
        if self.features != 'S':
            return df
        if self.target_name is None:
            raise ValueError(
                "features='S' requires a target column. Set 'target: <name>' "
                f"in {self.split_file} or pass --target via CLI."
            )
        if self.target_name not in df.columns:
            # Fall back to last column if target name not found, but warn loudly.
            raise ValueError(
                f"Target column '{self.target_name}' not found in CSV. "
                f"Available columns: {list(df.columns)}"
            )
        return df[[self.target_name]]

    def __read_data__(self):
        self.scaler = StandardScaler()

        with open(self.split_file, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}

        train_files = cfg.get('train', []) or []
        val_files = cfg.get('val', []) or []
        test_files = cfg.get('test', []) or []
        flag_files_map = {'train': train_files, 'val': val_files, 'test': test_files}
        flag_files = flag_files_map[self.flag]
        if not flag_files:
            raise ValueError(
                f"YAML split file {self.split_file} has no entries for flag '{self.flag}'"
            )
        if not train_files:
            raise ValueError(
                f"YAML split file {self.split_file} has no 'train' files; "
                f"cannot fit StandardScaler"
            )

        # Resolve target name: CLI override > YAML > None.
        self.target_name = self.target_override or cfg.get('target', None)

        if self.scale:
            train_chunks = [
                self._slice_target(self._load_csv_df(p)).values.astype(np.float32)
                for p in train_files
            ]
            self.scaler.fit(np.concatenate(train_chunks, axis=0))

        self.data_x = []
        self.data_y = []
        self._lengths = []
        n_var_ref = None
        for p in flag_files:
            arr = self._slice_target(self._load_csv_df(p)).values.astype(np.float32)
            if n_var_ref is None:
                n_var_ref = arr.shape[-1]
            elif arr.shape[-1] != n_var_ref:
                raise ValueError(
                    f"Inconsistent #columns: file {p} has {arr.shape[-1]} cols, "
                    f"expected {n_var_ref}"
                )
            if self.scale:
                arr = self.scaler.transform(arr).astype(np.float32)
            self.data_x.append(arr)
            self.data_y.append(arr)
            usable = len(arr) - self.seq_len - self.output_token_len + 1
            self._lengths.append(max(0, usable))

        self.n_var = n_var_ref
        self.n_timepoint = sum(self._lengths)
        if self.n_timepoint == 0:
            raise ValueError(
                f"No usable windows for flag '{self.flag}': every file is shorter "
                f"than seq_len={self.seq_len} + output_token_len={self.output_token_len}"
            )

    def _locate(self, index):
        idx = index
        for file_idx, length in enumerate(self._lengths):
            if idx < length:
                return file_idx, idx
            idx -= length
        raise IndexError(index)

    def __getitem__(self, index):
        file_idx, local_idx = self._locate(index)
        data_x_arr = self.data_x[file_idx]
        data_y_arr = self.data_y[file_idx]

        s_begin = local_idx
        s_end = s_begin + self.seq_len

        # Patch-based unfold logic mirrored verbatim from
        # MultivariateDatasetBenchmark.__getitem__ — only the source array changes.
        if not self.nonautoregressive:
            r_begin = s_begin + self.input_token_len
            r_end = s_end + self.output_token_len
            seq_x = data_x_arr[s_begin:s_end]
            seq_y = data_y_arr[r_begin:r_end]
            seq_y = torch.tensor(seq_y)
            seq_y = seq_y.unfold(dimension=0, size=self.output_token_len,
                                 step=self.input_token_len).permute(0, 2, 1)
            seq_y = seq_y.reshape(seq_y.shape[0] * seq_y.shape[1], -1)
        else:
            r_begin = s_end
            r_end = r_begin + self.output_token_len
            seq_x = data_x_arr[s_begin:s_end]
            seq_y = data_y_arr[r_begin:r_end]

        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if self.set_type == 0:
            return max(int(self.n_timepoint * self.subset_rand_ratio), 1)
        return self.n_timepoint


class Global_Temp(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv', scale=True, nonautoregressive=False, test_flag='T', subset_rand_ratio=1.0):
        self.seq_len = size[0]
        self.input_token_len = size[1]
        self.output_token_len = size[2]
        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.root_path = root_path
        self.data_path = data_path
        self.scale = scale
        self.nonautoregressive = nonautoregressive
        self.__read_data__()

    def __read_data__(self):
        self.raw_data = np.load(os.path.join(self.root_path,
                                             "temp_global_hourly_" + self.flag + ".npy"),
                                allow_pickle=True)
        raw_data = self.raw_data
        data_len, station, feat = raw_data.shape
        raw_data = raw_data.reshape(data_len, station * feat)
        data = raw_data.astype(np.float)

        self.data_x = data
        self.data_y = data

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len

        if not self.nonautoregressive:
            r_begin = s_begin + self.input_token_len
            r_end = s_end + self.output_token_len

            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
            seq_y = torch.tensor(seq_y)
            seq_y = seq_y.unfold(dimension=0, size=self.output_token_len,
                                 step=self.input_token_len).permute(0, 2, 1)
            seq_y = seq_y.reshape(seq_y.shape[0] * seq_y.shape[1], -1)
        else:
            r_begin = s_end
            r_end = r_begin + self.output_token_len
            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.output_token_len + 1


class Global_Wind(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv', scale=True, nonautoregressive=False, test_flag='T', subset_rand_ratio=1.0):
        self.seq_len = size[0]
        self.input_token_len = size[1]
        self.output_token_len = size[2]
        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.root_path = root_path
        self.data_path = data_path
        self.scale = scale
        self.nonautoregressive = nonautoregressive
        self.__read_data__()

    def __read_data__(self):
        self.raw_data = np.load(os.path.join(self.root_path,
                                             "wind_global_hourly_" + self.flag + ".npy"),
                                allow_pickle=True)
        raw_data = self.raw_data
        data_len, station, feat = raw_data.shape
        raw_data = raw_data.reshape(data_len, station * feat)
        data = raw_data.astype(np.float)

        self.data_x = data
        self.data_y = data

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len

        if not self.nonautoregressive:
            r_begin = s_begin + self.input_token_len
            r_end = s_end + self.output_token_len

            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
            seq_y = torch.tensor(seq_y)
            seq_y = seq_y.unfold(dimension=0, size=self.output_token_len,
                                 step=self.input_token_len).permute(0, 2, 1)
            seq_y = seq_y.reshape(seq_y.shape[0] * seq_y.shape[1], -1)
        else:
            r_begin = s_end
            r_end = r_begin + self.output_token_len
            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.output_token_len + 1


class Dataset_ERA5_Pretrain(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv', scale=True, nonautoregressive=False, test_flag='T', subset_rand_ratio=1.0):
        self.seq_len = size[0]
        self.input_token_len = size[1]
        self.output_token_len = size[2]
        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.root_path = root_path
        self.data_path = data_path
        self.scale = scale
        self.nonautoregressive = nonautoregressive
        self.__read_data__()
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - \
            self.output_token_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = np.load(os.path.join(self.root_path, self.data_path))
        # split only the train set
        L, S = df_raw.shape
        Train_S = int(S * 0.8)
        df_raw = df_raw[:, :Train_S]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len,
                    len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        s_end = s_begin + self.seq_len

        if not self.nonautoregressive:
            r_begin = s_begin + self.input_token_len
            r_end = s_end + self.output_token_len

            seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
            seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
            seq_y = torch.tensor(seq_y)
            seq_y = seq_y.unfold(dimension=0, size=self.output_token_len,
                                 step=self.input_token_len).permute(0, 2, 1)
            seq_y = seq_y.reshape(seq_y.shape[0] * seq_y.shape[1], -1)
        else:
            r_begin = s_end
            r_end = r_begin + self.output_token_len
            seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
            seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.output_token_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ERA5_Pretrain_Test(Dataset):
    def __init__(self, root_path, flag='test', size=None, data_path='ETTh1.csv', scale=True, nonautoregressive=False, test_flag='T', subset_rand_ratio=1.0):
        self.seq_len = size[0]
        self.input_token_len = size[1]
        self.output_token_len = size[2]
        self.test_flag = test_flag
        assert test_flag in ['T', 'V', 'TandV']
        type_map = {'T': 0, 'V': 1, 'TandV': 2}
        self.test_type = type_map[flag]
        self.root_path = root_path
        self.data_path = data_path
        self.scale = scale
        self.nonautoregressive = nonautoregressive
        self.__read_data__()
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - \
            self.output_token_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = np.load(os.path.join(self.root_path, self.data_path))
        # split only the train set
        L, S = df_raw.shape
        if self.test_type == 0:
            Train_S = int(S * 0.8)
            df_raw = df_raw[:, :Train_S]
            num_train = int(len(df_raw) * 0.7)
            num_test = int(len(df_raw) * 0.2)
            num_vali = len(df_raw) - num_train - num_test
            border1s = [0, num_train - self.seq_len,
                        len(df_raw) - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, len(df_raw)]
            data = df_raw
            border1 = border1s[-1]
            border2 = border2s[-1]

            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]
        else:
            Train_S = int(S * 0.8)
            df_raw = df_raw[:, Train_S:]
            num_train = int(len(df_raw) * 0.8)
            num_test = len(df_raw) - num_train
            border1s = [0, len(df_raw) - num_test - self.seq_len]
            border2s = [num_train, len(df_raw)]
            data = df_raw
            if self.test_type == 1:
                border1 = border1s[0]
                border2 = border2s[0]
            else:
                border1 = border1s[1]
                border2 = border2s[1]

            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        s_end = s_begin + self.seq_len

        if not self.nonautoregressive:
            r_begin = s_begin + self.input_token_len
            r_end = s_end + self.output_token_len

            seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
            seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
            seq_y = torch.tensor(seq_y)
            seq_y = seq_y.unfold(dimension=0, size=self.output_token_len,
                                 step=self.input_token_len).permute(0, 2, 1)
            seq_y = seq_y.reshape(seq_y.shape[0] * seq_y.shape[1], -1)
        else:
            r_begin = s_end
            r_end = r_begin + self.output_token_len
            seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
            seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.output_token_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# Download link: https://huggingface.co/datasets/thuml/UTSD
class UTSD(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv', scale=True, nonautoregressive=False, stride=1, split=0.9, test_flag='T', subset_rand_ratio=1.0):
        self.seq_len = size[0]
        self.input_token_len = size[1]
        self.output_token_len = size[2]
        self.context_len = self.seq_len + self.output_token_len
        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.scale = scale
        self.nonautoregressive = nonautoregressive
        self.split = split
        self.stride = stride
        self.data_list = []
        self.n_window_list = []
        self.root_path = root_path
        self.__confirm_data__()

    def __confirm_data__(self):
        for root, dirs, files in os.walk(self.root_path):
            for file in files:
                if file.endswith('.csv'):
                    dataset_path = os.path.join(root, file)

                    self.scaler = StandardScaler()
                    df_raw = pd.read_csv(dataset_path)

                    if isinstance(df_raw[df_raw.columns[0]][0], str):
                        data = df_raw[df_raw.columns[1:]].values
                    else:
                        data = df_raw.values

                    num_train = int(len(data) * self.split)
                    num_test = int(len(data) * (1 - self.split) / 2)
                    num_vali = len(data) - num_train - num_test
                    if num_train < self.context_len:
                        continue
                    border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
                    border2s = [num_train, num_train + num_vali, len(data)]

                    border1 = border1s[self.set_type]
                    border2 = border2s[self.set_type]

                    if self.scale:
                        train_data = data[border1s[0]:border2s[0]]
                        self.scaler.fit(train_data)
                        data = self.scaler.transform(data)
                    else:
                        data = data

                    data = data[border1:border2]
                    n_timepoint = (
                        len(data) - self.context_len) // self.stride + 1
                    n_var = data.shape[1]
                    self.data_list.append(data)
                    n_window = n_timepoint * n_var
                    self.n_window_list.append(n_window if len(
                        self.n_window_list) == 0 else self.n_window_list[-1] + n_window)
        print("Total number of windows in merged dataset: ",
              self.n_window_list[-1])

    def __getitem__(self, index):
        assert index >= 0
        # find the location of one dataset by the index
        dataset_index = 0
        while index >= self.n_window_list[dataset_index]:
            dataset_index += 1

        index = index - \
            self.n_window_list[dataset_index -
                               1] if dataset_index > 0 else index
        n_timepoint = (
            len(self.data_list[dataset_index]) - self.context_len) // self.stride + 1

        c_begin = index // n_timepoint  # select variable
        s_begin = index % n_timepoint  # select start timestamp
        s_begin = self.stride * s_begin
        s_end = s_begin + self.seq_len
        r_begin = s_begin + self.input_token_len
        r_end = s_end + self.output_token_len

        seq_x = self.data_list[dataset_index][s_begin:s_end,
                                              c_begin:c_begin + 1]
        seq_y = self.data_list[dataset_index][r_begin:r_end,
                                              c_begin:c_begin + 1]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return self.n_window_list[-1]


# Download link: https://cloud.tsinghua.edu.cn/f/93868e3a9fb144fe9719/
class UTSD_Npy(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv', scale=True, nonautoregressive=False, stride=1, split=0.9, test_flag='T', subset_rand_ratio=1.0):
        self.seq_len = size[0]
        self.input_token_len = size[1]
        self.output_token_len = size[2]
        self.context_len = self.seq_len + self.output_token_len
        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.scale = scale
        self.root_path = root_path
        self.nonautoregressive = nonautoregressive
        self.split = split
        self.stride = stride
        self.data_list = []
        self.n_window_list = []
        self.__confirm_data__()

    def __confirm_data__(self):
        for root, dirs, files in os.walk(self.root_path):
            for file in files:
                if file.endswith('.npy'):
                    dataset_path = os.path.join(root, file)

                    self.scaler = StandardScaler()
                    data = np.load(dataset_path)

                    num_train = int(len(data) * self.split)
                    num_test = int(len(data) * (1 - self.split) / 2)
                    num_vali = len(data) - num_train - num_test
                    if num_train < self.context_len:
                        continue
                    border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
                    border2s = [num_train, num_train + num_vali, len(data)]

                    border1 = border1s[self.set_type]
                    border2 = border2s[self.set_type]

                    if self.scale:
                        train_data = data[border1s[0]:border2s[0]]
                        self.scaler.fit(train_data)
                        data = self.scaler.transform(data)
                    else:
                        data = data

                    data = data[border1:border2]
                    n_timepoint = (
                        len(data) - self.context_len) // self.stride + 1
                    n_var = data.shape[1]
                    self.data_list.append(data)

                    n_window = n_timepoint * n_var
                    self.n_window_list.append(n_window if len(
                        self.n_window_list) == 0 else self.n_window_list[-1] + n_window)
        print("Total number of windows in merged dataset: ",
              self.n_window_list[-1])

    def __getitem__(self, index):
        assert index >= 0
        # find the location of one dataset by the index
        dataset_index = 0
        while index >= self.n_window_list[dataset_index]:
            dataset_index += 1

        index = index - \
            self.n_window_list[dataset_index -
                               1] if dataset_index > 0 else index
        n_timepoint = (
            len(self.data_list[dataset_index]) - self.context_len) // self.stride + 1

        c_begin = index // n_timepoint  # select variable
        s_begin = index % n_timepoint  # select start timestamp
        s_begin = self.stride * s_begin
        s_end = s_begin + self.seq_len
        r_begin = s_begin + self.input_token_len
        r_end = s_end + self.output_token_len

        seq_x = self.data_list[dataset_index][s_begin:s_end,
                                              c_begin:c_begin + 1]
        seq_y = self.data_list[dataset_index][r_begin:r_end,
                                              c_begin:c_begin + 1]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return self.n_window_list[-1]
