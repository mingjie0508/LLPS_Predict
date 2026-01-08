# PyTorch style dataset classes for LLPS prediction
import torch
import pandas as pd


class LLPSConditionsDataset(torch.utils.data.Dataset):
    """
    Dataset class for sequence and environmental conditions data (input) and labels (ground truth output).
    e.g., (sequence, (temperature, solute conc, salt conc, pH, crowding agent), label)
    """
    def __init__(self, df_path: str, sqn_column: str, cond_columns: list, label_column: str, 
                 seq_transform=None, cond_transform=None):
        df = pd.read_csv(df_path)
        self.sequences = df[sqn_column]
        self.conditions = df[cond_columns]
        self.labels = df[label_column]
        self.seq_transform = seq_transform
        self.cond_transform = cond_transform
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, index):
        seq = self.sequences.iloc[index]
        cond = self.conditions.iloc[index,:]
        label = self.labels.iloc[index]
        if self.seq_transform is not None:
            seq = self.seq_transform(seq)
        if self.cond_transform is not None:
            cond = self.cond_transform(cond)
        cond = torch.Tensor(cond.to_list())
        return seq, cond, label


if __name__ == '__main__':
    # command:
    # cd LLPS_Predict
    # python -m src.loaders.dataset_condition

    # transform experimental conditions
    from src.utils.experimental_condition import experimental_condition_transform_factory

    # input config
    LLPS_DATA_PATH = 'data/experimental_condition/LLPSDB2_phase_diagrams3.csv'
    CONDITION_COLUMNS = [
        'Temperature', 
        'SoluteConcentration', 
        'SaltConcSum', 
        'BufferpH', 
        'CrowdingAgent'
    ]
    SAMPLE_COLUMN = 'Column'
    RANGE_LOW_COLUMN = 'RangeLow'
    RANGE_HIGH_COLUMN = 'RangeHigh'

    # transform conditions, randomly sample one of the conditions within range
    cond_transform = experimental_condition_transform_factory(
        cond_columns=CONDITION_COLUMNS, 
        sample_column=SAMPLE_COLUMN, 
        low_column=RANGE_LOW_COLUMN, 
        high_column=RANGE_HIGH_COLUMN,
    )

    dataset = LLPSConditionsDataset(
        LLPS_DATA_PATH, 
        sqn_column='Sequence', 
        cond_columns=CONDITION_COLUMNS + [SAMPLE_COLUMN, RANGE_LOW_COLUMN, RANGE_HIGH_COLUMN], 
        label_column='Label',
        cond_transform=cond_transform
    )
    
    print('Number of entries:', len(dataset))
    print('Sequence [0]:', dataset[0][0])
    print('Experimental conditions [0]:', dataset[0][1])
    print('Label [0]:', dataset[0][2])
