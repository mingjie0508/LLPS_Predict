# LLPSDB2 experimental conditions helper function
import torch


def experimental_condition_transform_factory(
        cond_columns: list, sample_column: str, low_column: str, high_column: str
    ):
    def transform(cond):
        # uniformly sample a quantity of interest within range
        c = cond.loc[sample_column]
        low = cond.loc[low_column]
        high = cond.loc[high_column]
        x = cond.loc[cond_columns]
        sample = low + torch.rand(1).item() * (high - low)
        # save sampled value
        x.loc[c] = sample
        return x
    return transform
