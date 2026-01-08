# LLPS_Predict

Our ensemble model integrates both protein sequence and structure information to predict the propensity of proteins to undergo liquid-liquid phase separation (LLPS). In the first step, our ensemble model distinguishes LLPS proteins from non-LLPS proteins. In the second step, proteins predicted to undergo phase separation are further classified as drivers or partners. Our method offers improved interpretability by pinpointing the sequence regions that drive phase separation. It also supports mutagenesis analysis that reveals the effects of point mutations on phase separation propensity.

## Installation
This tool need to be installed before use. You can install this tool on a GPU machine. You may create a virtual environment.

Clone the repository and install the requirements in `requreiments.txt`
```
git clone https://github.com/mingjie0508/LLPS_Predict.git
cd LLPS_Predict
pip install -r requirements.txt
```

## Model Checkpoints

Download the model weights by running
```
./checkpoints/download_ckpts.sh
```

The above script creates two folders `foundation_models` and `trained_models`. It downloads pre-trained models from huggingface into `trained_models`:
- [Ensemble](https://huggingface.co/mingjiezhao0508/LLPS_Predict/blob/main/ensemble_baseline_h4_l1.pth): predicts the propensity of proteins to undergo LLPS
- [Ensemble_Driver](https://huggingface.co/mingjiezhao0508/LLPS_Predict/blob/main/ensemble_driver_baseline_h4_l1.pth): predicts the propensity of proteins to undergo homotypic LLPS
- [Ensemble_Partner](https://huggingface.co/mingjiezhao0508/LLPS_Predict/blob/main/ensemble_partner_baseline_h4_l1.pth): predicts the propensity of proteins to act as a partner in LLPS

Some foundation models may be cached in `foundation_models` while running subsequent scripts.

## Usage
To use as a library, refer to [tutorial.ipynb](./src/notebooks/tutorial.ipynb)
It illustrates how to:
- predict per-protein LLPS propensity scores
- predict per-residue scores and identify critical regions
- perform mutagenesis analysis

## Project Structure
```
├── checkpoints           # Download pretrained weights here
│   ├── foundation_models # Some foundation models downloaded here automatically
│   └── trained_models    # Project-specific models
├── data                  # Processed data
└── src                   # Source code
    ├── experiments       # Configuration files, bash scripts, and output
    ├── lib               # High-level functions for users
    ├── loaders           # PyTorch dataset classes
    ├── models            # Model architecture
    ├── notebooks         # Jupyter notebook tutorials
    ├── scripts           # Train, test, and evaluate scripts
    └── utils             # Utils code (get structure, moving window, etc.)
```

## Dataset
The datasets can be download [here](https://utoronto-my.sharepoint.com/:f:/g/personal/mingjie_zhao_mail_utoronto_ca/IgCSvmmwdwqkSJ27vixd9FG4Ae5y3MQLTPHQws3rJscIQRY?e=L6CNB1). All data are provided as csv files. The directory structure is:
```
└── data                  # Processed data
    ├── clinvar
    ├── denovo_db
    ├── experimental_condition
    ├── human_proteome
    ├── llps_driving_region
    ├── localization
    ├── test
    ├── training
    └── validation
```