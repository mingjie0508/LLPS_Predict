# LLPS models
import torch
from src.models.model import TransformerClassifier
from src.models.model_ensemble import (
    EnsembleClassifier,
    EnsembleIntermapClassifier,
    EnsembleIntermapClassifier2
)
from src.models.model_intermap import IntermapClassifier


def get_llps_model(model_name: str, local_files_only: bool = False):
    # dictionaries of all model info
    model_paths = {
        'esm2': 'checkpoints/trained_models/esm2_650m_baseline_h4_l1.pth',
        'saprot': 'checkpoints/trained_models/saprot_650m_baseline_h4_l1.pth',
        'ensemble': 'checkpoints/trained_models/ensemble_baseline_h4_l1.pth',
        'ensemble_driver': 'checkpoints/trained_models/ensemble_driver_baseline_h4_l1.pth',
        'ensemble_partner': 'checkpoints/trained_models/ensemble_partner_baseline_h4_l1.pth',
        'intermapcnn': 'checkpoints/trained_models/intermapcnn_baseline_h4_l1.pth',
        'esm2intermapcnn': 'checkpoints/trained_models/esm2intermapcnn_baseline_h4_l1.pth',
        'esm2saprotintermapcnn': 'checkpoints/trained_models/esm2saprotintermapcnn_baseline_h4_l1.pth',
    }
    kwargs_dict = {
        'esm2': {'embed_model': 'esm2_650m', 'embedding_dim': 1280, 'num_layers': 1, 'num_heads': 4, 'dim_feedforward': 1280, 'dropout': 0.6, 'local_files_only': local_files_only},
        'saprot': {'embed_model': 'saprot_650m', 'embedding_dim': 1280, 'num_layers': 1, 'num_heads': 4, 'dim_feedforward': 1280, 'dropout': 0.6, 'local_files_only': local_files_only},
        'ensemble': {'embed_model1': 'esm2_650m', 'embed_model2': 'saprot_650m', 'embedding_dim': 1280, 'num_layers': 1, 'num_heads': 4, 'dim_feedforward': 1280, 'dropout': 0.6, 'local_files_only': local_files_only},
        'ensemble_driver': {'embed_model1': 'esm2_650m', 'embed_model2': 'saprot_650m', 'embedding_dim': 1280, 'num_layers': 1, 'num_heads': 4, 'dim_feedforward': 1280, 'dropout': 0.6, 'local_files_only': local_files_only},
        'ensemble_partner': {'embed_model1': 'esm2_650m', 'embed_model2': 'saprot_650m', 'embedding_dim': 1280, 'num_layers': 1, 'num_heads': 4, 'dim_feedforward': 1280, 'dropout': 0.6, 'local_files_only': local_files_only},
        'intermapcnn': {'in_channels': 1, 'dropout': 0.2},
        'esm2intermapcnn': {'embed_model': 'esm2_650m', 'embedding_dim': 1280, 'num_layers': 1, 'num_heads': 4, 'dim_feedforward': 1280, 'dropout': 0.6, 'local_files_only': local_files_only},
        'esm2saprotintermapcnn': {'embed_model1': 'esm2_650m', 'embed_model2':'saprot_650m', 'embedding_dim': 1280, 'num_layers': 1, 'num_heads': 4, 'dim_feedforward': 1280, 'dropout': 0.6, 'local_files_only': local_files_only}
    }
    model_cls_dict = {
        'esm2': TransformerClassifier,
        'saprot': TransformerClassifier,
        'ensemble': EnsembleClassifier,
        'ensemble_driver': EnsembleClassifier,
        'ensemble_partner': EnsembleClassifier,
        'intermapcnn': IntermapClassifier,
        'esm2intermapcnn': EnsembleIntermapClassifier,
        'esm2saprotintermapcnn': EnsembleIntermapClassifier2,
    }
    # get model parameters based on input
    if not model_name in model_cls_dict:
        raise NotImplementedError(f"Model {model_name} is not implemented.")
    cls = model_cls_dict[model_name]
    kwargs = kwargs_dict[model_name]
    checkpoint_path = model_paths[model_name]
    # instantiate model
    model = cls(**kwargs)
    # load checkpoint
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    return model
