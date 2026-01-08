import torch
import torch.nn as nn
from src.models.embed_seq import get_sqn_embed_model
from src.models.attention import SequenceTransformer


class EnsembleConditionsClassifier(nn.Module):
    """
    Model class that uses ensemble of sequence-based embeddings and structure-based embeddings for prediction.
    The sequence-based embeddings are cross attended to the structure-based embeddings.
    """
    def __init__(self, embed_model: str, condition_dim: int,
                 embedding_dim: int, num_layers: int, num_heads: int, 
                 dim_feedforward: int, dropout: float = 0.1, 
                 local_files_only: bool = False):
        """
        :param embed_model: name of the first embedding model
        :type embed_model: str
        :param condition_dim: dimension of input conditions
        :type condition_dim: int
        :param embedding_dim: dimension of the embedding vector
        :type embedding_dim: int
        :param num_layers: number of attention layers
        :type num_layers: int
        :param num_heads: number of heads in multi-head attention layers
        :type num_heads: int
        :param dim_feedforward: dimension of output for the feedforward layer
            after each attention
        :type dim_feedforward: int
        :param drpoout: dropout rate in positional embedding and attention layers
        :type dropout: float
        :param local_files_only: True if load model checkpoints from local,
            False if from online hub (e.g., HuggingFaceHub)
        :type local_files_only: bool
        """
        super().__init__()
        self.embed_model1 = get_sqn_embed_model(embed_model, local_files_only=local_files_only)
        self.embed_model2 = nn.Sequential(
            nn.Linear(condition_dim, 32),
            nn.Tanh(),
            nn.Dropout(0.0),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Dropout(0.0),
            nn.Linear(64, 128)
        )
        self.encoder1 = SequenceTransformer(
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        # binary classifier (uses [CLS] token)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim+128, 64),
            nn.Tanh(),
            nn.Dropout(0.0),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Dropout(0.0),
            nn.Linear(32, 1)
        )
    
    def tokenize(self, sequence_list):
        self.embed_model1.eval()
        with torch.no_grad():
            tokens1 = self.embed_model1.tokenize(sequence_list)
            return tokens1
    
    def forward(self, sequence_list1, sequence_list2):
        # embedding model
        self.embed_model1.eval()
        with torch.no_grad():
            embeddings1, attention_masks1 = self.embed_model1(sequence_list1)
        embeddings1 = embeddings1.detach()
        attention_masks1 = attention_masks1.detach()
        embeddings2 = self.embed_model2(sequence_list2)
        # embeddings2 = embeddings2.detach()
        # transformer encoder
        x1 = self.encoder1(embeddings1, attention_masks1)
        # use the [CLS] token output
        cls_output = x1[:, 0, :]  # (B, D)
        cls_output = torch.cat((cls_output, embeddings2), dim=1)
        # final binary prediction
        logits = self.classifier(cls_output).squeeze(-1)  # (B,)
        return logits


# Example usage
if __name__ == "__main__":
    # command
    # cd LLPS_Predict
    # python -m src.models.model_condition

    # input config
    SQN_EMBED_MODEL = 'esm2_650m'
    SQN_EMBED_LOAD_LOCAL = True
    CONDITION_DIM = 5
    SQN_EMBED_DIM = 1280
    DROPOUT = 0.6
    N_LAYER = 1
    N_HEAD = 4
    FEED_FORWARD_DIM = 1280
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # sample input
    sequence_list = ('MEVQLVQYK', 'MEVQLVQYK')
    condition_list = torch.Tensor([(20.0, 0.5, 440.7978, 7.4, 0.0), (20.0, 0.5, 120.5, 7.4, 0.0)])
    condition_list = condition_list.to(DEVICE)

    # model
    classifier = EnsembleConditionsClassifier(
        SQN_EMBED_MODEL,
        condition_dim=CONDITION_DIM,
        embedding_dim=SQN_EMBED_DIM,
        num_layers=N_LAYER,
        num_heads=N_HEAD,
        dim_feedforward=FEED_FORWARD_DIM,
        dropout=DROPOUT,
        local_files_only=SQN_EMBED_LOAD_LOCAL
    )
    classifier = classifier.to(DEVICE)

    # forward pass
    x = classifier(sequence_list, condition_list)

    print('Model: Ensemble ESM2 + Experimental Condition MLP')
    print('Logits shape:', x.shape)     # Expected: (2,)
    print('Logits:', x)
