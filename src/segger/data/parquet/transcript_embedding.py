import torch.nn
import torch.nn.functional as F
from torch import Tensor, LongTensor
from sklearn.preprocessing import LabelEncoder
from typing import Optional, Union
from numpy.typing import ArrayLike
import pandas as pd
import numpy as np

# TODO: Add documentation
class TranscriptEmbedding(torch.nn.Module):
    '''
    Utility class to handle transcript embeddings in PyTorch so that they are
    optionally learnable in the future.
    
    Default behavior is to use the index of gene names.
    '''

    # TODO: Add documentation
    @staticmethod
    def _check_inputs(
        classes: ArrayLike,
        weights: Union[pd.DataFrame, None],
    ):
        # Classes is a 1D array
        if len(classes.shape) > 1:
            msg = (
                "'classes' should be a 1D array, got an array of shape "
                f"{classes.shape} instead."
            )
            raise ValueError(msg)
        # Items appear exactly once
        if len(classes) != len(set(classes)):
            msg = (
                "All embedding classes must be unique. One or more items in "
                "'classes' appears twice."
            )
            raise ValueError(msg)
        # All classes have an entry in weights
        elif weights is not None:
            missing = set(classes).difference(weights.index)
            if len(missing) > 0:
                msg = (
                    f"Index of 'weights' DataFrame is missing {len(missing)} "
                    "entries compared to classes."
                )
                raise ValueError(msg)

    # TODO: Add documentation
    def __init__(
        self,
        classes: ArrayLike,
        weights: Optional[pd.DataFrame] = None,
    ):
        # check input arguments
        TranscriptEmbedding._check_inputs(classes, weights)
        # Setup as PyTorch module
        super(TranscriptEmbedding, self).__init__()
        self._encoder = LabelEncoder().fit(classes)
        if weights is None:
            self._weights = None
        else:
            idx = np.argsort(self._encoder.transform(classes))
            self._weights = Tensor(weights.loc[classes].iloc[idx].values)

    # TODO: Add documentation
    def embed(self, classes: ArrayLike):
        """
        Transforms input labels into embeddings or indices.
        Unseen labels are mapped to a default index (-1).
        """
        indices = np.full(len(classes), -1, dtype=int)  # Default index for unseen labels
        try:
            valid_indices = self._encoder.transform(classes)
            indices = LongTensor(valid_indices)
        except ValueError:
            # Map unseen labels to -1
            valid_classes = set(self._encoder.classes_)
            indices = LongTensor([self._encoder.transform([c])[0] if c in valid_classes else -1 for c in classes])

        if self._weights is None:
            return indices
        else:
            return F.embedding(indices, self._weights)
    
    def to_indices(self, classes: ArrayLike) -> LongTensor:
        """
        Returns a stable mapping of classes to indices.
        """
        indices = np.full(len(classes), -1, dtype=int)
        try:
            valid_indices = self._encoder.transform(classes)
            indices = LongTensor(valid_indices)
        except ValueError:
            valid_classes = set(self._encoder.classes_)
            indices = LongTensor([
                self._encoder.transform([c])[0] if c in valid_classes else -1
                for c in classes
            ])
        return indices
