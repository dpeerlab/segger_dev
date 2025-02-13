#---- Covariance Utils -------------------------
# These utilities are used to compute the 
# covariance matrix of the gene expression data.
# ----------------------------------------------

import numpy as np
import pandas as pd

def build_gene_cov_matrix(
    transcripts: pd.DataFrame, 
    cell_col: str = 'cell_id', 
    gene_col: str = 'feature_name',
    embedding: "TranscriptEmbedding" = None, 
    global_matrix: pd.DataFrame or list or None = None
) -> pd.DataFrame:
    """
    Computes a gene-gene covariance matrix based on the distribution
    of transcripts across cells (or boundaries).

    Parameters
    ----------
    transcripts : pd.DataFrame
        DataFrame of transcripts with at least [cell_col, gene_col].
        Each row typically corresponds to one transcript, with a cell ID
        and a gene/feature name.
    cell_col : str, optional
        Column name corresponding to cell IDs. Default is 'cell_id'.
    gene_col : str, optional
        Column name corresponding to gene/feature names. Default is 'feature_name'.
    global_matrix : pd.DataFrame or list of str, optional
        A global gene covariance matrix (or a list of global gene names) that 
        defines the complete gene set. If provided, the resulting covariance 
        matrix will be reindexed to include all genes from the global set, with
        missing entries padded with 0's.

    Returns
    -------
    pd.DataFrame
        A DataFrame representing the gene-gene covariance matrix with gene labels
        as both its index and columns.
    """
    pivot = pd.pivot_table(
        transcripts,
        index=cell_col,
        columns=gene_col,
        aggfunc='size',      # counting transcripts
        fill_value=0
    )
    local_cov = pivot.cov()

    # If there is an embedding convert the gene names to the embedding
    if embedding is not None:
        # Convert the existing row/column labels to integer codes
        # local_cov.index/columns are currently string gene names
        row_codes = embedding.to_indices(local_cov.index).numpy()  # shape [N_genes]
        col_codes = embedding.to_indices(local_cov.columns).numpy()  # shape [N_genes]

        # Assign them as Pandas Index objects
        local_cov.index = pd.Index(row_codes, name='gene_code')
        local_cov.columns = pd.Index(col_codes, name='gene_code')

        # After mi local_cov:
        missing_rows = (local_cov.index == -1)
        missing_cols = (local_cov.columns == -1)

        # Drop them
        local_cov = local_cov.loc[~missing_rows, ~missing_cols]

    # If a global matrix or global gene list is provided, reindex the covariance matrix
    if global_matrix is not None:
        if isinstance(global_matrix, pd.DataFrame):
            global_genes = global_matrix.index
        elif isinstance(global_matrix, (list, tuple, np.ndarray)):
            global_genes = global_matrix
        else:
            raise ValueError("global_matrix must be a DataFrame or a list/tuple/array of gene names.")
        local_cov = local_cov.reindex(index=global_genes, columns=global_genes, fill_value=0)
    return local_cov