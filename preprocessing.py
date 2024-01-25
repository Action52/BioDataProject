import h5py
from Bio import SeqIO
import pandas as pd
import chromadb
from tqdm import tqdm
import yaml
import numpy as np
from utils import load_config


def parse_fasta(file_path, size=-1):
    """
    Parse a FASTA file to extract sequences.

    Args:
    file_path (str): Path to the FASTA file.
    size (int): If specified, will only process the n first sequences in the file.

    Returns:
    DataFrame: A pandas DataFrame with sequence identifiers as keys and sequences as values.
    """
    sequences_list = []
    with open(file_path, 'r') as fasta_file:
        for i, record in enumerate(SeqIO.parse(fasta_file, 'fasta')):
            sequences_list.append({
                'protein_id': record.id,
                'sequence': str(record.seq)
            })
            if size == i:
                break
    sequences = pd.DataFrame(sequences_list)

    return sequences


def load_data_into_chroma(embeddings_file_path, sequences, go_terms, infoprot,
                          prot_ids, size=-1,
                          collection_name='sequence_embeddings'):
    """
    Load embeddings from a .h5 file.

    Args:
    embeddings_file_path (str): Path to the .h5 file containing the embeddings.
    sequences (DataFrame): Loaded dataframe with the sequences to store in the
        embeddings db.
    go_terms (DataFrame): Loaded dataframe with the go terms associated to the
        protein_ids.
    infoprot (DataFrame): Loaded dataframe with the infoprot annotations of the
        protein_ids.
    prot_ids (list): List of protein_ids to process.
    size (int): If specified, will only process the n first sequences in the
        file.

    Returns:
    dict: Dictionary of embeddings, where keys are the identifiers and values
        are the embedding arrays.
    """

    chroma = chromadb.PersistentClient()
    collection = chroma.get_or_create_collection(collection_name)

    sequences_reduced = sequences[sequences['protein_id'].isin(prot_ids)]
    go_terms_reduced = go_terms[go_terms['protein_id'].isin(prot_ids)][['protein_id','GO_term']]
    infoprot_reduced = infoprot[infoprot['protein_id'].isin(prot_ids)][['protein_id','infoprot_id']]

    merged = pd.merge(sequences_reduced, go_terms_reduced, on='protein_id',
                         how='left')
    merged = pd.merge(merged, infoprot_reduced, on='protein_id',
                         how='left')

    with h5py.File(embeddings_file_path, 'r') as embeddings_file:
        prot_ids_docs = []
        embeddings = []

        for prot_id in prot_ids:
            if prot_id in embeddings_file:
                embedding_as_list = embeddings_file[prot_id][:].tolist()
                seq = merged[merged['protein_id'] == prot_id]['sequence'].unique()[0]
                gos = list(merged[merged['protein_id'] == prot_id]['GO_term'].unique())
                infos = list(merged[merged['protein_id'] == prot_id]['GO_term'].unique())

                doc = {'sequence': seq, 'go_terms': gos, 'infoprots': infos}

                prot_ids_docs.append(seq)
                embeddings.append(embedding_as_list)

        collection.add(
            documents=prot_ids_docs,
            embeddings=embeddings,
            ids=prot_ids
        )


def parse_go_terms(file_path, aspect, size=-1, sep='\t'):
    """
    Parse a file to extract GO terms for a given aspect.

    Args:
    file_path (str): Path to the file containing the data.
    aspect (str): The aspect to filter by (e.g., 'molecular_function').
    size (int): If specified, will only process the n first rows in the file.

    Returns:
    DataFrame: A pandas DataFrame with Protein_ID and GO_term for the specified aspect.
    """
    if size >= 0:
        df = pd.read_csv(file_path, sep=sep, nrows=size)
    else:
        df = pd.read_csv(file_path, sep=sep)

    df = df.rename(columns={'Protein_ID': 'protein_id'})
    filtered_df = df[df['aspect'] == aspect]

    return filtered_df


def parse_infoprot(file_path, size=-1, sep='\t',
                   col_names=('protein_id', 'infoprot_id', 'domain_description',
                              'other_iD', 'range_start', 'range_end')):
    """
    Parse a file to extract protein information.

    Args:
    file_path (str): Path to the file containing the data.
    size (int): If specified, will only process the n first rows in the file.

    Returns:
    DataFrame: A pandas DataFrame with the extracted information.
    """

    if size >= 0:
        df = pd.read_csv(file_path, sep=sep, names=col_names, nrows=size)
    else:
        df = pd.read_csv(file_path, sep=sep, names=col_names)

    return df


def load_train_ids(file_path, size=-1):
    """
    Load a list of train IDs from a file.

    Args:
    file_path (str): Path to the file containing the train IDs.

    Returns:
    list: A list of train IDs.
    """
    with open(file_path, 'r') as file:
        train_ids = file.read().splitlines()

    if size > 0:
        return train_ids[:size]
    return train_ids


def batch_ids(ids, batch_size):
    """
    Generator function to yield batches of IDs.

    Args:
    ids (list): List of IDs.
    batch_size (int): Size of each batch.

    Yields:
    list: A batch of train IDs.
    """
    for i in range(0, len(ids), batch_size):
        yield ids[i:i + batch_size]


def create_embeddings_db(config, prot_ids, sequences, go_terms, infoprot):
    batch_size = config['ingestion']['batch_size']
    batches = len(prot_ids)/batch_size
    print("Estimated batches", batches)
    for ids in tqdm(batch_ids(prot_ids, batch_size)):
        load_data_into_chroma(embeddings_file, sequences, go_terms, infoprot,
                              ids, collection_name=config['collection_name'])
    print("Finished loading embeddings into db.")


def preprocess_id(config, chroma_client: chromadb.ClientAPI, prot_id, neighbors,
                  go_terms, infoprot, out_rows=150):
    # Find related embeddings
    collection = chroma_client.get_collection(config['collection_name'])

    embds = collection.get(
        ids=[prot_id],
        include=['documents', 'embeddings']
    )['embeddings'][0]

    results = collection.query(
        query_embeddings=embds,
        n_results=neighbors,
        include=['distances', 'embeddings']
    )
    embeddings = results['embeddings'][0]
    embeddings = pd.DataFrame(embeddings)
    embeddings = embeddings.rename(columns={i: f'embd_dim{i}' for i in embeddings.columns})

    results = {'protein_id': results['ids'][0], 'distance': results['distances'][0]}
    results_df = pd.DataFrame().from_dict(results)
    results_df = pd.concat([results_df, embeddings], axis=1)

    go_terms_reduced = go_terms[go_terms['protein_id'].isin(results_df['protein_id'])][['protein_id', 'GO_term']]
    infoprot_reduced = infoprot[infoprot['protein_id'].isin(results_df['protein_id'])][['protein_id', 'infoprot_id']]

    merged_df = pd.merge(results_df, go_terms_reduced, on='protein_id',
                         how='left')

    merged_df = pd.merge(merged_df, infoprot_reduced, on='protein_id',
                         how='left').drop_duplicates()

    merged_df = resize_dataframe(merged_df, target_row_count=out_rows)
    merged_df = merged_df.reset_index()
    # print(merged_df)
    return merged_df


def resize_dataframe(df, target_row_count=150):
    """
    Resize a DataFrame to a specified number of rows.

    Args:
    df (pd.DataFrame): The original DataFrame.
    target_row_count (int): The desired number of rows.

    Returns:
    pd.DataFrame: The resized DataFrame.
    """
    current_row_count = df.shape[0]
    if current_row_count > target_row_count:
        # Truncate the DataFrame if it has more than the target number of rows
        resized_df = df.iloc[:target_row_count]
        return resized_df
    elif current_row_count < target_row_count:
        # Pad the DataFrame if it has fewer than the target number of rows
        rows_to_add = target_row_count - current_row_count
        padded_df = pd.DataFrame(np.nan, index=range(rows_to_add), columns=df.columns)
        resized_df = pd.concat([df, padded_df], ignore_index=True)
        return resized_df
    return df


def preprocess_batch(config, chroma_client, prot_ids, neighbors, go_terms, infoprot, out_rows=150):
    """
    Preprocess a batch of protein IDs.

    Args:
    config (dict): Configuration dictionary.
    chroma_client (chromadb.ClientAPI): ChromaDB client.
    prot_ids (list): List of protein IDs to process.
    neighbors (int): Number of neighbors to consider in ChromaDB query.
    go_terms (pd.DataFrame): DataFrame with GO terms.
    infoprot (pd.DataFrame): DataFrame with infoprot data.
    out_rows (int): Number of rows to output in each DataFrame.

    Returns:
    list: A list of DataFrames, each corresponding to a protein ID.
    """
    dataframes = [
        preprocess_id(
            config,
            chroma_client,
            prot_id,
            neighbors,
            go_terms,
            infoprot,
            out_rows
        )
        for prot_id in prot_ids
    ]
    return dataframes


def preprocess_generator():
    config_path = "config.yaml"
    config = load_config(config_path)

    sequences_file = config['files']['sequences_file']
    embeddings_file = config['files']['embeddings_file']
    train_file = config['files']['train_file']
    infoprot_file = config['files']['infoprot_file']
    train_ids_file = config['files']['train_ids_file']

    sequences = parse_fasta(sequences_file)
    go_terms = parse_go_terms(train_file, aspect='cellular_component')
    infoprot = parse_infoprot(infoprot_file)

    prot_ids = load_train_ids(train_ids_file)

    if config['ingestion']['execute']:
        create_embeddings_db(config, prot_ids, sequences, go_terms, infoprot)

    client = chromadb.PersistentClient()

    if config['preprocessing']['execute']:
        for batch in batch_ids(prot_ids, batch_size=config['preprocessing']['batch_size']):
            preprocessed_batch = preprocess_batch(
                config,
                client,
                batch,
                neighbors=config['neighbors'],
                go_terms=go_terms,
                infoprot=infoprot,
                out_rows=config['preprocessing']['out_rows']
            )
            yield preprocessed_batch


if __name__ == "__main__":
    for i, batch in enumerate(preprocess_generator()):
        print(batch)
        if i == 5:
            break
