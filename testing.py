import chromadb
import pandas as pd
from preprocessing import *


def preprocess_test_id(config, chroma_client: chromadb.ClientAPI, prot_id,
                       neighbors, train_go_terms, test_infoprot,
                       prot_id_embeddings, out_rows=150, include_neighbors=False):
    # Find related embeddings
    collection = chroma_client.get_collection(config['collection_name'])

    results = collection.query(
        query_embeddings=prot_id_embeddings,
        n_results=neighbors,
        include=['distances', 'embeddings']
    )
    embeddings = results['embeddings'][0]
    embeddings = pd.DataFrame(embeddings)
    embeddings = embeddings._append([prot_id_embeddings], ignore_index=True)
    embeddings = embeddings.rename(
        columns={i: f'embd_dim{i}' for i in embeddings.columns})

    original_row = {'protein_id': prot_id, 'distance': 0.0}

    results = {'protein_id': results['ids'][0],
               'distance': results['distances'][0]}
    results_df = pd.DataFrame().from_dict(results)
    results_df = results_df._append(original_row, ignore_index=True)

    results_df = pd.concat([results_df, embeddings], axis=1)

    go_terms_reduced = \
        train_go_terms[train_go_terms['protein_id'].isin(results_df['protein_id'])][
            ['protein_id', 'GO_term']]
    infoprot_reduced = \
        test_infoprot[test_infoprot['protein_id'] == prot_id][
            ['protein_id', 'infoprot_id']]

    merged_df = pd.merge(results_df, go_terms_reduced, on='protein_id',
                         how='left')

    merged_df = pd.merge(merged_df, infoprot_reduced, on='protein_id',
                         how='left').drop_duplicates()

    #print(merged_df)

    merged_df = resize_dataframe(merged_df, target_row_count=out_rows)
    merged_df = merged_df.reset_index(drop=True)
    merged_df['original_id'] = prot_id

    if not include_neighbors:
        merged_df = merged_df[merged_df['protein_id'] == prot_id]
    return merged_df


def preprocess_batch_test(config, chroma_client, prot_ids, neighbors,
                          train_go_terms, test_infoprot,
                          prot_id_embeddings, out_rows=150):
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
        preprocess_test_id(
            config,
            chroma_client,
            prot_id,
            neighbors,
            train_go_terms,
            test_infoprot,
            prot_id_embedding,
            out_rows
        )
        for prot_id, prot_id_embedding in zip(prot_ids, prot_id_embeddings)
    ]
    return dataframes


def get_prot_id_embeddings(embeddings_file_path, prot_ids):
    with h5py.File(embeddings_file_path, 'r') as embeddings_file:
        embeddings = [
            embeddings_file[prot_id][:].tolist()
            for prot_id in prot_ids if prot_id in embeddings_file
        ]
        return embeddings


def preprocess_test_generator(aspect="cellular_component"):
    config = load_config("config.yaml")

    sequences_file = config['test_files']['sequences_file']
    embeddings_file = config['test_files']['embeddings_file']
    infoprot_file = config['test_files']['infoprot_file']
    test_ids_file = config['test_files']['test_ids_file']

    train_go_terms_file = config['files']['train_file']

    # sequences = parse_fasta(sequences_file)
    infoprot = parse_infoprot(infoprot_file)
    train_go_terms = parse_go_terms(train_go_terms_file, aspect=aspect)

    prot_ids = load_train_ids(test_ids_file)

    client = chromadb.PersistentClient()

    for batch in batch_ids(prot_ids,
                           batch_size=config['preprocessing']['batch_size']):

        prot_id_embeddings = get_prot_id_embeddings(embeddings_file, batch)

        preprocessed_batch = preprocess_batch_test(
            config,
            client,
            batch,
            neighbors=config['neighbors'],
            train_go_terms=train_go_terms,
            test_infoprot=infoprot,
            prot_id_embeddings=prot_id_embeddings,
            out_rows=config['preprocessing']['out_rows']
        )
        yield preprocessed_batch


if __name__ == "__main__":
    for i, batch in enumerate(preprocess_test_generator()):
        print(batch)
        if i == 5:
            break