import pandas as pd

from preprocessing import *
from tensorflow import keras
from tensorflow.keras import backend as K
from argparse import ArgumentParser

class DataParser:
    embedding = None
    test_ids = None
    
    def __init__(self,  test_embedding_path, test_ids_path):
        self.embedding_path = test_embedding_path
        self.test_ids_path = test_ids_path
    
    def _build_test_data(self, test_embeddings_data, test_ids):
        X_test = []
        for id in test_ids:
            X_test.append(test_embeddings_data[id])
        X_test = np.array(X_test)
        return X_test
    
    def _readh5_to_dict(self, file_path):
    # Create an empty dictionary to store the data
        p_embeddings_data = {}

        # Open the HDF5 file
        with h5py.File(file_path, 'r') as p_embeddings:
            # Store the data in the dictionary
            for key in p_embeddings.keys():
                p_embeddings_data[key] = p_embeddings[key][...]

        return p_embeddings_data
    
    def _read_protein_ids(self, file_path, percentage=1.0):

        # Read the IDs from the text file
        with open(file_path, 'r') as file:
            ids = [line.strip() for line in file]

        # Calculate the index to get the first 30% of IDs
        split_index = int(len(ids) * percentage)

        # Select the first 30% of IDs
        selected_ids = ids[:split_index]

        return selected_ids
    
    def _load_test_data(self):
        self.embedding = self._readh5_to_dict(self.embedding_path)
        self.test_ids = self._read_protein_ids(self.test_ids_path)
        self.X_test = self._build_test_data(self.embedding, self.test_ids)
        
    def load(self):
        self._load_test_data()
        return self.embedding, self.test_ids, self.X_test
        

class TermFrequency:
    BP_LABELS = 1100
    CC_LABELS = 300
    MF_LABELS = 450
    
    pred_columns_cc = None
    pred_columns_bp = None
    pred_columns_mf = None
    
    def __init__(self, cc_frequency_table_path, bp_frequency_table_path, mf_frequency_table_path):
        self.cc_frequency_table_path = cc_frequency_table_path
        self.bp_frequency_table_path = bp_frequency_table_path
        self.mf_frequency_table_path = mf_frequency_table_path
    
    def load(self):
        bp_freq_df = pd.read_csv(self.bp_frequency_table_path)[:self.BP_LABELS]
        mf_freq_df = pd.read_csv(self.mf_frequency_table_path)[:self.MF_LABELS]
        cc_freq_df = pd.read_csv(self.bp_frequency_table_path)[:self.CC_LABELS]
        
        return {'bp': bp_freq_df['id'].tolist(), 'mf': mf_freq_df['id'].tolist(), 'cc': cc_freq_df['id'].tolist()}

class Predict:
    MAX_LABELS = 1500
    
    cc_model = None
    mf_model = None
    bp_model = None
    
    def __init__(self, cc_model_path, bp_model_path, mf_model_path):
        self.cc_model_path = cc_model_path
        self.bp_model_path = bp_model_path
        self.mf_model_path = mf_model_path
        
    def _load_model(self):
        self.cc_model = keras.models.load_model(self.cc_model_path, custom_objects={'f1_score': self._f1_score})
        self.bp_model = keras.models.load_model(self.bp_model_path, custom_objects={'f1_score': self._f1_score})
        self.mf_model = keras.models.load_model(self.mf_model_path, custom_objects={'f1_score': self._f1_score})

    def _f1_score(self, y_true, y_pred):
        precision = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) / (K.sum(K.round(K.clip(y_pred, 0, 1))) + K.epsilon())
        recall = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) / (K.sum(K.round(K.clip(y_true, 0, 1))) + K.epsilon())
        return 2 * (precision * recall) / (precision + recall + K.epsilon())
        
    def _generate_submission_df(self, y_pred, test_ids, pred_columns):
        # assert that the length of y_pred must be same as test_ids
        assert len(y_pred) == len(test_ids)
        
        # Group by the result and then sort by score id
        out = {'id': [], 'term': [], 'score': []}
        for i in range(len(y_pred)):
            for j in range(len(y_pred[i])):
                out['id'].append(test_ids[i])
                out['term'].append(pred_columns[j])
                out['score'].append(y_pred[i][j])
        
        out_df = pd.DataFrame(out).reset_index(drop=True)
        
        out_df = out_df.groupby('id', group_keys=False)
        out_df = out_df.apply(lambda x: x.sort_values(by='score', ascending=False))
        
        # Filter the DataFrame
        out_df = out_df[out_df['id'].isin(test_ids)]
        
        # Convert the 'ID' column to a Categorical with the order defined in filter_array
        out_df['id'] = pd.Categorical(out_df['id'], categories=test_ids, ordered=True)
        
        # Sort by the 'ID' column
        out_df = out_df.sort_values('id')
        out_df['id'] = out_df['id'].astype(str)
        return out_df

    def _concat_predictions(self, bp_df, cc_df, mf_df, test_ids):
        # Concatenate the DataFrames
        concatenated_df = pd.concat([bp_df, mf_df, cc_df])
        
        # Create a custom sorting order based on the external list
        sorting_order = {id: index for index, id in enumerate(test_ids)}
        concatenated_df['sort_order'] = concatenated_df['id'].map(sorting_order)
        
        # Sort by custom order and then by probability within each group
        sorted_df = concatenated_df.sort_values(by=['sort_order', 'score'], ascending=[True, False])
        
        # Limit to 1500 rows per ID
        limited_df = sorted_df.groupby('id').head(1500)
        
        # Drop the auxiliary 'sort_order' column
        limited_df = limited_df.drop(columns=['sort_order'])
        
        return limited_df
    
    def predict(self, X_test, test_ids, pred_columns):
        self._load_model()
        
        predictions = []
        for name, model in {'bp': self.bp_model, 'cc': self.cc_model, 'mf': self.mf_model}.items():
            y_pred = model.predict(X_test)
            pred_df = self._generate_submission_df(y_pred, test_ids, pred_columns[name])
            predictions.append(pred_df)

        submission_df = self._concat_predictions(predictions[0], predictions[1], predictions[2], test_ids)
        submission_df['score'] = submission_df['score'].round(3)
        submission_df = submission_df[submission_df['score'] > 0.000]
            
        return submission_df
    
    
def create_parser():
    """
    Creates the arguments to interact with the script.
    :return:
    """
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", default="config.yaml")
    args = parser.parse_args()
    return args


def main():
    args = create_parser()
    config = load_config(args.config)
    
    # Parse the test data for prediction
    data_parser = DataParser(config['test_files']['embeddings_file'], config['test_files']['test_ids_file'])
    _, test_ids, X_test = data_parser.load()
    
    # Parse the frequency table
    term_frequency = TermFrequency(config['frequency_table']['cc_frequency_table'], 
                                   config['frequency_table']['bp_frequency_table'], 
                                   config['frequency_table']['mf_frequency_table'])
    pred_columns = term_frequency.load()
    
    # Output the predictions
    predictor = Predict(config['models']['celullar_component_model'], 
                        config['models']['biological_process_model'], 
                        config['models']['molecular_function_model'])
    submission = predictor.predict(X_test, test_ids, pred_columns)
    
    output_file = config['output_file']
    submission.to_csv(output_file, sep='\t', index=False, header=False)

if __name__ == '__main__':
    main()
