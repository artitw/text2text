import os
import requests
import pickle
import text2text as t2t


class Identifier():

    def _get_model(self):
        """Download and loads model from google drive"""

        file_id = "10-YNx9yDVJTUijVz_2aopcIF8rdXR78R"
        URL = "https://docs.google.com/uc?export=download"
        session = requests.Session()
        response = session.get(URL, params = { 'id' : file_id }, stream = True)
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                params = { 'id' : file_id, 'confirm' : value }
                response = session.get(URL, params = params, stream = True)

        CHUNK_SIZE = 32768

        model_file = "identifier.pkl"
        if not os.path.exists(model_file):
            with open(model_file, "wb") as f:
                for chunk in response.iter_content(CHUNK_SIZE):
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)
            
            model = pickle.load(open(model_file, 'rb'))
        
        else:
            model = pickle.load(open(model_file, 'rb'))
    
        return model


    def transform(self, input_lines, src_lang='en', **kwargs):
        input_lines = t2t.Transformer.transform(self, input_lines, src_lang=src_lang, **kwargs)
        identifier = self.model 
        embeddings = t2t.Vectorizer.transform(self, input_lines, src_lang='en', **kwargs)
        predictions = identifier.predict(embeddings)
        for prediction in predictions:
            print('Language is {}'.format(prediction))

        return None

    def __init__(self, **kwargs):
        self.model = self._get_model()

