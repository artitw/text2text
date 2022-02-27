import requests
import pickle
import text2text as t2t


class Identifier(t2t.Vectorizer):

    def _get_model():
        """Download and loads model from google drive"""

        file_id = "1pB1GkWiafx23jmND6psg1cUyBqRq0rVY"
        model_file = "identifier.pkl"
        URL = "https://docs.google.com/uc?export=download"
        session = requests.Session()
        response = session.get(URL, params = { 'id' : file_id }, stream = True)
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                params = { 'id' : file_id, 'confirm' : value }
                response = session.get(URL, params = params, stream = True)

        CHUNK_SIZE = 32768
        with open(model_file, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
        
        model = pickle.load(open("identifier.pkl", 'rb'))
        return model

    def transform(self, input_lines, src_lang='en', **kwargs):
        input_lines = t2t.Transformer.transform(self, input_lines, src_lang=src_lang, **kwargs)
        identifier = self._get_model()
        embeddings = t2t.Vectorizer.transform(self, input_lines, src_lang='en', **kwargs)
        predictions = identifier.predict(embeddings)
        for prediction in predictions:
            print('Language is {}'.format(prediction))

        return None