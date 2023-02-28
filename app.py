from flask import Flask, request, render_template
import pickle
import numpy as np
from urllib.parse import urlparse
from sklearn.impute import SimpleImputer

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

important_features = ['length_url', 'length_hostname', 'nb_dots', 'nb_hyphens', 'nb_at', 'nb_slash', 'nb_comma', 'nb_dollar', 'nb_space', 'http_in_path']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    url = request.args.get('url')
    result = predict_phishing(url)
    return {'result': result}

def predict_phishing(url):
    parsed_url = urlparse(url)
    domain_name = parsed_url.netloc
    path = parsed_url.path
    features = np.array([
        len(parsed_url.geturl()), # length_url
        len(domain_name), # length_hostname
        parsed_url.geturl().count('.'), # nb_dots
        parsed_url.geturl().count('-'), # nb_hyphens
        parsed_url.geturl().count('@'), # nb_at
        parsed_url.geturl().count('/'), # nb_slash
        parsed_url.geturl().count(','), # nb_comma
        parsed_url.geturl().count('$'), # nb_dollar
        parsed_url.geturl().count(' '), # nb_space
        int(bool('http' in path)), # http_in_path
    ])
    
    # Filter features according to important_features
    features = features[np.isin(important_features, ['length_url', 'length_hostname', 'nb_dots', 'nb_hyphens', 'nb_at', 'nb_slash', 'nb_comma', 'nb_dollar', 'nb_space', 'http_in_path'])]
    
    features = np.nan_to_num(features)
    prediction = model.predict(features.reshape(1, -1))[0]
    print(prediction)
    if prediction == 'legitimate':
        return "Sitio web seguro"
    else:
        return "Sitio web no seguro"

if __name__ == '__main__':
    app.run(debug=True)
