import argparse as ap
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import warnings
from datetime import datetime as dt
import joblib

nltk.download('punkt')
nltk.download('stopwords')
warnings.filterwarnings("ignore")


def get_args():
    parser = ap.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to json file')
    parser.add_argument('--output', default='./', help='Dir to save output files')

    args = parser.parse_args()

    return args


def clean(text):
    tokens = word_tokenize(text)
    tokens = [w.lower() for w in tokens]

    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]

    words = [word for word in stripped if word.isalpha()]

    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]

    cleaned_text = ' '.join(words)

    return cleaned_text


def run():
    model_name = 'LinearSVC'

    args = get_args()
    print(f'Obecnie używany model: {model_name}')

    print('[INFO] Wczytywanie pliku json do obiektu DateFrame...')
    raw_data = pd.read_json(args.input, lines=True)

    print('[INFO] Skopiowanie potrzebnych kolumn...')
    df = raw_data[['category', 'short_description']].copy()

    print('[INFO] Balansowanie zbioru danych...')
    X = df[['short_description']]
    y = df[['category']]
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    print(f'Liczba rekordów przed zbalansowaniem: {len(df)}')
    print(f'Liczba rekordów po zbalansowaniu: {len(X_resampled)}')

    print('[INFO] Oczyszczanie i przygotowywanie danych do modelu uczącego (może chwilę potrwać)...')
    le = LabelEncoder()
    le.fit(y_resampled)
    data_res = X_resampled
    y_resampled = le.transform(y_resampled)
    data_res['description'] = X_resampled['short_description'].map(lambda x: clean(x))
    data_res['label'] = y_resampled
    data_res = data_res.sample(frac=1).reset_index(drop=True)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data_res['description'])
    y = data_res['label']

    print('[INFO] Podział danych na zbiór treningowy i testowy...')
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    print('[INFO] Trenowanie modelu (może chwilę potrwać)...')
    model = LinearSVC(C=20)
    model.fit(X_train, y_train)
    print(f'Uzysykana dokładność modelu: {model.score(X_test, y_test) * 100:.2f} %')

    print('Czy zapisać wytrenowany model wraz z enkoderami?[Y/n]')
    answer = input('>>>')

    if answer in ['Y', 'y']:
        date = dt.now().strftime("%d_%m_%Y_%H_%M")
        joblib.dump(model, f'{args.output}model_{date}.pkl')
        joblib.dump(le, f'{args.output}labels_{date}.pkl')
        joblib.dump(vectorizer, f'{args.output}vectorizer_{date}.pkl')


if __name__ == '__main__':
    run()
