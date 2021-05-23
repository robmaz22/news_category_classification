import argparse as ap
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string


def get_args():
    parser = ap.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to txt file')
    parser.add_argument('--model', default='model.pkl', help='Path to model')
    parser.add_argument('--labels', default='labels.pkl', help='Path to labels encoder')
    parser.add_argument('--vectorizer', default='vectorizer.pkl', help='Path to vectorizer')
    parser.add_argument('--gui', store=True, help='Graphic interface')

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
    args = get_args()

    with open(args.input, 'r') as file:
        text = file.read()
        text.replace('\n', '')

    model = joblib.load(args.model)
    le = joblib.load(args.labels)
    vectorizer = joblib.load(args.vectorizer)

    cleaned = clean(text)
    x = vectorizer.transform([cleaned])
    pred = model.predict(x)

    category = le.inverse_transform(pred)
    print(f'Category: {category}')


if __name__ == '__main__':
    run()
