"""
Prediction on string that returns top 5 matches
"""

from sklearn.externals import joblib
import pandas as pd


def predict_strain(text):
    """
    determine and return 5 id for the strains that fit the description provided
    """
    modelfile = 'CANNABIS_API/models/NN_MJrec.pkl'
    tfidffile = 'CANNABIS_API/models/tfidf.pkl'
    nn = joblib.load(modelfile)
    tfidf = joblib.load(tfidffile)

    # Transform
    text = pd.Series(text)
    vect = tfidf.transform(text)

    # Send to df
    vectdf = pd.DataFrame(vect.todense())

    # Return a list of indexes
    top5 = nn.kneighbors([vectdf][0], n_neighbors=5)[1][0].tolist()
    
    return top5



