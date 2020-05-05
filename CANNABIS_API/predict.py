"""
Prediction on string that returns top 5 matches
"""

import joblib
import pandas as pd


def predict_strain(text):
    """
    determine and return 5 id for the strains that fit the description provided
    """
    modelfile = 'CANNABIS_API/models/NN_MJrec.pkl.zip'
    tfidffile = 'CANNABIS_API/models/tfidf.pkl.zip'
    nn = joblib.load(modelfile)
    tfidf = joblib.load(tfidffile)

    df = pd.read_csv('CANNABIS_API/models/cannabis-strains.zip')

    # Transform
    text = pd.Series(text)
    vect = tfidf.transform(text)

    # Send to df
    vectdf = pd.DataFrame(vect.todense())

    # Return a list of indexes
    top5 = nn.kneighbors([vectdf][0], n_neighbors=5)[1][0].tolist()

    recommendations_df = df.iloc[top5]
    recommendations_df['index']= recommendations_df.index
    
    return recommendations_df.to_json()



