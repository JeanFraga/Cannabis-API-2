"""
Prediction on string that returns top 20 matches
"""

import joblib
import pandas as pd


def predict_strain(text):
    """
    determine and return 20 id for the strains that fit the description provided
    """
    # Load the compressed models into memory
    modelfile = 'CANNABIS_API/models/NN_MJrec.pkl.zip'
    tfidffile = 'CANNABIS_API/models/tfidf.pkl.zip'
    nn = joblib.load(modelfile)
    tfidf = joblib.load(tfidffile)

    # load the dataframe from pandas
    df = pd.read_csv('CANNABIS_API/models/cannabis-strains.zip')

    # Transform
    text = pd.Series(text)
    vect = tfidf.transform(text)

    # Send to df
    vectdf = pd.DataFrame(vect.todense())

    # Return a list of indexes
    top20 = nn.kneighbors([vectdf][0], n_neighbors=20)[1][0].tolist()


    recommendations_df = df.iloc[top20]
    
    return recommendations_df



