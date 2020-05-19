"""
Prediction on string that returns top 20 matches
"""

import joblib
import pandas as pd



def predict_strain(text, df):
    """
    determine and return 20 id for the strains that fit the description provided
    """
    # Load the compressed models into memory
    modelfile = 'CANNABIS_API/models/NN_MJrec.pkl.zip'
    tfidffile = 'CANNABIS_API/models/tfidf.pkl.zip'
    nn = joblib.load(modelfile)
    tfidf = joblib.load(tfidffile)

    # Transform
    text = pd.Series(text)
    vect = tfidf.transform(text)

    # Send to df
    vectdf = pd.DataFrame(vect.todense())

    # Return a list of indexes
    top20 = nn.kneighbors([vectdf][0], n_neighbors=20)[1][0].tolist()

    # locs the the top20 predictions into new DF
    recommendations_df = df.iloc[top20]
    
    return recommendations_df

def similar_strain(strain, df_token):
    """
    processes the strain if it exists in the dataframe to get it ready for 'predict_strain'
    """
    temp = df_token[df_token['Strain']==strain]
    text = temp.tokens
    return text


