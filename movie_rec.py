# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 22:36:23 2020

@author: shalo
"""


from flask import Flask, request, render_template

import numpy as np
import pandas as pd

app = Flask(__name__)


path1=r"C:\Users\shalo\Desktop\ML stuffs\recommender system\collabrating\Movie-Recommender-in-python simple\u.data"
path2=r"C:\Users\shalo\Desktop\ML stuffs\recommender system\collabrating\Movie-Recommender-in-python simple\Movie_Id_Titles"

column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv(path1, sep='\t', names=column_names)

movie_titles = pd.read_csv(path2)

df = pd.merge(df,movie_titles,on='item_id')

ratings = pd.DataFrame(df.groupby('title')['rating'].mean())

ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())

moviemat = df.pivot_table(index='user_id',columns='title',values='rating')

@app.route('/')#will route URL with the function
def home():
    return render_template('movie1.html')

@app.route('/predict',methods=['POST'])
def predict():
    movie=request.form['mv']
    str(movie)
    user_ratings = moviemat[movie]
    similar_to_movie = moviemat.corrwith(user_ratings)
    
    corr_movie = pd.DataFrame(similar_to_movie,columns=['Correlation'])
    corr_movie.dropna(inplace=True)
    corr_movie = corr_movie.join(ratings['num of ratings'])
    recom=corr_movie[corr_movie['num of ratings']>100].sort_values('Correlation',ascending=False).iloc[1:6,:]
    suggested=recom.index

    
    return render_template('movie1.html', prediction="Recommended Movies are...................",prediction1=suggested[0],prediction2=suggested[1],prediction3=suggested[2],
        prediction4=suggested[3],prediction5=suggested[4])


    
if __name__ == "__main__":#if this code is running other than python then this command will come into existence
    app.run(debug=True)#means it will show the realtime changes done by the user without stopping the command prompt
    
    
    
    
    