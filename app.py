from flask import Flask, request, render_template

import numpy as np
import pandas as pd

app = Flask(__name__)#just a module in python(starting point of the api)




column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv("u.data", sep='\t', names=column_names)

movie_titles = pd.read_csv("Movie_Id_Titles")

df = pd.merge(df,movie_titles,on='item_id')

ratings = pd.DataFrame(df.groupby('title')['rating'].mean())#considering average rating of every movie

ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())#adding another column in ratings dataframe in which rating count is taken

moviemat = df.pivot_table(index='user_id',columns='title',values='rating')#constructing a pivot table using user id as index, title as column and rating as values

@app.route('/')#will route URL with the function
def home():
    return render_template('movie1.html')#will take to the html file

@app.route('/predict',methods=['POST'])
def predict():
    movie=request.form['mv']#getting the value from html input
    if str(movie) in moviemat.columns:#checking movie existance in our database
        str(movie)#converting the datatype into string
        user_ratings = moviemat[movie]#taking the user ratings of that movie from the pivot table
        similar_to_movie = moviemat.corrwith(user_ratings)# checking correlation of user ratings of that movie with the user ratings of other movie

        corr_movie = pd.DataFrame(similar_to_movie,columns=['Correlation'])#taking it to the dataframe
        corr_movie.dropna(inplace=True)
        corr_movie = corr_movie.join(ratings['num of ratings'])#we will also consider the rating count because if a movie have good ratings but 
        #rating count is less that movie will not add value
        recom=corr_movie[corr_movie['num of ratings']>100].sort_values('Correlation',ascending=False).iloc[1:6,:]#for the recommendation max correlation and more than 100 user ratings will be considered
        suggested=recom.index#taking the index of the movie which is movie itself
        return render_template('movie1.html', prediction="You Have good choice.... Recommended Movies are...................",prediction1=suggested[0],prediction2=suggested[1],prediction3=suggested[2],
            prediction4=suggested[3],prediction5=suggested[4])
    else:
        return render_template('movie1.html', prediction="We are extremly sorry but this movies is not found in Our Movies Database. Please Check with database movies, thanks.")


    
if __name__ == "__main__":#if this code is running other than python then this command will come into existence
    app.run(debug=True)#means it will show the realtime changes done by the user without stopping the command prompt
#when this above written code is executed then only api will run       
    
    
    
    
