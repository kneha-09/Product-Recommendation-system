from os import read
from flask import Flask, render_template,request
import pickle
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


app = Flask(_name_)
@app.route('/')
def main():
  return render_template('index.html')



@app.route("/recommend",methods=['POST'])
def home():
  text=request.form['firstname']
  data=pd.read_csv("static/ratings.csv")
  # path="static/table.html"
  recommend(data,text,10)
  return render_template("table.html")

        


    

def recommend(dataset, songs, amount):
  distance = []
  song = dataset[(dataset.name.str.lower() == songs.lower())].head(1).values[0]
  rec = dataset[dataset.name.str.lower() != songs.lower()]
  for songs in tqdm(rec.values):
    d = 0
    for col in np.arange(len(rec.columns)):
       if not col in [1, 6, 12, 14, 18]:
          d = d + np.absolute(float(song[col]) - float(songs[col]))
    distance.append(d)
  rec['distance'] = distance
  rec = rec.sort_values('distance')
  columns = ['artists', 'name'] 
  var=(rec[columns][:amount])
  df=pd.DataFrame(var)
  html=df.to_html()
  text_file=open("templates/table.html","w")
  text_file.write(html)
  text_file.close()


        

if _name_ == "_main_":
    app.run(debug=True)