
from flask import Flask, render_template, request 
import numpy as np
import joblib

appf=Flask(__name__,template_folder="Templates")

model = joblib.load('C:/Users/Abd AL-Rahman/Desktop/new-project/graduation project modified/model.grpm4',"readwrite")
scaler = joblib.load('C:/Users/Abd AL-Rahman/Desktop/new-project/graduation project modified/scaler.grpm4',"readwrite")

@appf.route('/')
def home():
    return render_template('indexg.html')

@appf.route('/predict', methods=['POST'])
def predict():
    
    area_income = request.form["area income"]
    house_age = request.form["house age"]
    area_population = request.form["population"]
    no_bedrooms = request.form["no.bedrooms"]
    no_rooms = request.form['no.rooms']
    x=np.array([area_income,house_age,no_rooms,area_population,no_bedrooms])
    x2=scaler.transform([x])
    
    price=model.predict(x2)
    
    return render_template('indexg.html',prediction_text= "Price of House in dollars : {}$".format(price))
    
if __name__ == "__main__":
    
    
    appf.debug=True
    appf.run()

    



