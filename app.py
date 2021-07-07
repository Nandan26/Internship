from flask import Flask, request, render_template
import sklearn
import joblib
import pandas as pd
import numpy as np
import statsmodels.api as sm
from flask_cors import cross_origin,CORS 
#for enable cross platform communication

app = Flask(__name__)
#for Next day prediction
scaler= joblib.load('scaler.pkl')
model1 = joblib.load('Model1.pkl')

#for next minute prediction
scaler2= joblib.load('scaler2.pkl') 
model2=joblib.load(open('Model2.pkl','rb'))
#just another way of doing it
#CORS(app)

@app.route("/")
@cross_origin()
def home():
    return render_template("index.html")




@app.route("/predict", methods = ["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":

        # Open Price
        op = float(request.form["Open_price"])
        
        # High Price
        hp = float(request.form["High_price"])

        # Low Price
        lp = float(request.form["Low_price"])
        
        
        adj = float(request.form["Adj_Close"])
        
        
        vm = float(request.form["Volume"])
        
        # Date
        Date = request.form["Date_Time"]
        Day = int(pd.to_datetime(Date,format="%Y-%m-%d").day)
        Year = int(pd.to_datetime(Date,format="%Y-%m-%d").year)
        Month = int(pd.to_datetime(Date,format="%Y-%m-%d").month)
        DayofWeek = int(pd.to_datetime(Date,format="%Y-%m-%d").dayofweek)
        
        # Previous day Open Price
        pop = float(request.form["popen"])

        # Previous day close Price
        pcp = float(request.form["pclose"]) 
        scaled=scaler.transform([[op,hp,lp,adj,vm,op-pcp,op-pop,Day,Year,Month,DayofWeek]])
        prediction=model1.predict(scaled)

        if prediction[0]==1:
            output2= "Up"
        else:
            output2= "Down"

        scaled=scaler2.transform([[op,hp,lp]])
        dataset = pd.DataFrame({'const' : 1,'op': scaled[:, 0], 'hp': scaled[:, 1], 'lp': scaled[:, 2]})
        

        pred=model2.predict(dataset)
        if pred[0] > 0.5:
            output= "Up"
        else:
            output= "Down"
        return render_template('index.html',prediction_text="Your Stock will go {} next minute  Your Stock will go {} next day".format(output,output2))


    return render_template("index.html")




if __name__ == "__main__":
    app.run(debug=True)