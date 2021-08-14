import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS,cross_origin
import pickle

# load model
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
pca=pickle.load(open('pca.pkl','rb'))

# app
app = Flask(__name__)

@app.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")
# routes
@app.route('/pred',methods=['POST','GET']) # route to show the review comments in a web UI
@cross_origin()
def main():
    if request.method == 'GET':
        return (render_template('index.html'))
    if request.method == 'POST':
        try:

       # 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'
            pclass = request.form['Pclass']
            age = request.form['Age']
            sibSp = request.form['SibSp']
            fare = request.form['Fare']
            sex = request.form['Sex']
            parch = request.form['Parch']
            embarked_Q = request.form['Embarked_Q']
            embarked_S = request.form['Embarked_S']
            Age_Group_adult=0
            Age_Group_child=0
            Age_Group_infant=0
            Age_Group_middle_aged=0
            Age_Group_old=0
            Age_Group_senior_citizen=0
            Age_Group_teenager=0
            Age_Group_toddler=0

            if (sex == 'F'):
              sexvar = 1
            else:
              sexvar=0
            age = int(age)
            if age <= 1:
                Age_Group_infant = 1
            elif age <= 4:
                Age_Group_toddler = 1
            elif age <= 13:
                Age_Group_child = 1
            elif age <= 18:
                Age_Group_teenager = 1
            elif age <= 35:
                 pass
            elif age <= 45:
                 Age_Group_adult = 1
            elif age <= 55:
                 Age_Group_middle_aged=1
            elif age <= 65:
                Age_Group_senior_citizen=1
            else:
                Age_Group_old=1

            #input_variables = [[pclass, sexvar, age, sibSp, parch, fare, embarked_Q, embarked_S]]
            #pclass, sexvar,sibSp,parch,fare,embarked_Q,embarked_S,Age_Group_adult,Age_Group_child,Age_Group_infant,
             #Age_Group_middle_aged,Age_Group_old,Age_Group_senior_citizen, Age_Group_teenager,Age_Group_toddler
            input_variables = pd.DataFrame([pd.Series([pclass, sexvar,sibSp,parch,fare,embarked_Q,embarked_S,Age_Group_adult,Age_Group_child,Age_Group_infant,
            Age_Group_middle_aged,Age_Group_old,Age_Group_senior_citizen, Age_Group_teenager,Age_Group_toddler])])
            input_variables = pd.DataFrame(scaler.transform(input_variables))
            #input_pca = pca.transform(input_variables)

            prediction = model.predict(pca.transform(input_variables))
            #print(str(prediction[0]))
            #input_var=np.array([Pclass,Sex,SibSp,Parch,Fare,Embarked_Q,Embarked_S,Age_Group_adult,	Age_Group_child,	Age_Group_infant,	Age_Group_middle_aged,	Age_Group_old,Age_Group_senior_citizen,	Age_Group_teenager,	Age_Group_toddler]).reshape(-1,1)
            #prediction=model.predict(pca.transform(input_var))
            return render_template('results.html',prediction=prediction[0])

        except Exception as e:
            print('The Exception message is: ',e)
            #return 'something is wrong'
            return e
    else:
        return render_template('index.html')

if __name__ == '__main__':
    # app.run(host='127.0.0.1', port=8001, debug=True)
    app.run(debug=True)  # running the app