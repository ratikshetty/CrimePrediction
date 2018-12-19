import os
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression 
from datetime import datetime
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from flask import Flask, render_template, url_for, request

from sklearn.preprocessing import Imputer, LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion

from datetime import datetime

import math

CHICAGO_PATH="crime"

def load_data(gap_path = CHICAGO_PATH):
    data = os.path.join(gap_path,"Crimes.csv")
    return pd.read_csv(data)
chic = load_data()

# print(chic)

def hours(c):
    if c<=3:
        return 1
    elif c>3 and c<=7:
        return 2
    elif c>7 and c<=11:
        return 3
    elif c>11 and c<=15:
        return 4
    elif c>15 and c<=19:
        return 5
    elif c>19 and c<=23:
        return 6


chic_2015 = chic['Year'] == 2015
chic_15= chic[chic_2015]
chic_train, chic_test = train_test_split(chic_15, test_size=0.20)

primary_labels=chic_test['Primary Type']


year = lambda x: datetime.strptime(x, "%m/%d/%Y %H:%M:%S %p" ).month
months = chic_test['Date'].map(year)
mon=chic_train['Date'].map(year)
month = lambda x: datetime.strptime(x,"%m/%d/%Y %I:%M:%S %p" ).hour
times= chic_test['Date'].map(month)
hour= chic_train['Date'].map(month)
days = lambda x: datetime.strptime(x,"%m/%d/%Y %H:%M:%S %p" ).weekday()
day= chic_test['Date'].map(days)
day_train=chic_train['Date'].map(days)
mins = lambda x: datetime.strptime(x,"%m/%d/%Y %H:%M:%S %p" ).minute
min_test= chic_test['Date'].map(mins)
min_train=chic_train['Date'].map(mins)

chic_test['month']= months
chic_test['day']= day
chic_test['hour']=times
chic_test['mintues']=min_test

chic_train['month']=mon
chic_train['day']=day_train
chic_train['hour']=hour
chic_train['mintues']= min_train

chic_months_15_train= chic_train[['month','day','Community Area']].copy()
 
chic_months_15_test= chic_test[['month','day','Community Area']].copy()
targets=chic_train['hour']
lin_regs= LinearRegression()
lin_regs.fit(chic_months_15_train, targets)
y=lin_regs.predict(chic_months_15_test)

temp= chic_test['hour']
mse=mean_squared_error(temp,y)

chic_mon_train=chic_train['day']==0
chic_monday_train = chic_train[chic_mon_train]
print(y)

rep={'THEFT': 2, 'BATTERY': 1, 'CRIMINAL DAMAGE': 1, 'NARCOTICS': 2, 'OTHER OFFENSE': 3, 'ASSAULT': 1, 'DECEPTIVE PRACTICE': 2, 'BURGLARY': 2, 'MOTOR VEHICLE THEFT': 2, 'ROBBERY': 1, 'HOMICIDE': 1, 'CRIMINAL TRESPASS': 3, 'INTERFERENCE WITH PUBLIC OFFICER': 2, 'INTIMIDATION': 2, 'KIDNAPPING': 1, 'LIQUOR LAW VIOLATION': 3, 'HUMAN TRAFFICKING': 1, 'CONCEALED CARRY LICENSE VIOLATION': 2, 'NON - CRIMINAL': 3, 'NON-CRIMINAL': 3, 'OBSCENITY': 3, 'OFFENSE INVOLVING CHILDREN': 2, 'OTHER NARCOTIC VIOLATION': 2, 'GAMBLING': 2, 'PROSTITUTION': 2, 'PUBLIC INDECENCY': 4, 'PUBLIC PEACE VIOLATION': 3, 'CRIM SEXUAL ASSAULT': 1, 'SEX OFFENSE': 1, 'STALKING': 4, 'ARSON': 4, 'WEAPONS VIOLATION': 4}

chic_trains=chic_train.applymap(lambda s: rep.get(s) if s in rep else s)
chic_train1=chic_trains['hour'].replace([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],[1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6])
chic_trains['time_grp']=chic_train1
chic_tests=chic_test.applymap(lambda s: rep.get(s) if s in rep else s)
chic_tests1=chic_tests['hour'].replace([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],[1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6])
chic_tests['time_grp']=chic_tests1

chic_top_10_trains= chic_trains['Primary Type'] <=4
chic_10_trains=chic_trains[chic_top_10_trains]
chic_top_10_tests= chic_tests['Primary Type'] <=4
chic_10_tests=chic_tests[chic_top_10_tests]

chic_months_15_train1= chic_10_trains[['month','day','Community Area','time_grp']].copy()
 
chic_months_15_test1= chic_10_tests[['month','day','Community Area','time_grp']].copy()
targets1=chic_10_trains['Primary Type']
forest_reg = RandomForestRegressor()
forest_reg.fit(chic_months_15_train1,targets1)
y1=forest_reg.predict(chic_months_15_test1)
lin_regs1= LinearRegression()
lin_regs1.fit(chic_months_15_train1, targets1)
y2=lin_regs1.predict(chic_months_15_test1)
dec_reg = DecisionTreeRegressor()
dec_reg.fit(chic_months_15_train1,targets1)
y3=dec_reg.predict(chic_months_15_test1)

temp1= chic_10_tests['Primary Type']
ms=mean_squared_error(temp1,y1)
temp1= chic_10_tests['Primary Type']
ms1=mean_squared_error(temp1,y2)
ms2=mean_squared_error(temp1,y3)
lin_rmse=np.sqrt(ms1)
for_rmse=np.sqrt(ms)
dec_rmse=np.sqrt(ms2)

#print(ms)

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('Home.html')

@app.route('/predict', methods = ['POST'])
def predict():
    if request.method=="POST":
        locality=request.form['Locality']
        print(request.form['selectedDate'])
        month=request.form['selectedDate']
        month1=request.form['selectedDate']
        month1=datetime.strptime(month1, '%m/%d/%Y').date()
        day=(month1.weekday())
        month=month[:2]
        date=request.form['selectedDate']
        date=date[3:5]
        print(month)
        print(date)
        print(request.form['selectedTime'])
        hour=request.form['selectedTime']
        hour=hour[:2]
       
        minute=request.form['selectedTime']
        minute=minute[3:]
        print(hour)
        print(minute)
        h=hours(int(hour))
        pre=[int(month),int(day),int(locality),int(h)]
        l=forest_reg.predict([pre])
        l1=lin_regs1.predict([pre])
        l2=dec_reg.predict([pre])
        def round_half_down(n, decimals=0):
                multiplier = 10 ** decimals
                return math.ceil(n*multiplier - 0.5) / multiplier

        res=round_half_down(int(l2))
        print(locality)
        crimes = ["","VIOLENT CRIME","NORMAL CRIMES","OTHER OFFENSE",""]

        print(res)
        return render_template('Home.html',result=crimes[int(res)], area=locality, date=(request.form['selectedDate']), time=(request.form['selectedTime']))
    return render_template('Home.html')



#print(l1)
#print(l)


if __name__ == '__main__':
    app.run(debug = True)
