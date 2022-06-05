#Flask Libraries
from cgi import test
from flask import Flask, redirect, render_template,request
from werkzeug.utils import secure_filename
import os

#DS Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from matplotlib import pyplot as plt
import pickle as pkl

#S3 Libraries
import boto3

app = Flask(__name__)

BUCKET_NAME = 'mlsite-test'
s3 = boto3.client('s3',
                    aws_access_key_id='AKIATBAFUYJBJ3VXTGWY',
                    aws_secret_access_key= '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',
                )


def upload_S3(file):
    fname = file.split('.')
    s3.put_object(
        Bucket=BUCKET_NAME, 
        Key=fname[0]+"/"
    )

    s3.upload_file(
        Bucket = BUCKET_NAME,
        Filename=file,
        Key = fname[0]+"/"+file
    )
    s3.upload_file(
        Bucket = BUCKET_NAME,
        Filename=fname[0]+'-report.txt',
        Key = fname[0]+"/"+fname[0]+'-report.txt'
    )


def train(file):
    df = pd.read_csv(file)
    
    X = df.drop('CLASS',axis = 1)
    y = df['CLASS']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)

    model = GaussianNB()
    model.fit(X_train,y_train)

    return [X_test,y_test,model]


def test(X_test,y_test,model):
    y_pred = model.predict(X_test)
    acc_ = metrics.accuracy_score(y_test,y_pred) 
    prec_= metrics.precision_score(y_test,y_pred)
    rec_ = metrics.recall_score(y_test,y_pred)
    cm_ = metrics.confusion_matrix(y_test,y_pred)
    report = metrics.classification_report(y_test,y_pred)

    return [acc_,prec_,rec_,cm_,report]    

def graph(y_test,y_pred):
    plt.plot(y_test,y_pred)

def createReport(scores,fname):
    name = fname.split('.')
    with open(name[0]+'-report.txt','w') as f:
        f.write('Accuracy : ' + str(scores[0])+'\n')
        f.write('Precision: '+str(scores[1])+'\n')
        f.write('Recall: '+str(scores[2])+'\n')
        f.write('Confusion Matrix: '+str(scores[3])+'\n')
        f.write('Classification report: \n' + scores[4]+'\n')


#Home page
@app.route("/")
@app.route("/home", methods=["GET","POST"])
def home():
    return render_template("home.html")


#Result page
@app.route("/result", methods=["POST","GET"])
def result():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file.save(filename)

            train_res = train(filename)
            scores = test(train_res[0],train_res[1],train_res[2])

            #Generate model report
            createReport(scores,filename)   

            #Upload {dataset and Report} to S3 bucket
            #upload_S3(filename)

            #Confusion Matrix List for table creation
            cm = list()
            cm.append(scores[3][0,0])
            cm.append(scores[3][0,1])
            cm.append(scores[3][1,0])
            cm.append(scores[3][1,1])

            return render_template('result.html',acc = scores[0],prec = scores[1], rec = scores[2], cm = cm)

        else:
            return render_template('home.html')
        
    else:
        return render_template('home.html')

#Run Flask App
if(__name__== '__main__'):
    app.run(host="localhost",port=5000,debug=True)

    