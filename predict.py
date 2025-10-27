

import pickle 
from flask import Flask

input_file = 'model_C=0.1.bin'

#load the model
with open(input_file,'rb') as f_in:
    dv,model = pickle.load(f_in)
    
app = Flask('Churn Prediction')

@app.route('/predict', methods=['POST'])
#function to predict churn probability
def predict(customer):
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0,1]
    return y_pred
