

import pickle 
from flask import Flask,request,jsonify

input_file = 'model_C=0.1.bin'

#load the model
with open(input_file,'rb') as f_in:
    dv,model = pickle.load(f_in)
    
app = Flask('Churn Prediction')

@app.route('/predict', methods=['POST'])
#function to predict churn probability
def predict(customer):
    customer = request.get_json()
    
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0,1]
    churn = (y_pred >= 0.5)
    
    result = {
        'churn_probability': y_pred,
        'churn': churn
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port = 9696)
