# ## load the model

import pickle 

input_file = 'model_C=0.1.bin'
with open(input_file,'rb') as f_in:
    dv,model = pickle.load(f_in)

X = dv.transform([customer])


model.predict_proba(X)[0,1]
