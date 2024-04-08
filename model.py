import shap
import pandas as pd
import numpy as np
import pickle

def load_object(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)
    
city_list = load_object('artifacts/city_list.pkl')
model = load_object('artifacts/xgb_model.pkl')
preprocessor = load_object('artifacts/preprocessor.pkl')
feature_names = load_object('artifacts/feature_names.pkl')

class XGBoostModel:
    def __init__(self, model=model, preprocessor=preprocessor, feature_names=feature_names, city=city_list):
        self.model = model
        self.preprocessor = preprocessor
        self.feature_names = feature_names
        self.city_list = city
    
    def predict(self, input_dict):
        # Convert input dictionary to dataframe
        input_df = pd.DataFrame.from_dict(input_dict, orient='index').transpose()
        
        # Preprocess input dataframe using preprocessor
        X = self.preprocessor.transform(input_df)
        
        # Make prediction using XGBoost model
        y_pred = self.model.predict(X)[0]
        
        # Compute SHAP values for input data
        explainer = shap.Explainer(self.model, feature_names=self.feature_names)
        shap_values = explainer(X)
        img = shap.plots.waterfall(shap_values[0], show=False)

        if y_pred == 1:
            # Sort the absolute SHAP values in descending order
            sorted_idx = np.argsort(shap_values.values[0])[::-1]

            # Extract the names of the top 2 features with highest SHAP values
            top_2_features =  np.take(feature_names, sorted_idx)[:2]

            feedback = f"Based on the provided information, the following are top 2 factors may increase the likelihood of a churn: {', '.join(top_2_features)}. Please consider addressing these issues to improve customer retention."
        else:
            feedback = "Based on the provided information, we do not see any major issues that would result in a high likelihood of churn."

        return y_pred, img, feedback
