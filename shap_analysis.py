import time
import shap
from sklearn.model_selection import train_test_split
from preprocessing import CategoricalEncoder, AddFeatures
from flask import Flask, render_template, redirect, request, jsonify,url_for
import pandas as pd
from xgboost import XGBClassifier

# Load model and data outside the function to optimize performance

def load_model():
    model = XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=3.0,
        max_depth=3,
        learning_rate=0.1,
        n_estimators=80,
        colsample_bytree=0.6,
        reg_alpha=0,
        reg_lambda=5,
        n_jobs=-1
    )
    return model

def shap_analysis_function(row_index,customer_id, X, y, model):
    # Existing setup for transformation and model application
    ce = CategoricalEncoder()
    af = AddFeatures()
    X_transformed = ce.fit_transform(X, y)
    X_transformed = af.transform(X_transformed)
    model.fit(X_transformed, y)

    # SHAP analysis setup
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed.iloc[[row_index]])
    shap_html = shap.force_plot(explainer.expected_value, shap_values[0], X_transformed.iloc[row_index], feature_names=X_transformed.columns.tolist(), show=False)

    # Save the plot with a dynamic name based on row number
    filename = f'static/shap_plot_for_CustomerID:{customer_id}.html'
    shap.save_html(filename, shap_html)
    return filename
