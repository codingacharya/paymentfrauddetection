import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

def load_data(file):
    df = pd.read_csv(file)
    return df

def train_models(df):
    if 'is_fraud' not in df.columns:
        st.error("Dataset must contain an 'is_fraud' column for prediction.")
        return None, None
    
    X = df.drop(columns=['is_fraud'])
    y = df['is_fraud']
    
    # Encode categorical features
    X = X.apply(lambda col: LabelEncoder().fit_transform(col) if col.dtype == 'object' else col)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    
    # Train XGBoost
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_preds)
    
    return (rf_model, rf_acc), (xgb_model, xgb_acc)

def main():
    st.set_page_config(page_title='Online Payment Fraud Detection', layout='wide')
    
    st.title('💳 Online Payment Fraud Detection')
    st.markdown('''Upload your transaction dataset to analyze fraud patterns and predict fraudulent transactions.''')
    
    uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.success("File Uploaded Successfully!")
        
        st.subheader("🔍 Data Overview")
        st.dataframe(df.head())
        
        if 'is_fraud' in df.columns:
            fraud_count = df['is_fraud'].value_counts()
            st.subheader("📊 Fraud Distribution")
            fig_pie = px.pie(names=fraud_count.index, values=fraud_count.values, title="Fraud vs Non-Fraud Transactions", 
                              color=fraud_count.index, color_discrete_map={0: 'blue', 1: 'red'})
            st.plotly_chart(fig_pie)
            
            st.subheader("📈 Transaction Amount Analysis")
            fig_hist = px.histogram(df, x='amount', color='is_fraud', nbins=50,
                                    title='Transaction Amount Distribution', 
                                    color_discrete_map={0: 'blue', 1: 'red'})
            st.plotly_chart(fig_hist)
            
            st.subheader("📊 Correlation Heatmap")
            plt.figure(figsize=(10,6))
            sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
            st.pyplot(plt)
            
            st.subheader("📌 Fraud Transactions Table")
            fraud_df = df[df['is_fraud'] == 1]
            st.dataframe(fraud_df)
            
            st.subheader("🔮 Model Training and Prediction")
            rf_result, xgb_result = train_models(df)
            
            if rf_result and xgb_result:
                st.write(f"🎯 Random Forest Accuracy: {rf_result[1]:.2f}")
                st.write(f"🎯 XGBoost Accuracy: {xgb_result[1]:.2f}")
                
                st.subheader("💡 Predict Fraud on New Data")
                if st.button("Predict on Full Dataset"):
                    df_features = df.drop(columns=['is_fraud']).apply(lambda col: LabelEncoder().fit_transform(col) if col.dtype == 'object' else col)
                    rf_predictions = rf_result[0].predict(df_features)
                    df['RF_Prediction'] = rf_predictions
                    xgb_predictions = xgb_result[0].predict(df_features)
                    df['XGB_Prediction'] = xgb_predictions
                    st.dataframe(df[['is_fraud', 'RF_Prediction', 'XGB_Prediction']])
        else:
            st.warning("Column 'is_fraud' not found in dataset. Please upload a valid dataset.")
    
if __name__ == "__main__":
    main()
