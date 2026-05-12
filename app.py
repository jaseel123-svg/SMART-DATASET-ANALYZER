import io
import joblib
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    confusion_matrix
)
# st.set_page_config(
#     page_title="Dataset Analyzer",
#     layout="wide"
# )
st.title("SMART DATASET ANALYZER")
st.sidebar.title("Project Features")

st.sidebar.write("✔ Dataset Upload")
st.sidebar.write("✔ Data Cleaning")
st.sidebar.write("✔ Visualization")
st.sidebar.write("✔ Machine Learning")
st.sidebar.write("✔ Model Comparison")
st.sidebar.write("✔ Prediction System")
st.sidebar.write("✔ Download Cleaned CSV")

file=st.file_uploader("upload CSV",type=["csv"])

if file:
    df=pd.read_csv(file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    st.write("Shape:",df.shape)
    
    st.subheader("Column Data Types")
    st.write(df.dtypes)

    # missing values
    st.subheader("Missing values")
    missing=df.isnull().sum()
    missing=missing[missing>0]
    missing_df=missing.reset_index()
    missing_df.columns=["Column","Missing Values"]
    st.dataframe(missing_df)

    #handle missing values

    for col in df.columns:
        # categorical columns
        if df[col].dtype=="object":
            df[col]=df[col].fillna(
                df[col].mode()[0]
            )
        
        #numerical columns
        else:
            df[col]=df[col].fillna(
                df[col].mean()
            )
    
    ## download csv
    st.subheader("Download Cleaned Dataset")

    csv = df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Download Cleaned CSV",
        data=csv,
        file_name="cleaned_dataset.csv",
        mime="text/csv"
    )

    #statistics

    st.subheader("dataset statistics")
    st.write(df.describe())

    #histogram section

    st.subheader("histogram Visualization")
    numeric_col= df.select_dtypes(include='number').columns
    selected_col=st.selectbox(
        "Select Numeric Column",
        numeric_col
    )
    fig=px.histogram(
        df,
        x=selected_col,
        title=f"histogram of {selected_col}"
    )
    st.plotly_chart(fig)

    #scatter plot

    st.subheader("scatter plot")
    x_axis=st.selectbox(
        "select X-axis",
        numeric_col
    )
    y_axis=st.selectbox(
        "Select Y-axis",
        numeric_col,
        index=1
    )
    fig_scatter =px.scatter(
        df,
        x=x_axis,
        y=y_axis,
        title=f"{x_axis} vs {y_axis}"
    )

    st.plotly_chart(fig_scatter)

    ## correlation heatmap

    st.subheader("Correlation Heatmap")
    corr=df.corr(numeric_only=True)
    fig2,ax=plt.subplots(figsize=(8,5))
    sns.heatmap(
        corr,
        annot=True,
        cmap="Blues",
        ax=ax
    )
    st.pyplot(fig2)

## machine learning
    st.subheader("Machine Learning Model")
    target =st.selectbox(
        "Select  Target Column",
        df.columns
    )

    ## features+target
    X=df.drop(columns=[target])
    y=df[target]

    ## handle categorical columns
    for col in X.select_dtypes(include='object').columns:
        le=LabelEncoder()
        X[col]=le.fit_transform(X[col])

    # encode target if categorical

    if y.dtype=="object":
        target_encoder=LabelEncoder()
        y=target_encoder.fit_transform(y)

    #detect problem type
    unique_values= len(
        pd.Series(y).unique()
    )
    if unique_values<=10:
        problem_type="Classification"
    else:
        problem_type="Regression"

    st.write(
        "Detected Problem Type:",
        problem_type
    )

    # model selection
    if problem_type=="Regression":
        model_name=st.selectbox(
            "Select Regression Model",
            [
                "Linear Regression",
                "Decision Tree Regressor",
                "Random Forest Regressor"
            ]
        )
    else:
        model_name=st.selectbox(
            "Select Classification Model",
            [
                "Logistic Regression",
                "Decision Tree Classifier",
                "Random Forest Classifier"
            ]
        )
    ## train test split

    X_train,X_test,y_train,y_test=train_test_split(
        X,y,test_size=0.2,
        random_state=42
    )
    ## model creation

    model=None
    # regression model
    if model_name=="Linear Regression":
        model=LinearRegression()
    elif model_name=="Decision Tree Regressor":
        model=DecisionTreeRegressor()
    elif model_name=="Random Forest Regressor":
        model=RandomForestRegressor()
    # classification model
    elif model_name=="Logistic Regression":
        model=LogisticRegression()
    elif model_name=="Decision Tree Classifier":
        model=DecisionTreeClassifier()
    elif model_name=="Random Forest Classifier":
        model=RandomForestClassifier()

    ## train model
    model.fit(
        X_train,
        y_train
    )
    predictions=model.predict(X_test)

    ## result
    st.subheader("Model Results")

    ## regression results
    if problem_type=="Regression":
        mae=mean_absolute_error(
            y_test,
            predictions
        )
        fig3=px.scatter(
            x=y_test,
            y=predictions,
            labels={
                "x": "Actual Values",
                "y": "Predicted Values"
            },
            title="Actual vs Predicted"
        )
        st.write("Mean Absolute Error:", mae)
        st.plotly_chart(fig3)

    ## classification result
    else:
        accuracy=accuracy_score(
            y_test,
            predictions
        )
        st.write(
            "Accuracy:",
            accuracy
        )
        cm= confusion_matrix(
            y_test,
            predictions
        )
        fig4,ax2=plt.subplots(
            figsize=(6,4)
        )
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax2
        )
        ax2.set_title("Confusion Matrix")
        st.pyplot(fig4)
    ## download trained model
    st.subheader("Download Trained Model")
    model_bytes=io.BytesIO()
    joblib.dump(model,model_bytes)
    st.download_button(
        label="Download Model",
        data=model_bytes.getvalue(),
        file_name="trained_model.pkl",
        mime="application/octet-stream"
    )
    ## user prediction section
    st.subheader("Make predictions")
    input_data={}

    for col in X.columns:

    # categorical columns
        if df[col].dtype=="object":
            options=df[col].unique().tolist()
            selected=st.selectbox(
                f"Select {col}",
                options
            )
            # encode selected value
            le=LabelEncoder()
            le.fit(df[col])
            input_data[col]=le.transform([selected])[0]

    # numerical columns
        else:
            input_data[col]=st.number_input(
                f"Enter {col}",
                value=float(df[col].mean())
            )

    input_df=pd.DataFrame([input_data])
    if st.button("Predict"):
        prediction=model.predict(input_df)
        if problem_type=="Classification":
            if target_encoder is not None:
                prediction=target_encoder.inverse_transform(
                    prediction.astype(int)
                )
        st.success(f"Prediciton : {prediction[0]}")


else:
    st.info("Please upload a CSV file")

    

