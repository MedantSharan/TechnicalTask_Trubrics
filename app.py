import joblib
import pandas as pd
import streamlit as st


@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pickle")
    X_test = pd.read_csv("X.csv")
    y_test = pd.read_csv("y.csv")
    return model, X_test, y_test


def main():
    st.title("Iris Dataset ML Application")
    model, X_test, _ = load_artifacts()
    
    st.write(model.predict(X_test.iloc[:1]))


if __name__ == '__main__':
    main()
