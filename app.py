import joblib
import pandas as pd
import streamlit as st
from trubrics.integrations.streamlit import FeedbackCollector


@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pickle")
    X_test = pd.read_csv("X.csv")
    y_test = pd.read_csv("y.csv")
    return model, X_test, y_test

#Create a function to handle feature inputs
def make_inputs():
   sepalWidth =  st.number_input("Sepal Width (cm)", step=0.1 )
   sepalLength = st.number_input("Sepal Length (cm)", step=0.1 )
   petalWidth =  st.number_input("Petal Width (cm)", step=0.1 )
   petalLength = st.number_input("Petal Length (cm)", step=0.1 )
   return petalWidth, petalLength, sepalWidth, sepalLength

def deploy_thumbs_feedback():
    collector = FeedbackCollector(model="model")
    feedback = collector.st_feedback(feedback_type="thumbs")
    return feedback



def main():
    st.title("Iris Dataset ML Application")
    model, X_test, _ = load_artifacts()
    petalWidth, petalLength, sepalWidth, sepalLength = make_inputs()

    if st.button('Predict'):
        features = [sepalLength, sepalWidth, petalLength, petalWidth]
        prediction = model.predict([features])[0]
        st.write('Prediction:', prediction)
    feedback = deploy_thumbs_feedback()


if __name__ == '__main__':
    main()
