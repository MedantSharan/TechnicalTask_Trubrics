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
   #Take user inputs and store them in variables 
   sepalWidth =  st.number_input("Sepal Width (cm)", step=0.1 )
   sepalLength = st.number_input("Sepal Length (cm)", step=0.1 )
   petalWidth =  st.number_input("Petal Width (cm)", step=0.1 )
   petalLength = st.number_input("Petal Length (cm)", step=0.1 )
   #Return the input variables
   return petalWidth, petalLength, sepalWidth, sepalLength

#Create a function to make a feedback system using Trubrics
def deploy_thumbs_feedback():
    #Use a feedback collector object to create a feedback system
    collector = FeedbackCollector(model="model")

    #Store the feedback in a local variable and select the type of feedback system needed
    feedback = collector.st_feedback(feedback_type="thumbs")
    return feedback



def main():
    st.title("Iris Dataset ML Application")
    #Add some information for the user to use the application
    st.write("You can input the desired parameters and click on predict to see a prediction of the type of iris flower the input parameters may represent. After a prediction, you can use the thumbs-up and thumbs-down button to give feedback which is saved locally.")
    model, X_test, _ = load_artifacts()
    petalWidth, petalLength, sepalWidth, sepalLength = make_inputs()

    #Make a button to make predictions based on data given by the user
    if st.button('Predict'):
        features = [sepalLength, sepalWidth, petalLength, petalWidth]
        prediction = model.predict([features])[0]
        st.write('Prediction:', prediction)
    #Deploy the feedback system created
    feedback = deploy_thumbs_feedback()


if __name__ == '__main__':
    main()
