import streamlit as st 
st.title("Heart disease Detected")
st.write("This model you multi feature to detect heart problem")
from model import model_train
def converter(n):
        if n=="High":
            return 2
        elif n=="Medium":
            return 1
        elif n=="Low":
            return 0
        elif n=="Yes":
            return 1
        elif n=="No":
            return 0
        else:
            return None
model,scaler,score,c=model_train()
st.write("Model Accuracy:-  ",score*100,"%")
st.write("Confusion matrix:-    ",c)
age=st.number_input("Age:-")
Bp=st.number_input("Blood Pressure:-")
cl=st.number_input("cholestrol:-")
ex=converter(st.selectbox("Excersice Habit:-",["High","Medium","Low"]))
sm=converter(st.selectbox("Smoking Habit:-",["Yes","No"]))
#prediction
data=[[age,Bp,cl,ex,sm]]
data=scaler.transform(data)
#output
if st.button("Predict"):
    output=model.predict(data)
    if output[0]==1:
        st.write("Yes")
    else:
        st.write("No")