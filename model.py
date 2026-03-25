def model_train():
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
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix,accuracy_score
    from sklearn.neighbors import KNeighborsClassifier
    import numpy as np 
    x=[]
    y=[]
    with open("heart_disease.csv") as f:
        f.readline()
        while True:
            data=f.readline()
            if data=="":
                break
            data=data.strip()
            data=data.split(",")
            if "NA" in data or "" in data:
                continue
            x.append([float(data[0]),float(data[2]),float(data[3]),float(converter(data[4])),float(converter(data[5]))])
            y.append(float(converter(data[20])))
    x=np.array(x).reshape(-1,5)
    y=np.array(y)
    #we will distribute data
    xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=42)
    #Now we will use scaler 
    scaler=StandardScaler()
    xtrain=scaler.fit_transform(xtrain)
    xtest=scaler.transform(xtest)
    #model training
    model=KNeighborsClassifier(n_neighbors=3)
    model.fit(xtrain,ytrain)
    y_pred=model.predict(xtest)
    #model evaluation 
    score=accuracy_score(ytest,y_pred)
    c=confusion_matrix(ytest,y_pred)
    return model,scaler,score,c