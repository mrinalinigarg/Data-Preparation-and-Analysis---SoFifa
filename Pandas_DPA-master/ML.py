from Train_Test_Split import Train_Test_Split
class ML(Train_Test_Split):
    # Importing relevant packages.
    import pandas as pd
    import pickle
    import matplotlib.pyplot as plt
    from IPython.display import display
    
    def __init__(self):
        # Inheriting Train_Test_Split class.
        super().__init__()
        # Initializing with inherited data.
        self.xtrain = super().xy_train()[0]
        self.ytrain = super().xy_train()[1]
        self.xtest = super().xy_test()[0]
        self.ytest = super().xy_test()[1]
        self.test_dat = super().test_dat()
        
    
    # will return Decision Tree Regressor model.
    def decision_tree(self):
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.metrics import r2_score,mean_squared_error
        self.clf = DecisionTreeRegressor()
        self.clf.fit(self.xtrain, self.ytrain.values.ravel())
        self.clf_predict = self.clf.predict(self.xtest)
        self.r_sq = r2_score(self.ytest, self.clf_predict)
        self.mse = mean_squared_error(self.ytest, self.clf_predict)
        self.rmse = self.np.sqrt(self.mse)
        return self.clf_predict, self.clf, self.r_sq, self.rmse, self.mse
    
    # will return Random Forest Regressor model.
    def random_forest(self):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import r2_score,mean_squared_error
        self.clf = RandomForestRegressor()
        self.clf.fit(self.xtrain, self.ytrain.values.ravel())
        self.clf_predict = self.clf.predict(self.xtest)
        self.r_sq = r2_score(self.ytest, self.clf_predict)
        self.mse = mean_squared_error(self.ytest, self.clf_predict)
        self.rmse = self.np.sqrt(self.mse)
        return self.clf_predict, self.clf, self.r_sq, self.rmse, self.mse
    
    # will return Support Vector Machine(SVM) model.
    def svm(self):
        from sklearn.svm import SVR
        from sklearn.metrics import r2_score,mean_squared_error
        self.clf = SVR()
        self.clf.fit(self.xtrain, self.ytrain.values.ravel())
        self.clf_predict = self.clf.predict(self.xtest)
        self.r_sq = r2_score(self.ytest, self.clf_predict)
        self.mse = mean_squared_error(self.ytest, self.clf_predict)
        self.rmse = self.np.sqrt(self.mse)
        return self.clf_predict, self.clf, self.r_sq, self.rmse, self.mse
    
    # will return Linear Regression model.
    def lr(self):
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score,mean_squared_error
        self.clf = LinearRegression()
        self.clf.fit(self.xtrain, self.ytrain.values.ravel())
        self.clf_predict = self.clf.predict(self.xtest)
        self.r_sq = r2_score(self.ytest, self.clf_predict)
        self.mse = mean_squared_error(self.ytest, self.clf_predict)
        self.rmse = self.np.sqrt(self.mse)
        return self.clf_predict, self.clf, self.r_sq, self.rmse, self.mse
    
    # will return the best out of 4 models.
    def all_models(self):
        self.dict = {'Decision Tree': self.decision_tree(), 'Random Forest': self.random_forest(),
                'Support vector machine(SVM)': self.svm(), 'Linear Regression': self.lr()}
        self.model = ["Decision Tree", "Random Forest", "Support vector machine(SVM)", "Linear Regression"] 
        self.rmse = [self.dict['Decision Tree'][3], self.dict['Random Forest'][3], self.dict['Support vector machine(SVM)'][3], self.dict['Linear Regression'][3]]
        self.rsq = [self.dict['Decision Tree'][2], self.dict['Random Forest'][2], self.dict['Support vector machine(SVM)'][2], self.dict['Linear Regression'][2]]
        self.mse = [self.dict['Decision Tree'][4], self.dict['Random Forest'][4], self.dict['Support vector machine(SVM)'][4], self.dict['Linear Regression'][4]]
        self.data = {'Model':self.model, 'RMSE':self.rmse, "R-square":self.rsq, "MSE":self.mse}
        self.df = self.pd.DataFrame(self.data)
        self.d = self.df.style.apply(lambda x: ['background: lightblue' if i == min(self.df["RMSE"]) else '' for i in self.df["RMSE"]])
        self.best_model = self.dict[self.df[self.df["RMSE"] == min(self.df["RMSE"])]["Model"].values[0]]
        self.bm = self.best_model[1]
        self.display(self.d)
        return self.best_model[1]
    
    # saves the best model in the local system as .pkl file.
    # Initially we are not saving it as it requires around 2.3 GB of storage to store the model, but you can save it using the function save_best_model
    def save_best_model(self):
        self.pkl_filename = "best_model.pkl"
        with open(self.pkl_filename, 'wb') as file:
            self.pickle.dump(self.all_models(), file)
    
    # reads the saved model.
    def read_saved_model(self):
        self.pkl_filename = "best_model.pkl"
        with open(self.pkl_filename, 'rb') as file:
            self.pickle_model = self.pickle.load(file)
        return self.pickle_model
    
    # return the predicted rating of the test data.
    def after_predictions(self):
        import math
        import numpy as np
        # remove the below comment if you want to save the model in your current working directory.
        # self.save_best_model()
        self.predict = self.all_models().predict(self.xtest)
        self.roun = self.np.round(self.predict)
        self.integ = self.roun.astype(int)
        self.xtest.insert(10, "Predicted Overall Rating", self.integ)
        self.xtest.insert(11, "Actual Overall Rating", self.ytest)
        self.xtest.insert(1, "Name", self.test_dat["Name"].values)
        self.xtest.insert(2, "Team", self.test_dat["Team"].values)
        self.xtest.insert(4, "Position", self.test_dat["Position"].values)
        self.xtest.insert(0, "ID", self.test_dat["ID"].values)
        self.xtest.insert(8, "Goal-Keeper stats", self.test_dat["Goal-Keeper stats"].values)
        print("Showing Predicted Overall Rating of first 10 rows.")
        self.display(self.xtest.head(10))
        
        self.ax = self.xtest[["Actual Overall Rating", "Predicted Overall Rating"]][:10].plot(kind='bar')
        self.ax.set_xticklabels(self.xtest["Name"][:10],rotation=45)
        self.ax.set_xlabel("Player Name")
        
        self.ax = self.xtest[["Actual Overall Rating", "Predicted Overall Rating"]][-10:].plot(kind='bar', colormap='Accent')
        self.ax.set_xticklabels(self.xtest["ID"][-10:],rotation=45)
        self.ax.set_xlabel("Player ID")
        self.plt.show()
        return self.xtest
