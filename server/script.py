
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import time
import warnings
import os
import numpy as np
import flask
import pickle
from flask import Flask, render_template, request

#creating instance of the class
app=Flask(__name__)

#to tell flask what url shoud trigger the function index()
@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
	data = pd.read_csv('bc_data.csv', index_col=False)
	data.head(5)
	data['diagnosis'] = data['diagnosis'].apply(lambda x: '1' if x == 'M' else '0')
	data = data.set_index('id')
	del data['Unnamed: 32']
	num_folds=10
	Y = data['diagnosis'].values
	X = data.drop('diagnosis', axis=1).values

	X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size = 0.20, random_state=21)
	pipelines = []

	pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART',
                                                                        DecisionTreeClassifier())])))
	pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()),('SVM', SVC( ))])))
	pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('NB',
	                                                                      GaussianNB())])))
	pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN',
	                                                                       KNeighborsClassifier())])))
	results = []
	names = []
	with warnings.catch_warnings():
	    warnings.simplefilter("ignore")
	    kfold = KFold(n_splits=num_folds, random_state=123)
	    for name, model in pipelines:
	        start = time.time()
	        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	        end = time.time()
	        results.append(cv_results)
	        names.append(name)

	with warnings.catch_warnings():
	    warnings.simplefilter("ignore")
	    scaler = StandardScaler().fit(X_train)

	    rescaledX = scaler.transform(X_train)

	    c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]

	    kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']

	    param_grid = dict(C=c_values, kernel=kernel_values)

	    model = SVC()

	    kfold = KFold(n_splits=num_folds, random_state=21)

	    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=kfold)

	    grid_result = grid.fit(rescaledX, Y_train)

	    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

	    means = grid_result.cv_results_['mean_test_score']

	    stds = grid_result.cv_results_['std_test_score']

	    params = grid_result.cv_results_['params']
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		scaler = StandardScaler().fit(X_train)
	X_train_scaled = scaler.transform(X_train)
	model = SVC(C=2.0, kernel='rbf')
	model.fit(X_train_scaled, Y_train)


# In[28]:


# estimate accuracy on test dataset
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		X_test_scaled = scaler.transform(X_test)
	predictions = model.predict(X_test_scaled)

	darray = np.array(data.describe())
	data.describe()

	mean = np.array(darray[1])
	var = np.array(darray[2])

	if (request.method=="POST" ):
		radius_mean=request.form['radius_mean']
		texture_mean=request.form['texture_mean']
		perimeter_mean=request.form['perimeter_mean']
		area_mean=request.form['area_mean']
		smoothness_mean=request.form['smoothness_mean']
		compactness_mean=request.form['compactness_mean']
		concavity_mean=request.form['concavity_mean']
		concave_points_mean=request.form['concave_points_mean']
		symmetry_mean=request.form['symmetry_mean']
		fractal_dimension_mean=request.form['fractal_dimension_mean']
		radius_se=request.form['radius_se']
		texture_se=request.form['texture_se']
		perimeter_se=request.form['perimeter_se']
		area_se=request.form['area_se']
		smoothness_se=request.form['smoothness_se']
		compactness_se=request.form['compactness_se']
		concavity_se=request.form['concavity_se']
		concave_points_se=request.form['concave_points_se']
		symmetry_se=request.form['symmetry_se']
		fractal_dimension_se=request.form['fractal_dimension_se']
		radius_worst=request.form['radius_worst']
		texture_worst=request.form['texture_worst']
		perimeter_worst=request.form['perimeter_worst']
		area_worst=request.form['area_worst']
		smoothness_worst=request.form['smoothness_worst']
		compactness_worst=request.form['compactness_worst']
		concavity_worst=request.form['concavity_worst']
		concave_points_worst=request.form['concave_points_worst']
		symmetry_worst=request.form['symmetry_worst']
		fractal_dimension_worst=request.form['fractal_dimension_worst']
		data=[int(radius_mean),int(texture_mean),int(perimeter_mean),int(area_mean),int(smoothness_mean),int(compactness_mean),int(concavity_mean),int(concave_points_mean),int(symmetry_mean),int(fractal_dimension_mean),int(radius_se),int(texture_se),int(perimeter_se),int(area_se),int(smoothness_se),int(compactness_se),int(concavity_se),int(concave_points_se),int(symmetry_se),int(fractal_dimension_se),int(radius_worst),int(texture_worst),int(perimeter_worst),int(area_worst),int(smoothness_worst),int(compactness_worst),int(concavity_worst),int(concave_points_worst),int(symmetry_worst),int(fractal_dimension_worst)]

		inpt = np.array((data))


# In[80]:


		inpt = inpt.reshape(1, 30)
		inpt = np.divide(np.subtract(inpt,mean),var)
		ans = int(model.predict(inpt))
		print("Melignant" if ans == 1 else "Benign")
				
		return render_template('results.html',prediction=ans)
	

        
     
if __name__ == "__main__":
    app.run(debug=True)
