import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold,cross_val_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import export_graphviz
from io import StringIO
from IPython.display import Image
from pydot import graph_from_dot_data
import graphviz
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from mlxtend.evaluate import bias_variance_decomp
from adspy_shared_utilities import (plot_class_regions_for_classifier_subplot)
import warnings
warnings.filterwarnings("ignore")
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
#%%
plt.rc('font', size=10)

df = pd.read_csv(r"C:\Users\Jerry\Desktop\DS 2022 PROJECT\Dataset.csv", index_col=None, na_values='-')

# Columns 'File Name' and 'source' does not have significant effect on the response variable 'channels', so dropping column 'Filename' and 'source'...
df = df.drop(['File Name', 'Source'], axis = 1)

# Rows 214, 248, 371 have many missing values in the dataset, so removing these rows...
df = df.drop(index=[214, 248, 371], axis =0)

# Removing percentage symbols from data....
for i in df.columns[:-1]:
    df[i] = [float(str(j).replace('%', '')) for j in df[i]]

df_original = df.copy()

# Label encoding the categorical response variable...
df['Channels'] = LabelEncoder().fit_transform(df['Channels']) # stereo = 1, mono = 0....

# To convert categorical predictors...
df = pd.get_dummies(df, columns=['Sample Rate (Hz)'])

# Handling missing values...

# Using median values for missing values...
col_missing_values = [str(i) for i in df.columns]
temp = []
for j in col_missing_values:
    for i in range(len(df[j])):
        try:
            if str(df[j][i]) != 'nan':
                temp.append(float(df[j][i]))
        except KeyError:
            pass

    median = np.median(temp)
    for i in range(len(df[j])):
        try:
            if str(df[j][i]) == 'nan':
                df[j][i] = median
        except KeyError:
            pass

# Using mean values for missing values...
for j in col_missing_values:
    mean = np.mean(df[j])
    for i in range(len(df[j])):
        try:
            if str(df[j][i]) == 'nan':
                df[j][i] = mean
        except KeyError:
            pass

# Display first 10 rows...
df.head(n=10)

# Rearranging columns in order...
temp_df = df['Channels']
df = df.drop(['Channels'], axis = 1)
df = df.join(temp_df)

X_data = df.iloc[:, :-1]
Y_data = df['Channels']

# # Creating scatter plot matrix...
pd.plotting.scatter_matrix(X_data, c = Y_data, alpha=0.5)      # c argument gives colors to the datapoints based on the Y values...

# # Correlation matrices...

# response-predictor correlation...
r_p_corr = df.corr()
print('Correlation matrix of response-predictor')
print(r_p_corr)

plt.figure(figsize = (30, 25), num =2)
plt.title('Response-Predicitor pairwise correlation')
sns.heatmap(r_p_corr, annot = True, cmap="PuBuGn_r")
plt.show()

# # predictor-predictor correlation...
p_p_corr = X_data.corr()
print('Correlation matrix of predictor-predictor')
print(p_p_corr)

plt.figure(figsize = (30, 25), num = 3)
plt.title('Predictor-Predicitor pairwise correlation')
sns.heatmap(p_p_corr, annot = True, cmap="RdYlBu_r")
plt.show()

# Find ANOVA table...
X_data = sm.add_constant(X_data)
stat_model = sm.OLS(np.array(Y_data), np.array(X_data))
res = stat_model.fit()
print(res.summary())

# From correlation matrix it can be observed that sample rate 48000 HZ and Sixe on Memory are highly correlated, therefore dropping size on memory...
df = df.drop(['Size on memory  (kB)'], axis = 1)
X_data = df.iloc[:, :-1]
Y_data = df['Channels']

#%%
plt.rc('font', size=15)

# Test-train split...
X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size = 0.3, random_state=3)
X_train_scaled = MinMaxScaler().fit_transform(X_train)
X_test_scaled = MinMaxScaler().fit_transform(X_test)

# Logistic Regression...
print('\n\tLogistic Regression\n')
clf = LogisticRegression(solver= 'liblinear', random_state=3).fit(X_train, y_train)

print('Training accuracy of Logistic regression classifier: ', clf.score(X_train, y_train))
print('Testing accuracy of Logistic regression classifier: ', clf.score(X_test, y_test))

# Plot True vs Predicted response...
plt.figure(4)
plt.scatter(range(len(y_test)), y_test, color='black', marker='o', label='True')
plt.scatter(range(len(y_test)), clf.predict(X_test), color='r', marker='.', label='Predicted')
plt.legend()
plt.title('Logistic Regression')
plt.xlabel('Observations')
plt.ylabel('Response')

# Performing cross-validation...
cv_scores = cross_val_score(clf, X_train, y_train,cv=10)
print('Mean cross-validation score (10-fold) of Logistic regression: ', np.mean(cv_scores))
print('Cross-validation scores (10-fold) of Logistic regression:', cv_scores)
print('\n\n')

# KNN...
# Finding the best n_neighbors value and assign it...
print('\n\tKNN\n')
temp_r2_train = []
temp_r2_test = []
mse = []
bias = []
var = []
n_neighbors_range = range(1,20)
temp_r2 = 1
for i in n_neighbors_range:
    knnreg = KNeighborsClassifier(n_neighbors = i).fit(X_train, y_train)
    temp_r2_train.append(knnreg.score(X_train, y_train))
    temp_r2_test.append(knnreg.score(X_test, y_test))
    if abs(float(knnreg.score(X_train, y_train))-float(knnreg.score(X_test, y_test))) < temp_r2:
        temp_r2 = abs(float(knnreg.score(X_train, y_train))-float(knnreg.score(X_test, y_test)))
        best_n_neighbor = i
    else:
        pass
    m, b, v = bias_variance_decomp(knnreg, np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), loss='mse', num_rounds=200, random_seed=3)
    mse.append(m)
    bias.append(b)
    var.append(v)

plt.figure(5)
plt.plot(n_neighbors_range, temp_r2_train, color ='r', marker = 'o', label='Training set')
plt.plot(n_neighbors_range, temp_r2_test, color ='g', marker = 'o', label='Testing set')
plt.xlabel('Number of neighbors')
plt.ylabel('R-Squared value')
plt.legend()

# Bias Variance Tradeoff...
plt.figure(6)
plt.plot(n_neighbors_range, mse,  label = 'Error')
plt.plot(n_neighbors_range, bias,  label = 'Bias')
plt.plot(n_neighbors_range, var, label = 'Variance')
plt.xlabel('Number of neighbors')
plt.title('Bias Variance Tradeoff')
plt.legend()

# From the plot obtained from the code above (number of neighbors vs R2 value), the best N-neighbors value is 13...
knnreg = KNeighborsClassifier(n_neighbors = best_n_neighbor).fit(X_train, y_train)

print('Training accuracy of KNN regression: ', str(knnreg.score(X_train, y_train)))
print('Testing accuracy of KNN regression: ', str(knnreg.score(X_test, y_test)))

# Plot True vs Predicted response...
plt.figure(7)
plt.scatter(range(len(y_test)), y_test, color='black', marker='o', label='True')
plt.scatter(range(len(y_test)), knnreg.predict(X_test), color='r', marker='.', label='Predicted')
plt.legend()
plt.title('K_nearest Neighbors')
plt.xlabel('Observations')
plt.ylabel('Response')

# Performing cross-validation...
cv_scores = cross_val_score(knnreg, X_train, y_train,cv=10)
print('Mean cross-validation score (10-fold) of KNN: ', np.mean(cv_scores))
print('Cross-validation scores (10-fold) of KNN:', cv_scores)
print('\n\n')

# LDA...
print('\n\tLinear discrimant analysis\n')
LDA_model = LinearDiscriminantAnalysis().fit(X_train_scaled, y_train)
scores_train = cross_val_score(LDA_model, X_train_scaled, y_train, scoring='accuracy', n_jobs=-1)
scores_test = cross_val_score(LDA_model, X_test_scaled, y_test, scoring='accuracy', n_jobs=-1)

print('Training accuracy of Linear discrimant analysis: ', np.mean(scores_train))
print('Testing accuracy of Linear discrimant analysis: ', np.mean(scores_test))

# Plot True vs Predicted response...
plt.figure(8)
plt.scatter(range(len(y_test)), y_test, color='black', marker='o', label='True')
plt.scatter(range(len(y_test)), LDA_model.predict(X_test), color='r', marker='.', label='Predicted')
plt.legend()
plt.title('Linear discrimant analysis')
plt.xlabel('Observations')
plt.ylabel('Response')

# Performing cross-validation...
cv_scores = cross_val_score(LDA_model, X_train, y_train,cv=10)
print('Mean cross-validation score (10-fold) of Linear discrimant analysis: ', np.mean(cv_scores))
print('Cross-validation scores (10-fold) of Linear discrimant analysis:', cv_scores)
print('\n\n')

# QDA...
print('\n\tQuadratic discrimant analysis\n')
QDA_model = QuadraticDiscriminantAnalysis().fit(X_train, y_train)
scores_train = cross_val_score(QDA_model, X_train, y_train, scoring='accuracy', n_jobs=-1)
scores_test = cross_val_score(QDA_model, X_test, y_test, scoring='accuracy', n_jobs=-1)

print('Training accuracy of Quadratic discrimant analysis: ', np.mean(scores_train))
print('Testing accuracy of Quadratic discrimant analysis: ', np.mean(scores_test))

# Plot True vs Predicted response...
plt.figure(9)
plt.scatter(range(len(y_test)), y_test, color='black', marker='o', label='True')
plt.scatter(range(len(y_test)), QDA_model.predict(X_test), color='r', marker='.', label='Predicted')
plt.legend()
plt.title('Quadratic discrimant analysis')
plt.xlabel('Observations')
plt.ylabel('Response')

# Performing cross-validation...
cv_scores = cross_val_score(QDA_model, X_train, y_train,cv=10)
print('Mean cross-validation score (10-fold) of Quadratic discrimant analysis: ', np.mean(cv_scores))
print('Cross-validation scores (10-fold) of Quadratic discrimant analysis:', cv_scores)
print('\n\n')

#%%
# Decision Tree...
print('\n\tDecision Tree Classifier\n')

# Handling missing values...

# Using median values for missing values...
col_missing_values = [str(i) for i in df_original.columns[:-1]]
temp = []
for j in col_missing_values:
    for i in range(len(df_original[j])):
        try:
            if str(df_original[j][i]) != 'nan':
                temp.append(float(df_original[j][i]))
        except KeyError:
            pass

    median = np.median(temp)
    for i in range(len(df_original[j])):
        try:
            if str(df_original[j][i]) == 'nan':
                df_original[j][i] = median
        except KeyError:
            pass

# Using mean values for missing values...
for j in col_missing_values:
    mean = np.mean(df_original[j])
    for i in range(len(df_original[j])):
        try:
            if str(df_original[j][i]) == 'nan':
                df_original[j][i] = mean
        except KeyError:
            pass

X_data_dt = df_original.iloc[:, :-1]
Y_data_dt = df_original['Channels']
X_train_dt, X_test_dt, y_train_dt, y_test_dt = train_test_split(X_data_dt, Y_data_dt, test_size = 0.3, random_state=50)
X_train_scaled_dt = MinMaxScaler().fit_transform(X_train_dt)
X_test_scaled_dt = MinMaxScaler().fit_transform(X_test_dt)


# To find best mximum depth value...
max_depth_range = list(range(1, 20))
accuracy_list = []
accuracy = 100
for d in max_depth_range:
    dt_depth = DecisionTreeClassifier(max_depth = d, min_samples_split = 2)
    dt_depth_reg = dt_depth.fit(X_train_scaled_dt, y_train_dt)
    score = dt_depth_reg.score(X_test_scaled_dt, y_test_dt)
    accuracy_list.append(score)
    if score < accuracy:
        accuracy = score
        depth_desired = d
    else:
        pass

# From the above plot it can be observed that the best max_depth is 1...
plt.figure(10)
plt.plot(max_depth_range, accuracy_list, color ='r', marker = '*', label='Accuracy')
plt.xlabel('Maximum depth')
plt.ylabel('Accuracy')
plt.legend()

dt_class = DecisionTreeClassifier(max_depth = depth_desired, min_samples_split = 2)
dt_class_reg = dt_class.fit(X_train_scaled_dt, y_train_dt)
cv_scores = cross_val_score(dt_class, X_data_dt, Y_data_dt, cv=10)
print('Testing accuracy of Decision Tree classifier is: ', dt_class_reg.score(X_test_scaled_dt, y_test_dt))
print('Training accuracy of Decision Tree classifier is: ', dt_class_reg.score(X_train_scaled_dt, y_train_dt))
print('Mean cross-validation score (10-fold) of Decision Tree classifier: ', np.mean(cv_scores))
print('Cross-validation scores (10-fold) of Decision Tree classifier:', cv_scores)
test_error = 1-dt_class_reg.score(X_test_scaled_dt, y_test_dt)
train_error = 1-dt_class_reg.score(X_train_scaled_dt, y_train_dt)

# Plot True vs Predicted response...
plt.figure(11)
plt.scatter(range(len(y_test_dt)), y_test_dt, color='black', marker='o', label='True')
plt.scatter(range(len(y_test_dt)), dt_class_reg.predict(X_test_scaled_dt), color='r', marker='.', label='Predicted')
plt.legend()
plt.title('Decision Tree')
plt.xlabel('Observations')
plt.ylabel('Response')
#%%
dot_data = StringIO()
export_graphviz(dt_class_reg, out_file=dot_data, filled=True, feature_names=df_original.columns[:-1])
(graph, ) = graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())

#%%
# Random forest...
print('\n\tRandom forest classifier\n')
rnd_ft_class = RandomForestClassifier(n_estimators = 100, max_depth = depth_desired, min_samples_split = 2, random_state = 50)
rnd_ft_class.fit(X_train, y_train)


# Plot True vs Predicted response...
plt.figure(12)
plt.scatter(range(len(y_test)), y_test, color='black', marker='o', label='True')
plt.scatter(range(len(y_test)), rnd_ft_class.predict(X_test), color='r', marker='.', label='Predicted')
plt.legend()
plt.title('Random Forest')
plt.xlabel('Observations')
plt.ylabel('Response')

# Performing cross-validation...
cv_scores = cross_val_score(rnd_ft_class, X_train, y_train,cv=10)
print('Testing accuracy of Random Forest Classifier is: ', rnd_ft_class.score(X_test, y_test))
print('Training accuracy of Random Forest Classifier is: ', rnd_ft_class.score(X_train, y_train))
print('Mean cross-validation score (10-fold) of Random Forest Classifier: ', np.mean(cv_scores))
print('Cross-validation scores (10-fold) of Random Forest Classifier:', cv_scores)
print('\n\n')


#%%
# Ridge regression...
lamda = np.linspace(1e-20, 500, 2000)

# Performing Ridge classification model...
print('\n\tRidge classifier\n')
low_mse = 100
mse = []
mse2 = []
bias = []
var = []
r2_test = []
r2_train = []
for i in lamda:
    ridge_mod = RidgeClassifier(alpha = i, random_state=3)
    linridge = ridge_mod.fit(X_train_scaled, y_train)
    test_error  = mean_squared_error(linridge.predict(X_test_scaled), y_test)
    mse.append(test_error)
    r2_train.append(float(linridge.score(X_train_scaled, y_train)))
    r2_test.append(float(linridge.score(X_test_scaled, y_test)))
    if low_mse < test_error:
        low_mse = test_error
        ideal_lamda = i
    else:
        pass
    m, b, v = bias_variance_decomp(ridge_mod, np.array(X_train_scaled), np.array(y_train), np.array(X_test_scaled), np.array(y_test), loss='mse', num_rounds=200, random_seed=3)
    mse2.append(m)
    bias.append(b)
    var.append(v)

# Plotting...
plt.figure(15)
plt.scatter(lamda, np.array(mse), color = 'g')
plt.xlabel('Alpha')
plt.ylabel('Error')
plt.title('Ridge classifier')
plt.xlim(xmin=-1);

# Plotting...
plt.figure(16)
plt.plot(lamda, r2_train, color = 'g', marker = 'o', label='Training')
plt.plot(lamda, r2_test, color = 'r', marker = 'o', label='Testing')
plt.xlabel('Alpha')
plt.ylabel('R-Squared')
plt.xlim(xmin=-1);
plt.legend()

# Bias Variance Tradeoff...
plt.figure(17)
plt.plot(lamda, mse2,  label = 'Error')
plt.plot(lamda, bias,  label = 'Bias')
plt.plot(lamda, var, label = 'Variance')
plt.xlabel('Alpha', fontsize=20)
plt.title('Bias Variance Tradeoff', fontsize=20)
plt.legend()

ridge_mod = RidgeClassifierCV(alphas = lamda, scoring='r2', cv=10)
linridge = ridge_mod.fit(X_train_scaled, y_train)

print('The best alpha value is: ', linridge.alpha_)
print('Training accuracy of Ridge classifier: ', linridge.score(X_train_scaled, y_train))
print('Testing accuracy of Ridge classifier: ', linridge.score(X_test_scaled, y_test))

# Plot True vs Predicted response...
plt.figure(18)
plt.scatter(range(len(y_test)), y_test, color='black', marker='o', label='True')
plt.scatter(range(len(y_test)),ridge_mod.predict(X_test_scaled), color='r', marker='.', label='Predicted')
plt.legend()
plt.title('Ridge Regression')
plt.xlabel('Observations')
plt.ylabel('Response')

# Performing cross-validation...
cv_scores = cross_val_score(linridge, X_train, y_train,cv=5)
print('Mean cross-validation score (10-fold) of Ridge regression: ', np.mean(cv_scores))
print('Cross-validation scores (10-fold) of Ridge regression:', cv_scores)
print('\n\n')

# Performing Lasso regression...
lamda = np.linspace(1e-20, 50, 2000)


#%%
# Performing PCA analysis...
print('\n\tPCA\n')
LDA_model = LinearDiscriminantAnalysis()
X_reduced = PCA().fit_transform(X_train_scaled)
X_reduced_test = PCA().fit_transform(X_test_scaled)
X_reduced_train = PCA().fit_transform(X_train_scaled)

# Performing cross-validation...
n = len(X_reduced)
kf_20 = KFold(n_splits=10, shuffle=True, random_state=3)
low_mse = 100
mse = []
mse2 = []
bias = []
var = []
mse_score = -1*cross_val_score(LDA_model, np.ones((n,1)), y_train.ravel(), cv=kf_20, scoring='neg_mean_squared_error').mean()
mse.append(mse_score)
for i in np.arange(0, 30):
    mse_score = -1*cross_val_score(LDA_model, X_reduced[:,:i], y_train.ravel(), cv=kf_20, scoring='neg_mean_squared_error').mean()
    mse.append(mse_score)
    if mse_score <= low_mse:
        low_mse = mse_score
        no_comp_PCA = i
    else:
        pass

# Plotting...
plt.figure(21)
plt.plot(np.array(mse), '-o')
plt.xlabel('Number of principal components in regression')
plt.ylabel('Error')
plt.title('PCA-LDA')
plt.xlim(xmin=-1);

# From graph obtained, it can be concluded that the best value for M is 4...
X_reduced = PCA(n_components=no_comp_PCA).fit_transform(X_train_scaled)
X_reduced_test = PCA(n_components=no_comp_PCA).fit_transform(X_test_scaled)
X_reduced_train = PCA(n_components=no_comp_PCA).fit_transform(X_train_scaled)
regress_mod = LDA_model.fit(X_reduced_train, y_train)

print('Training accuracy of PCA: ', regress_mod.score(X_reduced_train, y_train))
print('Testing accuracy OF PCA: ', regress_mod.score(X_reduced_test, y_test))

# Plot True vs Predicted response...
plt.figure(22)
plt.scatter(range(len(y_test)), y_test, color='black', marker='o', label='True')
plt.scatter(range(len(y_test)), regress_mod.predict(X_reduced_test), color='r', marker='.', label='Predicted')
plt.legend()
plt.title('PCA-LDA')
plt.xlabel('Observations')
plt.ylabel('Response')

# Performing cross-validation...
cv_scores = cross_val_score(regress_mod, X_reduced_train, y_train,cv=20)
print('Mean cross-validation score (10-fold) of PCA regression: ', np.mean(cv_scores))
print('Cross-validation scores (10-fold) of PCA regression:', cv_scores)
print('\n\n')

#%%
# Performing PLS regression...
# Performing PLS regression...
print('\n\tPLS\n')
kf_20 = KFold(n_splits=20, shuffle=True, random_state=3)
low_mse = 100
mse = []
for i in np.arange(0, 20):
    pls = PLSRegression(n_components=i)
    mse_score = -1*cross_val_score(pls, X_train_scaled, y_train, cv=kf_20, scoring='neg_mean_squared_error').mean()
    mse.append(mse_score)
    if mse_score < low_mse:
        low_mse = mse_score
        no_comp_PLS = i
    else:
        pass

print('The best number of components of PLS is: ', no_comp_PLS)

# Plotting...
plt.figure(23)
plt.plot(np.array(mse), '-o')
plt.xlabel('Number of principal components in regression')
plt.ylabel('Error')
plt.title('PLS')
plt.xlim(xmin=-1);

# From graph obtained, it can be concluded that the best value for M is 1...
pls = PLSRegression(n_components=no_comp_PLS, scale=False)
pls_regress_mod = pls.fit(X_train_scaled, y_train)

print('Training accuracy of PLS: ', pls_regress_mod.score(X_train_scaled, y_train))
print('Testing accuracy of PLS: ', pls_regress_mod.score(X_test_scaled, y_test))

# Plot True vs Predicted response...
plt.figure(24)
plt.scatter(range(len(y_test)), y_test, color='black', marker='o', label='True')
plt.scatter(range(len(y_test)), rnd_ft_class.predict(X_test_scaled), color='r', marker='.', label='Predicted')
plt.legend()
plt.title('PLS')
plt.xlabel('Observations')
plt.ylabel('Response')

# Performing cross-validation...
cv_scores = cross_val_score(pls, X_train_scaled, y_train,cv=10)
print('Mean cross-validation score (10-fold) of PLS regression: ', np.mean(cv_scores))
print('Cross-validation scores (10-fold) of PLS regression:', cv_scores)
print('\n\n')


#%%
#SVC...

# Parameters of the model to tune
parameters = {}



# define the model/ estimator
model = SVC()



# define the grid search
SVC_clf= GridSearchCV(model, parameters, scoring='r2',cv=10,return_train_score= True, n_jobs = -1)#initial cv =5



#fit the grid search
SVC_clf.fit(X_train, y_train)

# best model
best_model = SVC_clf.best_estimator_

best_model.fit(X_train, y_train)


print('Training accuracy of SVC: ', best_model.score(X_train, y_train))
print('Testing accuracy of SVC: ', best_model.score(X_test, y_test))



# Plot True vs Predicted response...
plt.figure(26)
plt.scatter(range(len(y_test)), y_test, color='black', marker='o', label='True')
plt.scatter(range(len(y_test)), best_model.predict(X_test), color='r', marker='.', label='Predicted')
plt.legend()
plt.title('SVC')
plt.xlabel('Observations')
plt.ylabel('Response')



# Performing cross-validation...
cv_scores = cross_val_score(best_model, X_train, y_train,cv=10)
print()
print('Mean cross-validation score (10-fold) of SVC: ', np.mean(cv_scores))
print('Cross-validation scores (10-fold) of SVC:', cv_scores)
print('\n\n')
#%%
# To plot method vs testing accuracy...
method = ['Logistic', 'LDA', 'QDA', 'KNN', 'Decision Tree', 'Random Forest', 'Ridge', 'PCA_LDA', 'SVC']
method_accuracy_testing = [0.864130435, 0.97826087, 0.880780781, 0.961956522, 0.89673913, 0.940217391, 0.967391304, 0.983783784, 0.8695652173913043]
method_accuarcy_training = [0.93676815, 0.992995896, 0.702599179, 0.964871194, 1.0, 0.985948478, 0.995316159, 0.99765808, 0.9297423887587822]

plt.figure(25)
plt.rc('font', size=15)
for i in range(0, len(method)):
    plt.scatter(method_accuarcy_training[i], method_accuracy_testing[i], marker = '*', s=1000, label=method[i])
plt.xlabel('Training Accuracy', fontsize = 20)
plt.ylabel('Testing Accuracy', fontsize = 20)
plt.title('Training Accuracy vs Testing Accuracy of the Models', fontsize = 20)
plt.legend()
plt.show()
