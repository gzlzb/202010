import statsmodels.formula.api as smf
from sklearn import linear_model
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import scipy.stats as stats
import warnings
from sklearn.svm import SVR
import statsmodels.api as sm


def q_squared(X, y, clf, kf):
    #n = len(data)
    #y = data.LogBIO
    #X = data.drop(['LogBIO'], axis=1)
    #lr = linear_model.LinearRegression()
    y_pred = cross_val_predict(clf, X, y, cv = kf, n_jobs=12)
    """lr = linear_model.LinearRegression(fit_intercept=False)
    predicted0 = cross_val_predict(lr, X, y, cv = n)"""
    #cv = pd.DataFrame({'y':y, 'y1':predicted})
    """cv = pd.DataFrame({'y':y, 'y1':predicted, 'y0':predicted0})
    cv = pd.DataFrame({'y':y, 'y0':predicted0})"""
    #r2 = smf.ols('y~y1', cv).fit().rsquared_adj
    """r02 = smf.ols('y~y0',cv).fit().rsquared_adj"""
    """import math
    rm2 = r2 * math.sqrt(abs(1-abs(r2-r02)))"""
    r, p = stats.pearsonr(y, y_pred)
    #print(r*r)
    return r * r

def r0_squared(data, response):
    y = data.LogBIO
    X = data.drop([response], axis=1)
    """lr1 = linear_model.LinearRegression()
    lr1.fit(X, y)
    predicted = lr1.predict(X)"""
    lr0 = linear_model.LinearRegression(fit_intercept=False)
    lr0.fit(X, y)
    predicted0 = lr0.predict(X)
    cv = pd.DataFrame({'y':y, 'y0':predicted0})
    """cv = pd.DataFrame({'y':y, 'y1':predicted, 'y0':predicted0})
    r2 = smf.ols('y~y1', cv).fit().rsquared"""
    r02 = smf.ols('y~y0',cv).fit().rsquared_adj
                
    """import math
    rm2 = r2 * math.sqrt(abs(1-abs(r2-r02)))"""
                        
    return r02

def r_squared(data, response):
    y = data.LogBIO
    X = data.drop([response], axis=1)
    #lr1 = linear_model.LinearRegression()
    #lr1.fit(X, y)
    #predicted = lr1.predict(X)
    """lr0 = linear_model.LinearRegression(fit_intercept=False)
    lr0.fit(X, y)
    predicted0 = lr0.predict(X)
    cv = pd.DataFrame({'y':y, 'y0':predicted0})
    cv = pd.DataFrame({'y':y, 'y1':predicted, 'y0':predicted0})"""
    #cv = pd.DataFrame({'y':y, 'y1':predicted})
    r2 = smf.OLS(y,X).fit().rsquared
    """r2 = smf.ols('y~y1', cv).fit().rsquared
    radj_2 = smf.ols('y~y0',cv).fit().rsquared_adj"""
                
    """import math
    rm2 = r2 * math.sqrt(abs(1-abs(r2-r02)))"""
                        
    return r2

def ols(data, response):
    y = data.LogBIO
    X = data.drop([response], axis=1)
    #lr1 = linear_model.LinearRegression()
    #lr1.fit(X, y)
    #predicted = lr1.predict(X)
    """lr0 = linear_model.LinearRegression(fit_intercept=False)
    lr0.fit(X, y)
    predicted0 = lr0.predict(X)
    cv = pd.DataFrame({'y':y, 'y0':predicted0})
    cv = pd.DataFrame({'y':y, 'y1':predicted, 'y0':predicted0})"""
    #cv = pd.DataFrame({'y':y, 'y1':predicted})
    ols = smf.OLS(y,X).fit()
    """r2 = smf.ols('y~y1', cv).fit().rsquared
    radj_2 = smf.ols('y~y0',cv).fit().rsquared_adj"""
                
    """import math
    rm2 = r2 * math.sqrt(abs(1-abs(r2-r02)))"""
                        
    return ols

def MCCV(data, clf):
    warnings.filterwarnings("ignore")
    n = len(data)
    N = int(n**0.5)
    #N = int(n/kf)
    y = data.LogBIO
    X = data.drop(['LogBIO'], axis=1)
    q2_MCCV = 0
    for i in range(N):
        X1, X2, y1, y2 = train_test_split(X, y, test_size=0.25)
        clf.fit(X1, y1)
        pred_test = clf.predict(X2)
        r, p = stats.pearsonr(y2, pred_test)
        #print(q2_LMO)
        q2_MCCV = q2_MCCV + r
    return q2_MCCV / N


def pearson(y_true, y_pred):
    err = y_true - y_pred
    pred_avg = np.mean(y_pred)
    true_avg = np.mean(y_true)
    err_avg = pred_avg - true_avg
    SE = err * err
    PRESS = np.sum(SE)
    n = len(y_true)
    true_mean = [true_avg for i in range(n)]
    err = y_true - true_mean
    SE = err * err
    TSS = np.sum(SE)
    r2 = 1 - PRESS/TSS
    return r2


def q2ven(cor, clf):

    #clf = LinearRegression()
    #clf = SVR(gamma='scale')
    
    val = pd.DataFrame(columns=['ture', 'pred'])
    import warnings
    warnings.filterwarnings("ignore")

    fold = 10
    for group in range(fold):
        va = cor[cor.index%fold==group]
        tr = cor[cor.index%fold!=group]
        y_true = tr.LogBIO
        X = tr.drop(["LogBIO"], axis=1)
        clf.fit(X, y_true)
        y_true = va.LogBIO
        X = va.drop(["LogBIO"], axis=1)
        y_pred = clf.predict(X)
        cv = pd.DataFrame({'true':y_true, 'pred':y_pred})
        val = val.append(cv)

    y_true = val.true
    y_pred = val.pred
    
    import scipy.stats as stats
    corr, p = stats.pearsonr(y_true, y_pred)
    #PRESS = numpy.sum(y_true-y_pred)**2
    #TSS = numpy.sum((y_true-numpy.mean(y_true))**2)

    return pearson(y_true, y_pred)



def Spearman_Correlation(data, response):
    y = data.LogBIO
    X = data.drop([response], axis=1)
    lr1 = linear_model.LinearRegression()
    lr1.fit(X, y)
    predicted = lr1.predict(X)
    import scipy.stats as stats
    corr = stats.spearmanr(y, predicted)
    return corr.correlation


def get_AD(train, test, response):

    X_train = train.drop([response], axis=1)
    y_test = test[response]
    X_test = test.drop([response], axis=1)
    
    mod = sm.OLS(y_test, X_test).fit()
    outliers = mod.get_influence()
    leverage = outliers.hat_matrix_diag
    
    n, p = X_train.shape
    AD = np.sum(np.where((np.abs(leverage)<3*p/n),1,0))/len(leverage)
    return AD

    
def forward_selected(data, pre, response, clf):
    n = len(data)
    remaining = set(data.columns)
    remaining.remove(response)
    """k = 0
    for i in remaining:
        k = k + 1"""
    #import math
    #kinit = math.log10(kinit)
    selected = ['LogBIO']
    p = 0
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates  = []
        for candidate in remaining:
            feature_cols = selected + [candidate]
            p = len(feature_cols)
            cv = pd.DataFrame(data[feature_cols])
            pre_feature = pd.DataFrame(pre[feature_cols])

            #r02 = r0_squared(cv, 'LogBIO')
            #r2 = r_squared(cv, 'LogBIO')
            #fit = ols(cv, 'LogBIO')
            #radj2 = radj2 * n
            #r2 = fit.rsquared_adj
            q2 = q2ven(cv, clf)
            AD = get_AD(cv, pre_feature, response)
            score = q2 * AD
            #print(radj2,aicc)
            """score = r2_avg * math.sqrt(1-d_avg)
            score = r2_min * math.sqrt(1-d_max)"""
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
            #delta = best_new_score - current_score
            print(current_score)
            #print(score)
            cv.to_csv("correct_2.csv", index=False)
            pre_feature.to_csv("predict_2.csv", index=False)

def test(training, test, response):
    y = training.LogBIO
    X = training.drop([response], axis=1)
    lr1 = linear_model.LinearRegression()
    lr1.fit(X, y)
    lr0 = linear_model.LinearRegression(fit_intercept=False)
    lr0.fit(X, y)
    
    y1 = test.LogBIO
    X1 = test.drop([response], axis=1)
    predicted = lr1.predict(X1)
    predicted0 = lr0.predict(X1)
    data = pd.DataFrame({'y':y1, 'y1':predicted, 'y0':predicted0})
    data.to_csv('Test.csv')
    r2 = smf.ols('y~y1', data).fit().rsquared
    r02 = smf.ols('y~y0',data).fit().rsquared
    import math
    rm2 = r2 * math.sqrt(abs(1-abs(r2-r02)))
    return rm2


file1 = "Correct_1.csv"
data1 = pd.read_csv(file1)
file2 = "Predict_1.csv"
data2 = pd.read_csv(file2)

"""#s = [2,3,4,5,6,7,8,9]
#s = [10,20,30,40,50,60,70,80,90]
s = [11,12,13,14,15,16,17,18,19]
#s = [2]

result = pd.DataFrame(columns=['s', 'rm2'])

for i in s:
    forward_selected(data1, data2, 'LogBIO', i)
    tra_file = "Correct_2.csv"
    tes_file = "Predict_2.csv"
    tra_da = pd.read_csv(tra_file)
    tes_da = pd.read_csv(tes_file)
    rm2 = test(tra_da, tes_da, 'LogBIO')
    print(round(rm2,4))
    rst = pd.DataFrame({'s':[i], 'rm2':[rm2]})
    result=result.append(rst)

result.to_csv("Result.csv")"""

clf = linear_model.LinearRegression()
#clf = SVR(gamma='scale')
model = forward_selected(data1, data2, 'LogBIO', clf)
#model = forward_selected(data1, data2, 'LogBIO', RandomForestRegressor())