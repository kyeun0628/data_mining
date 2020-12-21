import pandas as pd
import numpy as np
        
class Analysis:
    def OLS(self, x,y):
        xx = x.T @ x
        xy = x.T @ y
        inv_xx = pd.DataFrame(np.linalg.pinv(xx.values), xx.columns, xx.index)
        beta_hat = inv_xx @ xy
    
        return beta_hat, x @ beta_hat
    
class Regression(Analysis):
    
    def SSE(self, y_actual, y_pred):
        error = y_actual - y_pred
        return (error.T).dot(error)
    
    def SST(self, y_actual, y_pred):
        n = len(y_actual)
        ones = np.ones(n)
        y_mean = (y_actual.T).dot(ones) / n
        y_centered = y_actual - y_mean
        return y_centered.dot(y_centered)
        
    def SSR(self, y_actual, y_pred):
        sse = self.SSE(y_actual, y_pred)
        sst = self.SST(y_actual, y_pred)
        return 1 - (sse / sst)
    
    def MSE(self, y_actual, y_pred):
        sse = self.SSE(y_actual, y_pred)
        return sse / (len(y_actual) - self._p)
    
    def RMSE(self,y_actual, y_pred):
        sse = self.SSE(y_actual, y_pred)
        return (sse/ len(y_actual))**0.5
        
    def R_square(self, y_actual, y_pred):
        return np.corrcoef(y_actual, y_pred)[0][1]**2
    
    def MAE(self,y_actual, y_pred):
        abs_error = abs(y_actual - y_pred)
        return sum(abs_error)/len(y_actual)
    
    def MAPE(self,y_actual, y_pred):
        abs_error = abs(y_actual - y_pred)
        return sum(abs_error / y_actual) / len(y_actual) * 100
