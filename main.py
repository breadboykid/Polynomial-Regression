import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


def TrueModel(x):
    return 2*x**2 + 4*x + 5

def GenerateSamples(n):
    X = 6 * np.random.rand(n) - 3
    noise = 5 * np.random.rand(n)
    y = TrueModel(X) + noise
    X = X.reshape(-1, 1)

    return X, y

def PrintSampleInfo(X, y):
    print(f'Feature Matrix: {X.shape}')
    print(f'Target Vector: {y.shape}')

    print(f'Samples: {X.shape[0]}')
    print(f'Features: {X.shape[1]}')

def PlotGraph(X_train, X_test, y_train, y_test, X_curve=None, y_curve=None):
    plt.plot(X_train, y_train, 'o', label='Training set')
    plt.plot(X_test, y_test, 'o', label='Test set')
    if X_curve is not None and y_curve is not None:
        plt.plot(X_curve, y_curve, 'r', linewidth=3)

    plt.legend()
    plt.title('Train test split')
    plt.xlabel('Feature')
    plt.ylabel('Target value')

    plt.show()

def GetBestModel(X_train, y_train, degrees=range(1, 20)):
    parameters = {"poly_features__degree": degrees}

    pipeline = Pipeline((
        ("poly_features", PolynomialFeatures(degree=2)),
        ("lin_reg", LinearRegression(fit_intercept=False)),))

    grid_search = GridSearchCV(pipeline, parameters, cv=5)

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    
    return best_model


if __name__ == '__main__':
    X, y = GenerateSamples(100)

    bins = np.round(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=bins)

    PlotGraph(X_train, X_test, y_train, y_test)
    
    best_model = GetBestModel(X_train, y_train)

    r2 = best_model.score(X_test, y_test)
    print('R2 score on test set is: ', round(r2, 2))

    X_curve = np.linspace(-3, 3, 100).reshape(-1, 1)
    y_curve = best_model.predict(X_curve)

    PlotGraph(X_train, X_test, y_train, y_test, X_curve, y_curve)