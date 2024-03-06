from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load data sử dụng bộ dữ liêu iris của sklearn
iris = datasets.load_iris()
X = iris.data
y = iris.target

# chia dataset thành 2 bộ training set và test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3)  # 70% training and 30% test

# Tạo adaboost  object
abc = AdaBoostClassifier(n_estimators=50,
                         learning_rate=1)
# Huấn luyện Adaboost Classifer
model = abc.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = model.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
