X_boston, y_boston = datasets.load_boston(return_X_y=True)

# X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
# X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train, y_train, test_size=0.25) # 0.25 x 0.8 = 0.2

# Normalize the data
sc = preprocessing.StandardScaler()
sc.fit(X_boston)
X_boston = sc.transform(X_boston)

X_boston = torch.Tensor(X_boston)
y_boston = torch.Tensor(y_boston)

linear_regressor_boston = LinearRegressorTorchy(X_boston.shape[1], 1)
linear_regressor_boston.fit(X_boston, y_boston, epochs=500, lr=0.01, print_losses=False)

from sklearn.metrics import r2_score

r2_error = r2_score(
    linear_regressor_boston(X_boston).detach().numpy(), y_boston.detach().numpy()
)
print(f"R^2 error: {r2_error}")
