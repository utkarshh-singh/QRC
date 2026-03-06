from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from utils import DataHandler
from ESN import ESNetwork, ClassicalRC
from reservoirs import CPRC, GBPermanents
from circuits import CPCircuit


def run_single_prediction_horizon(p, tau, window_size, n_samples, train_size):
    X, Y = DataHandler().load_dataset(
        'mackey_glass',
        n_samples=n_samples,
        tau=tau,
        window_size=window_size,
        prediction_horizon=p,
        plot=False
    )

    X_train_, X_test_ = X[:train_size], X[train_size:]
    y_train, y_test = Y[:train_size], Y[train_size:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_)
    X_test = scaler.transform(X_test_)

    dim = X_train.shape[1]
    cprc = CPRC(dim=dim, execution_mode='simulation', kernel=True)
    esn = ESNetwork(
        reservoir=cprc,
        dim=dim,
        regularization=1e-6,
        alpha=1,
        show_progress=False,
        approach='feedback',
        model_type='ridge',
        limit=0.4,
        cpk=True
    )

    esn.fit(X_train, y_train)
    y_pred = esn.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred)
    print(rmse)
    return p, rmse
