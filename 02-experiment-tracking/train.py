# python train.py --data_original 


import os
import pickle
import click
import mlflow

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-taxi-02homework")

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)

def run_train(data_path: str):

    with mlflow.start_run():
        
        mlflow.set_tag("developer","Mariela")

        mlflow.log_param("train-data-path", "./data/green_tripdata_2023-01.parquet")
        mlflow.log_param("valid-data-path", "./data/green_tripdata_2023-02.parquet")
        
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        m_depth = 10
        rmd_state= 0
        mlflow.log_param("max_depth", m_depth)
        mlflow.log_param("random_state",  rmd_state)

        rf = RandomForestRegressor(max_depth=m_depth, random_state= rmd_state)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        print("rmse = ",rmse)
        mlflow.log_metric("rmse", rmse)

if __name__ == '__main__':

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("nyc-taxi-experiment")
    mlflow.autolog()
    run_train()
