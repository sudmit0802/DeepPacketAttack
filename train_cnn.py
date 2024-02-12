import click

from ml.utils import (
    train_application_classification_cnn_model,
    train_traffic_classification_cnn_model,
)


@click.command()
@click.option(
    "-d",
    "--data_path",
    help="training data dir path containing parquet files",
    default = "",
    required=False,
)
@click.option(
    "-p",
    "--app_data_path",
    help="training app data dir path containing parquet files",
    default = "",
    required=False,
)
@click.option(
    "-r",
    "--traffic_data_path",
    help="training traffic data dir path containing parquet files",
    default = "",
    required=False,
)
@click.option("-m", "--model_path", help="output model path", default="", required=False)
@click.option("-a", "--app_model_path", help="output app model path", default = "", required=False)
@click.option("-t", "--traffic_model_path", help="output traffic model path", default = "", required=False)
@click.option(
    "-v",
    "--task",
    help='classification variant type. Option: "app" or "traffic" or "both"',
    required=True,
)
def main(data_path, app_data_path, traffic_data_path, model_path, app_model_path, traffic_model_path, task):
            
    if task == "app":
        if not (model_path and data_path):
            exit("Not Support")
        print("Training for app classification started...")
        train_application_classification_cnn_model(data_path, model_path)
        print("Training for app classification successfully ended!")
    elif task == "traffic":
        if not (model_path and data_path):
            exit("Not Support")
        print("Training for traffic classification started...")
        train_traffic_classification_cnn_model(data_path, model_path)
        print("Training for traffic classification successfully ended!")
    elif task == "both":
        if model_path or not (app_model_path and traffic_model_path):
            exit("Not Support")
        if data_path or not (app_data_path and traffic_data_path):
            exit("Not Support")
        print("Training for app classification started...")
        train_application_classification_cnn_model(app_data_path, app_model_path)
        print("Training for app classification successfully ended!")
        print()
        print()
        print("Training for traffic classification started...")
        train_traffic_classification_cnn_model(traffic_data_path, traffic_model_path)
        print("Training for traffic classification successfully ended!")
    else:
        exit("Not Support")


if __name__ == "__main__":
    main()
