import click

from ml.utils import (
    train_application_classification_resnet_model,
    train_traffic_classification_resnet_model,
)


@click.command()
@click.option(
    "-d",
    "--data_path",
    help="training data dir path containing parquet files",
    required=True,
)
@click.option("-m", "--model_path", help="output model path", required=True)
@click.option(
    "-t",
    "--task",
    help='classification task. Option: "app" or "traffic"',
    required=True,
)
def main(data_path, model_path, task):
    if task == "app":
        print("Training for app classification started...")
        train_application_classification_resnet_model(data_path, model_path)
        print("Training for app classification successfully ended!")
    elif task == "traffic":
        print("Training for traffic classification started...")
        train_traffic_classification_resnet_model(data_path, model_path)
        print("Training for traffic classification successfully ended!")
    elif task == "both":
        print("Training for app classification started...")
        train_application_classification_resnet_model(data_path, model_path)
        print("Training for app classification successfully ended!")
        print()
        print()
        print("Training for traffic classification started...")
        train_traffic_classification_resnet_model(data_path, model_path)
        print("Training for traffic classification successfully ended!")
    else:
        exit("Not Support")



if __name__ == "__main__":
    main()
