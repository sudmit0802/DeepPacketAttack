import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import click
from IPython.display import display

from ml.utils import load_application_classification_cnn_model, load_traffic_classification_cnn_model, normalise_cm
from ml.metrics import confusion_matrix, get_classification_report
from utils import ID_TO_APP, ID_TO_TRAFFIC


def plot_confusion_matrix(cm, labels):
    normalised_cm, normalised_labels = normalise_cm(cm, labels)
    _, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(
        data=normalised_cm, cmap='YlGnBu',
        xticklabels=normalised_labels, yticklabels=normalised_labels,
        annot=True, ax=ax, fmt='.2f'
    )
    ax.set_xlabel('Predict labels')
    ax.set_ylabel('True labels')
    plt.show()
        
@click.command()
@click.option(
    "--accmp",
    default = 'model/application_classification.cnn.model',
    help="Application classification CNN model path",
)
@click.option(
    "--tccmp",
    default = 'model/traffic_classification.cnn.model',
    help="Traffic classification CNN model path",
)
@click.option(
    "--actdp",
    default = 'train_test_data/application_classification/test.parquet',
    help="Application classification test data path",
)
@click.option(
    "--tctdp",
    default = 'train_test_data/traffic_classification/test.parquet',
    help="Traffic classification test data path",
)
@click.option(
    "--ct",
    default="app",
    help="Type of classification: app/traffic",
)
@click.option(
    "--dpi",
    default=80,
    help="Matplotlib figure DPI: (30-300)",
    type=int
)
@click.option(
    "--gpu",
    default=False,
    help="Flag for using GPU: True/False",
    type=bool
)
def main(accmp, tccmp, actdp, tctdp, ct, dpi, gpu):
    # plot dpi
    mpl.rcParams['figure.dpi'] = dpi

    # model path
    application_classification_cnn_model_path = accmp
    traffic_classification_cnn_model_path = tccmp

    # test data path
    application_classification_test_data_path = actdp
    traffic_classification_test_data_path = tctdp

    # APP CLASSIFICATION
    if ct=='app':
        print("App classification started...")
        
        application_classification_cnn = load_application_classification_cnn_model(application_classification_cnn_model_path,
                                                                             gpu=gpu)
        print("Building confusion matrix...")
        app_cnn_cm = confusion_matrix(
        data_path=application_classification_test_data_path,
        model=application_classification_cnn,
        num_class=len(ID_TO_APP))
        print("Confusion matrix successfully built.")

        app_labels = []
        for i in sorted(list(ID_TO_APP.keys())):
            app_labels.append(ID_TO_APP[i])

        print("Getting classification report...")
        cr, avg_prec, avg_rec = get_classification_report(app_cnn_cm, app_labels)
        display(cr)
        print("Average precision:", avg_prec)
        print("Average recall:", avg_rec)
        print("Classification report done.")
        print("Plotting confusion matrix...")
        plot_confusion_matrix(app_cnn_cm, app_labels)
        print("Plotting confusion matrix done.")
        print("App classification done.")
        
    # TRAFFIC CLASSIFICATION
    elif ct=='traffic':
        print("Traffic classification started...")
        traffic_classification_cnn = load_traffic_classification_cnn_model(traffic_classification_cnn_model_path, gpu=gpu)
        
        print("Building confusion matrix...")
        traffic_cnn_cm = confusion_matrix(
        data_path=traffic_classification_test_data_path,
        model=traffic_classification_cnn,
        num_class=len(ID_TO_TRAFFIC))
        print("Confusion matrix successfully built.")

        traffic_labels = []
        for i in sorted(list(ID_TO_TRAFFIC.keys())):
            traffic_labels.append(ID_TO_TRAFFIC[i])

        
        print("Getting classification report...")
        cr, avg_prec, avg_rec = get_classification_report(traffic_cnn_cm, traffic_labels)
        display(cr)
        print("Average precision:", avg_prec)
        print("Average recall:", avg_rec)
        print("Classification report done.")
        print("Plotting confusion matrix...")
        plot_confusion_matrix(traffic_cnn_cm, traffic_labels)
        print("Plotting confusion matrix done.")
        print("Traffic classification done.")
    else:
        print("Incorrect classification type!")
        exit(1)


if __name__ == "__main__":
    main()