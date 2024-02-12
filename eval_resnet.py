import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from ml.utils import load_application_classification_resnet_model, load_traffic_classification_resnet_model, normalise_cm
from ml.metrics import confusion_matrix, get_classification_report
from utils import ID_TO_APP, ID_TO_TRAFFIC

# plot dpi
mpl.rcParams['figure.dpi'] = 300

# model path
application_classification_resnet_model_path = 'model/application_classification.resnet.model'
traffic_classification_resnet_model_path = 'model/traffic_classification.resnet.model'

# test data path
application_classification_test_data_path = 'train_test_data/application_classification/test.parquet'
traffic_classification_test_data_path = 'train_test_data/traffic_classification/test.parquet'

application_classification_resnet = load_application_classification_resnet_model(application_classification_resnet_model_path, gpu=True)
traffic_classification_resnet = load_traffic_classification_resnet_model(traffic_classification_resnet_model_path, gpu=True)

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


# APP CLASSIFICATION
app_resnet_cm = confusion_matrix(
    data_path=application_classification_test_data_path,
    model=application_classification_resnet,
    num_class=len(ID_TO_APP)
)
app_labels = []
for i in sorted(list(ID_TO_APP.keys())):
    app_labels.append(ID_TO_APP[i])

plot_confusion_matrix(app_resnet_cm, app_labels)

get_classification_report(app_resnet_cm, app_labels)



# TRAFFIC CLASSIFICATION
traffic_resnet_cm = confusion_matrix(
    data_path=traffic_classification_test_data_path,
    model=traffic_classification_resnet,
    num_class=len(ID_TO_TRAFFIC)
)
traffic_labels = []
for i in sorted(list(ID_TO_TRAFFIC.keys())):
    traffic_labels.append(ID_TO_TRAFFIC[i])

plot_confusion_matrix(traffic_resnet_cm, traffic_labels)

get_classification_report(traffic_resnet_cm, traffic_labels)