
# CLEAR DATASET PIPELINE
1. python preprocessing.py -s clear_pcaps -t clear_processed_data
2. python create_train_test_set.py -s clear_processed_data -t train_test_data
4. python train_cnn.py -p train_test_data/application_classification/train.parquet -r train_test_data/traffic_classification/train.parquet -a model/application_classification.cnn.model -t model/traffic_classification.cnn.model -v both
5. python eval_cnn.py --ct app --gpu True
6. python eval_cnn.py --ct traffic --gpu True
---------------------------------------------------------
# OBFUSCATED DATASET PIPELINE
1. python preprocessing.py -s obfuscated_pcaps -t obfuscated_processed_data
2. python create_test_set.py -s obfuscated_processed_data -t only_test_data
3. python eval_cnn.py --actdp only_test_data/app_classification/test.parquet --ct app --gpu True
4. python eval_cnn.py --tctdp only_test_data/traffic_classification/test.parquet --ct traffic --gpu True