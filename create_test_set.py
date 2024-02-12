import os
import sys
from pathlib import Path

import click
import psutil
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, monotonically_increasing_id, lit
from pyspark.sql.types import StructType, StructField, ArrayType, LongType, DoubleType


def make_test(df):
    
    # add increasing id for df
    df = df.withColumn("id", monotonically_increasing_id())
    df.withColumn("is_test", lit(True))
    test_df = df.select("feature", "label", "is_test")
    return test_df



def save_parquet(df, path):
    output_path = path.absolute().as_uri()
    (df.write.mode("overwrite").parquet(output_path))


def save_test(df, path_dir):
    path = path_dir / "test.parquet"
    save_parquet(df, path)
    
    
    
def create_test_for_task(df, label_col, data_dir_path):
    
    task_df = df.filter(col(label_col).isNotNull()).selectExpr(
        "feature", f"{label_col} as label"
    )

    test_df = make_test(task_df)

    print("saving test")
    save_test(test_df, data_dir_path)
    print("saving test done")
    
    
    
def print_df_label_distribution(spark, path):
    print(path)
    print(
        spark.read.parquet(path.absolute().as_uri()).groupby("label").count().toPandas()
    )


@click.command()
@click.option(
    "-s",
    "--source",
    help="path to the directory containing preprocessed files",
    required=True,
)
@click.option(
    "-t",
    "--target",
    help="path to the directory for persisting test set for app and traffic classification",
    required=True,
)
def main(source, target):
    source_data_dir_path = Path(source)
    target_data_dir_path = Path(target)

    # prepare dir for dataset
    application_data_dir_path = target_data_dir_path / "application_classification"
    traffic_data_dir_path = target_data_dir_path / "traffic_classification"

    # initialise local spark
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    memory_gb = psutil.virtual_memory().available // 1024 // 1024 // 1024
    spark = (
        SparkSession.builder.master("local[*]")
        .config("spark.driver.memory", f"{memory_gb}g")
        .config("spark.driver.host", "127.0.0.1")
        .getOrCreate()
    )

    # read data
    schema = StructType(
        [
            StructField("app_label", LongType(), True),
            StructField("traffic_label", LongType(), True),
            StructField("feature", ArrayType(DoubleType()), True),
        ]
    )

    df = spark.read.schema(schema).json(
        f"{source_data_dir_path.absolute().as_uri()}/*.json.gz"
    )

    # prepare data for application classification
    print("processing application classification dataset")
    create_test_for_task(
        df=df,
        label_col="app_label",
        data_dir_path=application_data_dir_path,
    )

    # prepare data for traffic classification
    print("processing traffic classification dataset")
    create_test_for_task(
        df=df,
        label_col="traffic_label",
        data_dir_path=traffic_data_dir_path,
    )

    # stats
    print_df_label_distribution(spark, application_data_dir_path / "test.parquet")
    print_df_label_distribution(spark, traffic_data_dir_path / "test.parquet")


if __name__ == "__main__":
    main()