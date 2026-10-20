from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.utils.config import load_config
from src.utils.validation import (
    check_input_files_exist,
    validate_output_not_empty,
    validate_required_columns,
)


USERS_REQUIRED_COLUMNS = ["user_id", "country"]
ORDERS_REQUIRED_COLUMNS = [
    "order_id",
    "user_id",
    "order_date",
    "amount",
    "status",
]


def transform(users_df: DataFrame, orders_df: DataFrame) -> DataFrame:
    """
    Transform raw users and orders data into daily revenue by country.

    Expected output columns:
    - order_date
    - country
    - total_revenue
    - completed_orders_count
    - unique_customers_count
    - avg_order_value

    Hints:
    - Use only completed orders.
    - Join orders with users by user_id.
    - Aggregate by order_date and country.
    - Use PySpark functions, not pandas.
    """

    completed_orders_df = orders_df.filter(F.col("status") == "completed")

    joined_df = completed_orders_df.join(users_df, "user_id")

    result_df = joined_df.groupBy("order_date", "country").agg(
        F.sum("amount").alias("total_revenue"),
        F.count("order_id").alias("completed_orders_count"),
        F.countDistinct("user_id").alias("unique_customers_count"),
        (F.sum("amount") / F.count("order_id")).alias("avg_order_value"),
    )

    return result_df.select(
        "order_date",
        "country",
        "total_revenue",
        "completed_orders_count",
        "unique_customers_count",
        "avg_order_value",
    )


def main() -> None:
    config = load_config()
    paths = config["paths"]

    check_input_files_exist([paths["users"], paths["orders"]])

    spark = (
        SparkSession.builder.master("local[1]")
        .appName("daily_revenue_by_country")
        .getOrCreate()
    )

    try:
        users_df = spark.read.csv(
            paths["users"],
            header=True,
            inferSchema=True,
        )
        orders_df = spark.read.csv(
            paths["orders"],
            header=True,
            inferSchema=True,
        )

        validate_required_columns(
            users_df,
            USERS_REQUIRED_COLUMNS,
            "users",
        )
        validate_required_columns(
            orders_df,
            ORDERS_REQUIRED_COLUMNS,
            "orders",
        )

        result_df = transform(users_df, orders_df)
        validate_output_not_empty(result_df)

        (
            result_df.write.mode("overwrite")
            .partitionBy("order_date")
            .parquet(paths["output"])
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
