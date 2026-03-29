from src.data_ingestion import ingest_data
from src.train import train_model


def main():
    train_df, test_df = ingest_data()
    _, metrics = train_model(train_df, test_df)
    print("Pipeline completed successfully")
    print("Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
