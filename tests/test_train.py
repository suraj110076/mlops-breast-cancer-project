from src.data_ingestion import ingest_data
from src.train import train_model


def test_training_pipeline_runs():
    train_df, test_df = ingest_data()
    model, metrics = train_model(train_df, test_df)

    assert model is not None
    assert "accuracy" in metrics
    assert metrics["accuracy"] > 0.80
