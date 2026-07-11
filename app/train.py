"""CLI entrypoint to train the model and write the artifact to disk.

Usage:
    python -m app.train
"""

from __future__ import annotations

from app.model import DEFAULT_MODEL_PATH, save_model, train_model


def main() -> None:
    print("Training digit classifier on sklearn load_digits ...")
    model = train_model()
    path = save_model(model)
    print(f"  model type : {model.model_type}")
    print(f"  features   : {model.n_features}")
    print(f"  labels     : {model.labels}")
    print(f"  test acc   : {model.accuracy:.4f}")
    print(f"  saved to   : {path}")


if __name__ == "__main__":
    main()
