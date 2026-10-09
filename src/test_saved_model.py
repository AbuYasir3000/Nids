
from pathlib import Path
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = PROJECT_ROOT / "models" / "random_forest.joblib"


def main():
    if not MODEL_FILE.exists():
        print("ERROR: Saved model not found.")
        print(f"Expected location: {MODEL_FILE}")
        return

    print("Loading saved Random Forest model...")

    saved_data = joblib.load(MODEL_FILE)

    model = saved_data["model"]
    feature_columns = saved_data["feature_columns"]
    medians = saved_data["medians"]

    print("\nModel loaded successfully!")
    print(f"Model type: {type(model).__name__}")
    print(f"Number of features: {len(feature_columns)}")
    print(f"Number of classes: {len(model.classes_)}")

    print("\nClasses:")
    for label in model.classes_:
        print(f"- {label}")

    print(f"\nSaved median values: {len(medians)}")
    print("\nThe model is ready to be reused without retraining.")


if __name__ == "__main__":
    main()
