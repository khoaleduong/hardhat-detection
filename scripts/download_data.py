import os
from pathlib import Path
from dotenv import load_dotenv
from roboflow import Roboflow


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

load_dotenv(ROOT_DIR / ".env")
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")


def main():
    if not ROBOFLOW_API_KEY:
        raise RuntimeError("ROBOFLOW_API_KEY not found in .env file!")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Project root: {ROOT_DIR}")
    print(f"Download to : {DATA_DIR}")

    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace("robertatroc-7xoey").project("hard-hat-workers-dataset-hnvfo")
    version = project.version(1)

    dataset = version.download("yolov8", location=str(DATA_DIR), overwrite=True)

    print("\nDownload completed.")
    print(f"Dataset location: {dataset.location}")


if __name__ == "__main__":
    main()