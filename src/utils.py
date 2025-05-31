from pathlib import Path


def get_project_dir() -> Path:

    return Path(__file__).resolve().parents[1]

def get_data_dir() -> Path:
    return get_project_dir() / "data"

def get_raw_dir() -> Path:
    return get_data_dir() / "00--raw"

def get_clean_dir() -> Path:
    return get_data_dir() / "01--clean"

def get_metadata_dir() -> Path:
    return get_data_dir() / "10--metadata"


def get_metadata_dir() -> Path:
    return get_data_dir() / "10--metadata"

if __name__ == "__main__":
    project_dir = get_project_dir()
    data_dir = get_data_dir()

    print(f"project_dir: {project_dir}")
    print(f"data_dir:    {data_dir}")
    