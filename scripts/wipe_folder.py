# used as a helper script, practice caution when executing independently
from pathlib import Path
from temperature_forecaster.paths import PROJECT_ROOT

# will wipe all .py files, including ones in all subdirectories
def wipe_folder(folder_name):

    directory = Path(PROJECT_ROOT / folder_name) #if isinstance(folder_name, str) else Path(f"{folder_name}")

    if not directory.exists():
        print("Folder not found")
        return False

    for file in directory.rglob("*"):
        if file.is_file():
            file.unlink()
            print(f"Removed: {file}")
    print("Successfully wiped past data")
    return True
    
