# used as a helper script, practice caution when executing independently
from pathlib import Path
from temperature_forecaster.paths import PROJECT_ROOT

# will wipe all .py files, including ones in all subdirectories
    
from pathlib import Path

#vibecoded btw
def wipe_folder(folder_name, subfolder_name=None, file_name=None):
    root = PROJECT_ROOT / folder_name

    if not root.is_dir():
        raise FileNotFoundError(f"{folder_name} does not exist.")

    # Case 1: Delete all files under the root, preserving all folders.
    if subfolder_name is None:
        for file in root.rglob("*"):
            if file.is_file():
                file.unlink()
                print(f"Removing {file}")
        return

    # Find the target subfolder(s)
    print(root)
    print(list(root.rglob("*")))
    print(repr(subfolder_name))

    matches = list(root.rglob(subfolder_name))
    print(matches)

    for directory in matches:
        print("Found:", directory)

    for directory in root.rglob(subfolder_name):
        if not directory.is_dir():
            continue

        # Case 2: Delete all files inside the subfolder (and its descendants).
        if file_name is None:
            for file in directory.rglob("*"):
                if file.is_file():
                    file.unlink()
            return

        # Case 3: Delete one specific file from the subfolder.
        target = directory / file_name
        if target.is_file():
            target.unlink()
            return

    #raise FileNotFoundError(f"Subfolder '{subfolder_name}' not found.")
