import os
import sys
from pathlib import Path

def rename_images(folder_path):
    folder = Path(folder_path)

    if not folder.exists():
        print(f"Error: Folder '{folder_path}' does not exist.")
        return

    folder_name = folder.name
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}

    images = sorted([
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ])

    if not images:
        print(f"No images found in '{folder_path}'.")
        return

    print(f"Found {len(images)} images in '{folder_name}'. Renaming...")

    for i, image in enumerate(images, start=1):
        new_name = f"{folder_name}_{i:04d}.png"
        new_path = folder / new_name
        image.rename(new_path)
        print(f"  {image.name} -> {new_name}")

    print(f"\nDone! Renamed {len(images)} images.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # If no argument given, look for ToAdd folder in current directory
        to_add = Path(os.getcwd()) / "ToAdd"
    else:
        to_add = Path(sys.argv[1]) / "ToAdd"

    if not to_add.exists():
        print(f"Error: 'ToAdd' folder not found at '{to_add}'.")
        sys.exit(1)

    subfolders = [f for f in to_add.iterdir() if f.is_dir()]

    if not subfolders:
        print("No subfolders found inside 'ToAdd'.")
        sys.exit(1)

    print(f"Found {len(subfolders)} subfolders in 'ToAdd'.\n")
    for subfolder in sorted(subfolders):
        rename_images(subfolder)
