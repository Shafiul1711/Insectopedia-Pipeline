#!/usr/bin/env python3
import time
import threading
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

ROOT = Path("test_suite")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
TEMP_EXTS = {".crdownload", ".part", ".tmp"}

STARTING_INDEX = {
    "alfalfa_weevil": 56,
}

rename_lock = threading.Lock()
recently_processed = set()


def get_next_index(class_dir: Path) -> int:
    """Figure out the next index based on existing files in the folder."""
    class_name = class_dir.name
    existing = [
        f for f in class_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    ]

    max_index = 0
    for f in existing:
        stem = f.stem
        if stem.startswith(class_name + "_"):
            try:
                idx = int(stem[len(class_name) + 1:])
                max_index = max(max_index, idx)
            except ValueError:
                pass

    if max_index == 0:
        return STARTING_INDEX.get(class_name, 1)
    return max_index + 1


def wait_until_complete(path: Path, timeout: float = 30.0, stable_checks: int = 3, interval: float = 0.5) -> bool:
    """
    Wait until the file exists and its size stays unchanged for several checks.
    Returns True if stable, False if timeout or disappearance.
    """
    start = time.time()
    last_size = -1
    stable_count = 0

    while time.time() - start < timeout:
        if not path.exists():
            time.sleep(interval)
            continue

        try:
            size = path.stat().st_size
        except OSError:
            time.sleep(interval)
            continue

        if size > 0 and size == last_size:
            stable_count += 1
            if stable_count >= stable_checks:
                return True
        else:
            stable_count = 0
            last_size = size

        time.sleep(interval)

    return False


def rename_file(path: Path):
    """Rename a newly added image to follow class naming convention."""
    path = Path(path)

    if path.suffix.lower() not in IMAGE_EXTS:
        return

    if path in recently_processed:
        return

    class_dir = path.parent
    class_name = class_dir.name

    if not wait_until_complete(path):
        print(f"  [SKIP] File not stable in time: {path.name}")
        return

    with rename_lock:
        if not path.exists():
            return

        if path in recently_processed:
            return

        # Skip already-renamed files
        if path.stem.startswith(class_name + "_"):
            return

        next_idx = get_next_index(class_dir)
        ext = path.suffix.lower()
        new_name = f"{class_name}_{next_idx:04d}{ext}"
        new_path = class_dir / new_name

        while new_path.exists():
            next_idx += 1
            new_name = f"{class_name}_{next_idx:04d}{ext}"
            new_path = class_dir / new_name

        try:
            path.rename(new_path)
            recently_processed.add(new_path)
            print(f"  [RENAMED] {path.name} -> {new_name}")
        except Exception as e:
            print(f"  [ERROR] Could not rename {path.name}: {e}")


def count_images():
    counts = {}
    for class_dir in ROOT.iterdir():
        if class_dir.is_dir():
            counts[class_dir.name] = sum(
                1 for f in class_dir.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTS
            )
    return counts


def print_counts():
    counts = count_images()
    print("\n=== Current Counts ===")
    for cls in sorted(counts):
        print(f"  {cls:<25} {counts[cls]}")
    print("----------------------")


def process_async(path: Path):
    t = threading.Thread(target=_process_and_print, args=(path,), daemon=True)
    t.start()


def _process_and_print(path: Path):
    rename_file(path)
    print_counts()


class Handler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return

        path = Path(event.src_path)
        suffix = path.suffix.lower()

        if suffix in TEMP_EXTS:
            return

        if suffix in IMAGE_EXTS:
            print(f"\n[NEW] Detected: {path.name} in '{path.parent.name}'")
            process_async(path)

    def on_moved(self, event):
        if event.is_directory:
            return

        dest = Path(event.dest_path)
        suffix = dest.suffix.lower()

        if suffix in TEMP_EXTS:
            return

        if suffix in IMAGE_EXTS:
            print(f"\n[MOVED] Detected: {dest.name} in '{dest.parent.name}'")
            process_async(dest)

    def on_deleted(self, event):
        if not event.is_directory:
            path = Path(event.src_path)
            if path.suffix.lower() in IMAGE_EXTS:
                print(f"\n[DELETED] {path.name} from '{path.parent.name}'")
                print_counts()


if __name__ == "__main__":
    if not ROOT.exists():
        print("Create a 'ToAdd' folder first.")
        raise SystemExit(1)

    print("Watching folder:", ROOT.resolve())
    print_counts()

    event_handler = Handler()
    observer = Observer()
    observer.schedule(event_handler, str(ROOT), recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        observer.join()