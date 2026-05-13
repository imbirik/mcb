#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


DEFAULT_FOLDER_URL = (
    "https://drive.google.com/drive/folders/"
    "1zjHxcyoPY7FL7_SAajToGEaNvVAglGps?usp=sharing"
)
FLOW_CHECKPOINTS = [
    "lm1b_flm.ckpt",
    "owt_flm.ckpt",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download released FLM checkpoints from the public Drive folder."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/flm"),
        help="Directory where checkpoints should be stored.",
    )
    parser.add_argument(
        "--folder-url",
        default=DEFAULT_FOLDER_URL,
        help="Public Google Drive folder URL.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the expected checkpoint files already exist.",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        choices=FLOW_CHECKPOINTS,
        help=(
            "Checkpoint filename to download. Can be repeated. Defaults to the "
            "two released FLM checkpoints only."
        ),
    )
    return parser.parse_args()


def _write_manifest(output_dir, manifest):
    manifest_path = output_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote manifest: {manifest_path}")


def _list_drive_folder(folder_url):
    try:
        import gdown
    except ImportError as exc:
        raise SystemExit(
            "gdown is required for checkpoint download. "
            "Install it with `pip install gdown` or `pip install -r requirements.txt`."
        ) from exc

    try:
        files = gdown.download_folder(
            url=folder_url,
            output="__gdown_listing_only__",
            quiet=True,
            use_cookies=False,
            remaining_ok=True,
            skip_download=True,
        )
    except TypeError as exc:
        raise SystemExit(
            "The installed gdown version does not support folder listing via "
            "`skip_download=True`. Install `gdown==5.2.0` or newer."
        ) from exc

    if files is None:
        raise RuntimeError(f"Failed to list Google Drive folder: {folder_url}")
    return files


def download_selected_checkpoints(output_dir, folder_url, expected, force=False):
    expected = list(dict.fromkeys(expected))
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = {name: output_dir / name for name in expected}
    if not force and all(path.exists() for path in existing.values()):
        manifest = {name: str(path.resolve()) for name, path in existing.items()}
        _write_manifest(output_dir, manifest)
        for name, path in manifest.items():
            print(f"Checkpoint already present: {name} -> {path}")
        return manifest

    try:
        import gdown
    except ImportError as exc:
        raise SystemExit(
            "gdown is required for checkpoint download. "
            "Install it with `pip install gdown` or `pip install -r requirements.txt`."
        ) from exc

    listed_files = _list_drive_folder(folder_url)
    by_name = {}
    for item in listed_files:
        name = Path(item.path).name
        if name in expected and name not in by_name:
            by_name[name] = item

    missing = [name for name in expected if name not in by_name]
    if missing:
        available = sorted({Path(item.path).name for item in listed_files})
        raise FileNotFoundError(
            "Could not find requested checkpoint(s) in Drive folder: "
            + ", ".join(missing)
            + "\nAvailable files include: "
            + ", ".join(available[:50])
        )

    manifest = {}
    for name in expected:
        dst = output_dir / name
        if dst.exists() and not force:
            print(f"Checkpoint already present: {dst.resolve()}")
            manifest[name] = str(dst.resolve())
            continue
        if dst.exists() and force:
            dst.unlink()

        file_id = by_name[name].id
        print(f"Downloading {name} from file id {file_id}")
        downloaded = gdown.download(
            url=f"https://drive.google.com/uc?id={file_id}",
            output=str(dst),
            quiet=False,
            use_cookies=False,
        )
        if downloaded is None or not dst.exists():
            raise RuntimeError(f"Download failed for {name}")
        manifest[name] = str(dst.resolve())
        print(f"Checkpoint ready: {dst.resolve()}")

    _write_manifest(output_dir, manifest)
    return manifest


def main():
    args = parse_args()
    expected = args.checkpoint or list(FLOW_CHECKPOINTS)
    download_selected_checkpoints(
        output_dir=args.output_dir,
        folder_url=args.folder_url,
        expected=expected,
        force=args.force,
    )


if __name__ == "__main__":
    main()
