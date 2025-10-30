"""Build a standalone desktop executable for the SPT training GUI."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _require_pyinstaller() -> None:
    try:
        import PyInstaller  # noqa: F401
    except ImportError as exc:  # pragma: no cover - runtime dependency check
        raise SystemExit(
            "PyInstaller ist nicht installiert. Führe 'pip install pyinstaller' aus und starte den Befehl erneut."
        ) from exc


def _format_data_arg(source: Path, target: str = ".") -> str:
    return f"{source}{os.pathsep}{target}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Erzeugt eine ausführbare Desktop-App für den Trainer.")
    parser.add_argument("--name", default="SPTTrainerApp", help="Name der erzeugten Anwendung")
    parser.add_argument(
        "--onefile",
        action="store_true",
        help="Erzeugt ein einzelnes ausführbares File (langsamerer Start, einfacher zum Verteilen)",
    )
    parser.add_argument(
        "--windowed",
        action="store_true",
        help="Unterdrückt die Konsolenausgabe bei GUI-Start (empfohlen unter Windows)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Leert vor dem Build die PyInstaller build/dist Verzeichnisse",
    )
    args = parser.parse_args(argv)

    _require_pyinstaller()

    project_root = Path(__file__).resolve().parent
    script = project_root / "spt_training_app.py"
    if not script.exists():
        raise SystemExit(f"GUI-Script wurde nicht gefunden: {script}")

    if args.clean:
        for folder in (project_root / "build", project_root / "dist"):
            if folder.exists():
                shutil.rmtree(folder)

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--name",
        args.name,
    ]
    cmd.append("--onefile" if args.onefile else "--onedir")
    if args.windowed:
        cmd.append("--windowed")

    dependencies = [
        project_root / "spt_training_app.py",
        project_root / "train_spt_classifier.py",
        project_root / "spt_feature_extractor.py",
        project_root / "spt_trajectory_generator.py",
        project_root / "config_SPT.py",
    ]

    for dep in dependencies:
        if dep.exists():
            cmd.extend(["--add-data", _format_data_arg(dep)])

    cmd.append(str(script))

    print("Starte PyInstaller Build:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print("Build abgeschlossen. Ergebnis liegt im 'dist' Verzeichnis.")
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
