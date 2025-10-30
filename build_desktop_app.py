"""Build a standalone desktop executable for the SPT training GUI."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

DEFAULT_QT_EXCLUDES: tuple[str, ...] = ("PyQt6", "PyQt5", "PySide6", "PySide2")


def _require_pyinstaller() -> None:
    try:
        import PyInstaller  # noqa: F401
    except ImportError as exc:  # pragma: no cover - runtime dependency check
        raise SystemExit(
            "PyInstaller ist nicht installiert. Führe 'pip install pyinstaller' aus und starte den Befehl erneut."
        ) from exc


def _format_data_arg(source: Path, target: str = ".") -> str:
    return f"{source}{os.pathsep}{target}"


def _extend_with_excludes(cmd: List[str], modules: Iterable[str]) -> None:
    for module in modules:
        cmd.extend(["--exclude-module", module])


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
    parser.add_argument(
        "--keep-qt",
        action="store_true",
        help=(
            "Standardmäßig werden PyQt/PySide-Module ausgeschlossen, um Konflikte zwischen mehreren Qt-"
            "Bindings zu vermeiden. Mit diesem Flag kannst du diese Ausschlüsse deaktivieren."
        ),
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="MODUL",
        help="Zusätzliche Module, die via --exclude-module an PyInstaller übergeben werden sollen",
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

    cmd: List[str] = [
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

    excludes: List[str] = list(args.exclude)
    if not args.keep_qt:
        excludes.extend(DEFAULT_QT_EXCLUDES)
    _extend_with_excludes(cmd, excludes)

    cmd.append(str(script))

    print("Starte PyInstaller Build:")
    print(" ".join(cmd))
    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.stdout:
        print(result.stdout)

    if result.returncode != 0:
        if "multiple Qt bindings packages" in (result.stdout or "") and not args.keep_qt:
            print(
                "Hinweis: PyInstaller hat mehrere Qt-Bindings gefunden. Standardmäßig wurden PyQt/PySide"
                "-Pakete ausgeschlossen. Sollte dein Build dennoch Qt benötigen, wiederhole den Befehl"
                " mit --keep-qt und sorge dafür, dass nur eine Qt-Variante installiert ist."
            )
        raise SystemExit(result.returncode)

    print("Build abgeschlossen. Ergebnis liegt im 'dist' Verzeichnis.")
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
