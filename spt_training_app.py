"""GUI-Anwendung zum Starten und Überwachen des SPT-Klassifikator-Trainings."""
from __future__ import annotations

import io
import os
import platform
import queue
import subprocess
import threading
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from train_spt_classifier import SPTClassifierTrainer, TrainingConfig


class QueueStream(io.TextIOBase):
    """Leitet stdout/stderr-Zeilen in eine Thread-Queue um."""

    def __init__(self, sink):
        super().__init__()
        self._sink = sink

    def write(self, s):
        if s:
            self._sink(s)
        return len(s)

    def flush(self):
        pass


class TrainingApp:
    """Tkinter-App zum Starten des Trainings mit visueller Fortschrittsanzeige."""

    def __init__(self, master: tk.Tk) -> None:
        self.master = master
        self.master.title("SPT Diffusion Classifier - Training")
        self.master.geometry("1100x720")

        self.queue: "queue.Queue[tuple]" = queue.Queue()
        self.training_thread: threading.Thread | None = None
        self.trainer: SPTClassifierTrainer | None = None
        self.training_config: TrainingConfig | None = None
        self.last_artifact_dir: Path | None = None

        # GUI-States
        self.samples_var = tk.IntVar(value=500)
        self.mode_var = tk.StringVar(value="both")
        self.ratio_var = tk.DoubleVar(value=0.5)
        self.poly_var = tk.DoubleVar(value=0.5)
        self.epochs_var = tk.IntVar(value=60)
        self.batch_var = tk.IntVar(value=128)
        self.max_len_var = tk.IntVar(value=600)
        default_output = Path("./spt_trained_model_app").resolve()
        self.output_dir_var = tk.StringVar(value=str(default_output))

        self.stage_var = tk.StringVar(value="Bereit.")
        self.stage_progress_var = tk.DoubleVar(value=0.0)
        self.epoch_progress_var = tk.DoubleVar(value=0.0)
        self.artifact_path_var = tk.StringVar(value="Noch kein Lauf ausgeführt.")

        self.train_acc = []
        self.val_acc = []

        self._build_layout()

        # Poll Queue
        self.master.after(200, self._process_queue)

    # ------------------------------------------------------------------
    # GUI Aufbau
    # ------------------------------------------------------------------
    def _build_layout(self) -> None:
        config_frame = ttk.LabelFrame(self.master, text="Training konfigurieren")
        config_frame.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)

        for i in range(4):
            config_frame.grid_columnconfigure(i, weight=1)

        ttk.Label(config_frame, text="Samples je Klasse:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Spinbox(config_frame, from_=10, to=20000, textvariable=self.samples_var, increment=10, width=10).grid(
            row=0, column=1, sticky="we", padx=5, pady=5
        )

        ttk.Label(config_frame, text="Dimensionalität:").grid(row=0, column=2, sticky="w", padx=5, pady=5)
        mode_box = ttk.Combobox(config_frame, textvariable=self.mode_var, values=("2D", "3D", "both"), state="readonly")
        mode_box.grid(row=0, column=3, sticky="we", padx=5, pady=5)
        mode_box.bind("<<ComboboxSelected>>", lambda _: self._update_ratio_state())

        ttk.Label(config_frame, text="Anteil 3D (bei 'both'):").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.ratio_scale = ttk.Scale(
            config_frame,
            from_=0.0,
            to=1.0,
            variable=self.ratio_var,
            command=lambda _: self._update_ratio_label(),
        )
        self.ratio_scale.grid(row=1, column=1, sticky="we", padx=5, pady=5)
        self.ratio_label = ttk.Label(config_frame, text="50 %")
        self.ratio_label.grid(row=1, column=2, sticky="w", padx=5, pady=5)

        ttk.Label(config_frame, text="Polymerisationsgrad:").grid(row=1, column=3, sticky="w", padx=5, pady=5)
        ttk.Scale(config_frame, from_=0.0, to=1.0, variable=self.poly_var).grid(row=1, column=4, sticky="we", padx=5, pady=5)

        ttk.Label(config_frame, text="Max. Trajektorienlänge:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        ttk.Spinbox(config_frame, from_=50, to=5000, textvariable=self.max_len_var, increment=50, width=10).grid(
            row=2, column=1, sticky="we", padx=5, pady=5
        )

        ttk.Label(config_frame, text="Epochen:").grid(row=2, column=2, sticky="w", padx=5, pady=5)
        ttk.Spinbox(config_frame, from_=5, to=500, textvariable=self.epochs_var, increment=5, width=10).grid(
            row=2, column=3, sticky="we", padx=5, pady=5
        )

        ttk.Label(config_frame, text="Batch Size:").grid(row=2, column=4, sticky="w", padx=5, pady=5)
        ttk.Spinbox(config_frame, from_=16, to=1024, textvariable=self.batch_var, increment=16, width=10).grid(
            row=2, column=5, sticky="we", padx=5, pady=5
        )

        ttk.Label(config_frame, text="Ausgabeordner:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        output_entry = ttk.Entry(config_frame, textvariable=self.output_dir_var)
        output_entry.grid(row=3, column=1, columnspan=3, sticky="we", padx=5, pady=5)
        ttk.Button(config_frame, text="Ordner wählen", command=self._choose_output_dir).grid(
            row=3, column=4, sticky="we", padx=5, pady=5
        )

        self.start_button = ttk.Button(config_frame, text="Training starten", command=self.start_training)
        self.start_button.grid(row=3, column=5, sticky="we", padx=5, pady=5)

        # Fortschritt
        progress_frame = ttk.LabelFrame(self.master, text="Fortschritt")
        progress_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=5)

        ttk.Label(progress_frame, textvariable=self.stage_var).grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.stage_bar = ttk.Progressbar(progress_frame, maximum=5, variable=self.stage_progress_var)
        self.stage_bar.grid(row=1, column=0, sticky="we", padx=5, pady=5)

        ttk.Label(progress_frame, text="Epoche").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.epoch_bar = ttk.Progressbar(progress_frame, maximum=1, variable=self.epoch_progress_var)
        self.epoch_bar.grid(row=3, column=0, sticky="we", padx=5, pady=2)

        artifact_row = ttk.Frame(progress_frame)
        artifact_row.grid(row=4, column=0, sticky="we", padx=5, pady=(10, 0))
        artifact_row.grid_columnconfigure(0, weight=1)
        ttk.Label(artifact_row, textvariable=self.artifact_path_var, wraplength=500, justify="left").grid(
            row=0, column=0, sticky="w"
        )
        self.open_dir_button = ttk.Button(
            artifact_row,
            text="Ordner öffnen",
            command=self._open_artifact_dir,
            state="disabled",
            width=18,
        )
        self.open_dir_button.grid(row=0, column=1, sticky="e", padx=(10, 0))

        # Logs + Plot Bereich
        main_frame = ttk.Frame(self.master)
        main_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)
        self.master.grid_rowconfigure(2, weight=1)
        self.master.grid_columnconfigure(0, weight=1)
        self.master.grid_columnconfigure(1, weight=1)
        main_frame.grid_columnconfigure(0, weight=2)
        main_frame.grid_columnconfigure(1, weight=3)
        main_frame.grid_rowconfigure(0, weight=1)

        log_frame = ttk.LabelFrame(main_frame, text="Log")
        log_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        log_frame.grid_rowconfigure(0, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)
        self.log_widget = tk.Text(log_frame, wrap="word", state="disabled")
        self.log_widget.grid(row=0, column=0, sticky="nsew")
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_widget.yview)
        log_scroll.grid(row=0, column=1, sticky="ns")
        self.log_widget["yscrollcommand"] = log_scroll.set

        plot_frame = ttk.LabelFrame(main_frame, text="Trainingsgenauigkeit")
        plot_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        plot_frame.grid_rowconfigure(0, weight=1)
        plot_frame.grid_columnconfigure(0, weight=1)

        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Accuracy pro Epoche")
        self.ax.set_xlabel("Epoche")
        self.ax.set_ylabel("Accuracy")
        self.ax.set_ylim(0, 1)
        self.ax.grid(True, alpha=0.3)
        (self.train_line,) = self.ax.plot([], [], label="Train", color="#1f77b4", linewidth=2)
        (self.val_line,) = self.ax.plot([], [], label="Val", color="#ff7f0e", linewidth=2)
        self.ax.legend(loc="lower right")

        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self._update_ratio_state()
        self._update_ratio_label()

    # ------------------------------------------------------------------
    # Hilfsfunktionen
    # ------------------------------------------------------------------
    def _choose_output_dir(self) -> None:
        directory = filedialog.askdirectory(initialdir=self.output_dir_var.get())
        if directory:
            self.output_dir_var.set(directory)

    def _append_log(self, text: str) -> None:
        if not text:
            return
        self.log_widget.configure(state="normal")
        self.log_widget.insert("end", text)
        self.log_widget.see("end")
        self.log_widget.configure(state="disabled")

    def _queue_log(self, text: str) -> None:
        self.queue.put(("log", text))

    def _update_ratio_label(self) -> None:
        value = self.ratio_var.get()
        self.ratio_label.configure(text=f"{int(round(value * 100))} %")

    def _update_ratio_state(self) -> None:
        mode = self.mode_var.get().lower()
        if mode == "both":
            self.ratio_scale.state(["!disabled"])
            self.ratio_label.configure(foreground="black")
        else:
            self.ratio_scale.state(["disabled"])
            self.ratio_label.configure(foreground="grey")
            if mode == "2d":
                self.ratio_var.set(0.0)
            else:
                self.ratio_var.set(1.0)
            self._update_ratio_label()

    def _reset_progress(self) -> None:
        self.stage_var.set("Initialisiere...")
        self.stage_progress_var.set(0.0)
        self.epoch_progress_var.set(0.0)
        self.epoch_bar.configure(maximum=1)
        self.last_artifact_dir = None
        self.artifact_path_var.set("Noch kein Lauf ausgeführt.")
        self.open_dir_button.configure(state="disabled")
        self.train_acc.clear()
        self.val_acc.clear()
        self._update_plot()

    def _update_plot(self) -> None:
        epochs = range(1, len(self.train_acc) + 1)
        self.train_line.set_data(list(epochs), self.train_acc)
        self.val_line.set_data(list(epochs), self.val_acc)
        if self.train_acc or self.val_acc:
            max_epochs = max(len(self.train_acc), len(self.val_acc))
            self.ax.set_xlim(1, max_epochs if max_epochs > 1 else 1)
            max_acc = max(self.train_acc + self.val_acc)
            min_acc = min(self.train_acc + self.val_acc)
            span = max(0.1, max_acc - min_acc)
            lower = max(0.0, min_acc - span * 0.2)
            upper = min(1.0, max_acc + span * 0.2)
            self.ax.set_ylim(lower, upper)
        else:
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(0, 1)
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Training Pipeline
    # ------------------------------------------------------------------
    def start_training(self) -> None:
        if self.training_thread and self.training_thread.is_alive():
            messagebox.showinfo("Training läuft", "Das Training läuft bereits.")
            return

        try:
            samples = max(1, int(self.samples_var.get()))
            epochs = max(1, int(self.epochs_var.get()))
            batch = max(1, int(self.batch_var.get()))
            max_len = max(10, int(self.max_len_var.get()))
        except tk.TclError:
            messagebox.showerror("Eingabefehler", "Bitte gültige numerische Werte eingeben.")
            return

        mode = self.mode_var.get().lower()
        ratio = float(np.clip(self.ratio_var.get(), 0.0, 1.0))
        polymer = float(np.clip(self.poly_var.get(), 0.0, 1.0))
        output_dir = Path(self.output_dir_var.get()).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        self.start_button.configure(state="disabled")
        self._reset_progress()
        self._clear_log()
        self._append_log(f"[{datetime.now().strftime('%H:%M:%S')}] Starte Training...\n")

        self.training_config = TrainingConfig(
            n_samples_per_class=samples,
            mode=mode,
            ratio_3d=ratio,
            polymerization_degree=polymer,
            epochs=epochs,
            batch_size=batch,
            cache_path=(output_dir / "cached_dataset.npz"),
        )

        def worker():
            try:
                self.trainer = SPTClassifierTrainer(max_length=max_len, output_dir=str(output_dir))
                stages = [
                    ("Generiere Trainingsdaten", self._run_data_generation, {}),
                    ("Baue Modell", self._run_build_model, {}),
                    ("Trainiere Modell", self._run_training, {}),
                    ("Evaluiere", self._run_evaluation, {}),
                    ("Speichere Ergebnisse", self._run_save, {})
                ]

                total_stages = len(stages)
                for idx, (label, func, kwargs) in enumerate(stages, start=1):
                    self.queue.put(("stage", idx, total_stages, label))
                    result = func(**kwargs)
                    self.queue.put(("stage_complete", idx, total_stages))

                artifacts_dir = result if result else output_dir
                self.queue.put(("done", True, str(artifacts_dir)))
            except Exception as exc:  # pylint: disable=broad-except
                self.queue.put(("error", str(exc)))
            finally:
                self.queue.put(("finished", None, None))

        self.training_thread = threading.Thread(target=worker, daemon=True)
        self.training_thread.start()

    def _run_data_generation(self) -> None:
        if self.training_config is None:
            raise RuntimeError("Training configuration missing")
        self.queue.put(("log", f"\n[{datetime.now().strftime('%H:%M:%S')}] Datengenerierung...\n"))
        stream = QueueStream(self._queue_log)
        with redirect_stdout(stream), redirect_stderr(stream):
            self.trainer.generate_training_data(config=self.training_config, verbose=True, reuse_cache=False)

    def _run_build_model(self) -> None:
        self.queue.put(("log", f"\n[{datetime.now().strftime('%H:%M:%S')}] Modellkonstruktion...\n"))
        stream = QueueStream(self._queue_log)
        with redirect_stdout(stream), redirect_stderr(stream):
            self.trainer.build_model(verbose=True)

    def _run_training(self) -> None:
        if self.training_config is None:
            raise RuntimeError("Training configuration missing")
        self.queue.put(("log", f"\n[{datetime.now().strftime('%H:%M:%S')}] Training...\n"))

        def epoch_update(epoch, logs, total_epochs):
            self.queue.put(("epoch", epoch, logs, total_epochs))

        stream = QueueStream(self._queue_log)
        with redirect_stdout(stream), redirect_stderr(stream):
            self.trainer.train(
                epochs=self.training_config.epochs,
                batch_size=self.training_config.batch_size,
                verbose=0,
                epoch_callback=epoch_update,
            )

    def _run_evaluation(self) -> None:
        self.queue.put(("log", f"\n[{datetime.now().strftime('%H:%M:%S')}] Evaluation...\n"))
        stream = QueueStream(self._queue_log)
        with redirect_stdout(stream), redirect_stderr(stream):
            accuracy, _, _ = self.trainer.evaluate(verbose=True)
        self.queue.put(("log", f"\nFinale Test-Accuracy: {accuracy:.4f}\n"))

    def _run_save(self):
        self.queue.put(("log", f"\n[{datetime.now().strftime('%H:%M:%S')}] Speichere Modell und Artefakte...\n"))
        stream = QueueStream(self._queue_log)
        with redirect_stdout(stream), redirect_stderr(stream):
            path = self.trainer.save_model(verbose=True)
        return path

    # ------------------------------------------------------------------
    # Queue Handling
    # ------------------------------------------------------------------
    def _process_queue(self) -> None:
        try:
            while True:
                item = self.queue.get_nowait()
                kind = item[0]
                if kind == "log":
                    self._append_log(item[1])
                elif kind == "stage":
                    _, index, total, label = item
                    self.stage_var.set(label)
                    self.stage_bar.configure(maximum=total)
                    self.stage_progress_var.set(index - 1)
                elif kind == "stage_complete":
                    _, index, total = item
                    self.stage_bar.configure(maximum=total)
                    self.stage_progress_var.set(index)
                elif kind == "epoch":
                    _, epoch, logs, total_epochs = item
                    self.epoch_bar.configure(maximum=total_epochs)
                    self.epoch_progress_var.set(epoch + 1)
                    train_acc = logs.get("accuracy")
                    val_acc = logs.get("val_accuracy")
                    if train_acc is not None:
                        if len(self.train_acc) <= epoch:
                            self.train_acc.append(float(train_acc))
                        else:
                            self.train_acc[epoch] = float(train_acc)
                    if val_acc is not None:
                        if len(self.val_acc) <= epoch:
                            self.val_acc.append(float(val_acc))
                        else:
                            self.val_acc[epoch] = float(val_acc)
                    self.stage_var.set(f"Training – Epoche {epoch + 1}/{total_epochs}")
                    self._update_plot()
                elif kind == "done":
                    _, success, out_dir = item
                    if success:
                        self.last_artifact_dir = Path(out_dir)
                        self.artifact_path_var.set(f"Artefakte: {out_dir}")
                        self.open_dir_button.configure(state="normal")
                        messagebox.showinfo(
                            "Training abgeschlossen",
                            f"Das Training wurde erfolgreich abgeschlossen.\nArtefakte befinden sich in:\n{out_dir}",
                        )
                elif kind == "error":
                    _, msg = item
                    messagebox.showerror("Fehler", f"Während des Trainings ist ein Fehler aufgetreten:\n{msg}")
                elif kind == "finished":
                    self.start_button.configure(state="normal")
                    self.stage_var.set("Bereit.")
                    self.stage_progress_var.set(self.stage_bar["maximum"])
                    self.epoch_progress_var.set(0)
                self.queue.task_done()
        except queue.Empty:
            pass
        finally:
            self.master.after(200, self._process_queue)

    def _clear_log(self) -> None:
        self.log_widget.configure(state="normal")
        self.log_widget.delete("1.0", tk.END)
        self.log_widget.configure(state="disabled")

    def _open_artifact_dir(self) -> None:
        if not self.last_artifact_dir:
            messagebox.showinfo("Keine Artefakte", "Es wurde noch kein Lauf abgeschlossen.")
            return

        path = self.last_artifact_dir
        if platform.system() == "Windows":
            os.startfile(path)  # type: ignore[attr-defined]
        elif platform.system() == "Darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)


def main() -> None:
    root = tk.Tk()
    app = TrainingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
