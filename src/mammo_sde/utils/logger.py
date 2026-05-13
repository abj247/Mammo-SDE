"""Rich-based colorful step logger.

Provides StepLogger for professional console output with Step N/M progress,
inline progress bars, status spinners, time tracking, and parallel file logging.

Example:
    logger = StepLogger(total_steps=3, log_file="run.log", title="GMM Analysis")
    with logger.step("Loading embeddings"):
        embeddings = load(...)
    with logger.step("Fitting GMM with K sweep"):
        for k in logger.track(range(2, 21), description="K sweep"):
            fit(k)
    with logger.step("Saving results"):
        save(...)
    logger.summary()
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text


class StepLogger:
    """Colorful step-by-step logger with rich formatting and file mirroring.

    Each call to ``step(name)`` increments an internal counter and prints
    ``Step N/M: name`` in cyan. The step is wrapped in a context manager that
    tracks wall time and prints a green check on success or red cross on error.

    All console output is mirrored to ``log_file`` as plain text (no ANSI codes)
    so logs are readable in any viewer.
    """

    def __init__(
        self,
        total_steps: int,
        log_file: str | Path | None = None,
        title: str | None = None,
    ):
        self.total_steps = int(total_steps)
        self.current_step = 0
        self._step_times: list[tuple[str, float, str]] = []
        self._start_time = time.time()

        self.console = Console()

        self._log_path: Path | None = None
        self._log_console: Console | None = None
        if log_file is not None:
            self._log_path = Path(log_file)
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_fp = open(self._log_path, "w", encoding="utf-8")
            self._log_console = Console(file=self._log_fp, force_terminal=False, no_color=True, width=120)

        if title:
            self._emit_panel(title)

    def _emit(self, renderable: Any) -> None:
        self.console.print(renderable)
        if self._log_console is not None:
            self._log_console.print(renderable)
            self._log_fp.flush()

    def _emit_panel(self, title: str) -> None:
        panel = Panel(Text(title, justify="center", style="bold white"), border_style="bright_blue")
        self._emit(panel)

    @contextmanager
    def step(self, name: str, description: str | None = None):
        """Context manager that wraps one step with timing and status reporting."""
        self.current_step += 1
        header = Text()
        header.append(f"\n[Step {self.current_step}/{self.total_steps}] ", style="bold cyan")
        header.append(name, style="bold white")
        if description:
            header.append(f"\n    {description}", style="dim white")
        self._emit(header)

        t0 = time.time()
        status = "completed"
        try:
            yield self
        except Exception as exc:
            status = f"failed ({type(exc).__name__})"
            elapsed = time.time() - t0
            self._step_times.append((name, elapsed, status))
            err = Text()
            err.append("  ✗ ", style="bold red")
            err.append(f"{name} failed after {elapsed:.2f}s: {exc}", style="red")
            self._emit(err)
            raise
        else:
            elapsed = time.time() - t0
            self._step_times.append((name, elapsed, status))
            done = Text()
            done.append("  ✓ ", style="bold green")
            done.append(f"{name} completed in {elapsed:.2f}s", style="green")
            self._emit(done)

    def track(self, iterable: Iterable, description: str = "Working", total: int | None = None):
        """Progress bar around an inner loop. Use inside a step() context."""
        if total is None:
            try:
                total = len(iterable)  # type: ignore[arg-type]
            except TypeError:
                total = None
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
            transient=False,
        )
        with progress:
            task_id = progress.add_task(description, total=total)
            for item in iterable:
                yield item
                progress.advance(task_id)

    def info(self, message: str) -> None:
        text = Text("  ℹ ", style="bold blue") + Text(message, style="white")
        self._emit(text)

    def warn(self, message: str) -> None:
        text = Text("  ⚠ ", style="bold yellow") + Text(message, style="yellow")
        self._emit(text)

    def error(self, message: str) -> None:
        text = Text("  ✗ ", style="bold red") + Text(message, style="red")
        self._emit(text)

    def success(self, message: str) -> None:
        text = Text("  ✓ ", style="bold green") + Text(message, style="green")
        self._emit(text)

    def metric(self, name: str, value: Any, fmt: str = "") -> None:
        """Print a single named metric in a clean format."""
        value_str = format(value, fmt) if fmt else str(value)
        text = Text()
        text.append("  • ", style="bold magenta")
        text.append(f"{name}: ", style="bold white")
        text.append(value_str, style="bright_white")
        self._emit(text)

    def table(self, headers: list[str], rows: list[list[Any]], title: str | None = None) -> None:
        """Print a rich Table."""
        t = Table(title=title, show_lines=False, header_style="bold magenta", border_style="dim")
        for h in headers:
            t.add_column(h)
        for row in rows:
            t.add_row(*[str(c) for c in row])
        self._emit(t)

    def summary(self) -> None:
        """Final summary table of all steps and total time."""
        total = time.time() - self._start_time
        t = Table(title="Run Summary", show_lines=False, header_style="bold magenta")
        t.add_column("Step", style="cyan")
        t.add_column("Status", style="green")
        t.add_column("Duration", style="yellow")
        for name, elapsed, status in self._step_times:
            color = "green" if status == "completed" else "red"
            t.add_row(name, Text(status, style=color), f"{elapsed:.2f}s")
        t.add_row(Text("TOTAL", style="bold white"), Text("", style=""), Text(f"{total:.2f}s", style="bold yellow"))
        self._emit(t)
        if self._log_path is not None:
            footer = Text(f"\nLog written to: {self._log_path}", style="dim")
            self._emit(footer)

    def close(self) -> None:
        if self._log_console is not None:
            self._log_fp.close()
            self._log_console = None

    def __enter__(self) -> StepLogger:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
