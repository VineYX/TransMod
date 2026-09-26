"""
 EV_DT : rich  + TensorBoard
"""
from __future__ import annotations
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Optional, Any

try:
    from rich.console import Console
    from rich.table   import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.panel   import Panel
    from rich.text    import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False


class TransModLogger:
    """
     rich  + TensorBoard 
    """

    def __init__(
        self,
        log_dir:  str | Path,
        exp_name: str = 'transmod',
        use_tb:   bool = True,
    ):
        self.log_dir  = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.exp_name = exp_name
        self.console  = Console() if HAS_RICH else None
        self._step    = 0
        self._epoch   = 0

        # TensorBoard
        self.writer: Optional[Any] = None
        if use_tb and HAS_TB:
            tb_dir = self.log_dir / 'tensorboard' / exp_name
            tb_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(str(tb_dir))

        fh = logging.FileHandler(self.log_dir / f'{exp_name}.log', encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s  %(message)s'))
        self._file_logger = logging.getLogger(exp_name)
        self._file_logger.setLevel(logging.INFO)
        self._file_logger.addHandler(fh)
        self._file_logger.addHandler(logging.StreamHandler(sys.stdout))


    def info(self, msg: str):
        if self.console:
            self.console.print(f'[cyan]{msg}[/cyan]')
        else:
            self._file_logger.info(msg)

    def success(self, msg: str):
        if self.console:
            self.console.print(f'[bold green]{msg}[/bold green]')
        self._file_logger.info(msg)

    def warning(self, msg: str):
        if self.console:
            self.console.print(f'[bold yellow][WARN]  {msg}[/bold yellow]')
        self._file_logger.warning(msg)


    def log_scalars(self, tag: str, values: Dict[str, float], step: int):
        if self.writer:
            for k, v in values.items():
                self.writer.add_scalar(f'{tag}/{k}', v, step)
        kv = '  '.join(f'{k}={v:.4f}' for k, v in values.items())
        self._file_logger.info(f'[step {step:06d}] {tag}: {kv}')


    def print_epoch(self, epoch: int, metrics: Dict[str, float], stage: str = 'train'):
        color = {'train': 'blue', 'val': 'green', 'test': 'magenta'}.get(stage, 'white')
        if self.console and HAS_RICH:
            tbl = Table(title=f'Epoch {epoch} [{stage}]', style=color)
            tbl.add_column('Metric')
            tbl.add_column('Value', justify='right')
            for k, v in metrics.items():
                tbl.add_row(k, f'{v:.4f}')
            self.console.print(tbl)
        else:
            kv = '  '.join(f'{k}={v:.4f}' for k, v in metrics.items())
            self._file_logger.info(f'Epoch {epoch} [{stage}]: {kv}')

    def print_banner(self, msg: str):
        if self.console and HAS_RICH:
            self.console.print(Panel(Text(msg, justify='center'), style='bold cyan'))
        else:
            self._file_logger.info('=' * 60)
            self._file_logger.info(f'  {msg}')
            self._file_logger.info('=' * 60)

    def close(self):
        if self.writer:
            self.writer.close()


class MovingAvg:
    """"""

    def __init__(self, beta: float = 0.95):
        self.beta  = beta
        self.value = None

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = x
        else:
            self.value = self.beta * self.value + (1 - self.beta) * x
        return self.value


class MetricTracker:
    """"""

    def __init__(self):
        self._sums:   Dict[str, float] = {}
        self._counts: Dict[str, int]   = {}

    def update(self, d: Dict[str, float]):
        for k, v in d.items():
            self._sums[k]   = self._sums.get(k, 0.0) + float(v)
            self._counts[k] = self._counts.get(k, 0) + 1

    def mean(self) -> Dict[str, float]:
        return {k: self._sums[k] / max(self._counts[k], 1) for k in self._sums}

    def reset(self):
        self._sums.clear()
        self._counts.clear()
