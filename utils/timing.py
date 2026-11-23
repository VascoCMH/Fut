"""Medição simples do tempo de cada etapa do programa."""

import time
from collections import defaultdict
from contextlib import contextmanager


class Timings:
    def __init__(self):
        self.seconds = defaultdict(float)
        self.counts = defaultdict(int)

    def start(self, name):
        self._started = getattr(self, "_started", {})
        self._started[name] = time.perf_counter()

    def stop(self, name):
        self.seconds[name] += time.perf_counter() - self._started.pop(name)
        self.counts[name] += 1

    @contextmanager
    def measure(self, name):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] += time.perf_counter() - start
            self.counts[name] += 1

    def report(self, video_seconds=None):
        total = sum(self.seconds.values())
        lines = ["Tempo por etapa:"]
        for name, seconds in sorted(self.seconds.items(), key=lambda kv: -kv[1]):
            share = 100 * seconds / total if total else 0
            calls = self.counts[name]
            extra = f" ({calls} chamadas)" if calls > 1 else ""
            lines.append(f"  {name:<28} {seconds:8.1f} s  {share:5.1f} %{extra}")
        line = f"  {'TOTAL':<28} {total:8.1f} s"
        if video_seconds:
            line += f"  ({total / video_seconds:.1f} s por segundo de vídeo)"
        lines.append(line)
        return "\n".join(lines)
