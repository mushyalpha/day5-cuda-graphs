from __future__ import annotations
import torch

class CUDAGraphRunner:

    def __init__(self):
        self._graph: torch.cuda.CUDAGraph | None = None
        self._pool = None

    @property
    def pool(self):

        if self._graph is None:
            raise RuntimeError("Call capture() first")
        return self._graph.pool()

    def capture(
        self,
        fn,
        buf,
        batch_bucket: int,
        ctx_bucket: int,
        pool=None,
    ):

        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph, pool=pool):
            fn(buf, batch_bucket, ctx_bucket)

    def replay(self):

        if self._graph is None:
            raise RuntimeError("Call capture() before replay()")
        self._graph.replay()
