from .base import OpSpec, Tolerance
from .rmsnorm import RMSNORM
from .softmax import SOFTMAX

OPS: dict[str, OpSpec] = {op.name: op for op in (SOFTMAX, RMSNORM)}


def get_op(name: str) -> OpSpec:
    try:
        return OPS[name]
    except KeyError:
        raise KeyError(f"Unknown op {name!r}. Available: {sorted(OPS)}") from None


__all__ = ["OPS", "OpSpec", "Tolerance", "get_op"]
