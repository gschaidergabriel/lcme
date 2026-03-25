"""
L.C.M.E. — Local Cognitive Memory Engine

A local, deterministic context, memory, and retrieval system featuring:
- Tri-hybrid storage: SQLite + Vector Store + Knowledge Graph
- 10 neural micro-networks (~303K params total) for living memory
- Bio-inspired retrieval with Hopfield associative memory
- Epistemological humility: claims not facts, confidence decay
- Controlled forgetting with permanent memory graduation

Quick start::

    from lcme import LCME, LCMEConfig

    config = LCMEConfig(data_dir="./my_memory")
    memory = LCME(config)

    memory.ingest("Alice works at a startup in Berlin.", origin="user")
    results = memory.retrieve("What do we know about Alice?")
    context = memory.get_context_string("Tell me about Alice")

Convenience functions (use a global singleton)::

    from lcme import remember, recall, get_context, forget, protect

    remember("The user prefers dark mode.")
    results = recall("user preferences")
"""

__version__ = "1.0.0"
__codename__ = "LCME"

from .core import LCME, LCMEConfig, get_lcme, reset_lcme
from .core import remember, recall, get_context, forget, protect

__all__ = [
    "LCME",
    "LCMEConfig",
    "get_lcme",
    "reset_lcme",
    "remember",
    "recall",
    "get_context",
    "forget",
    "protect",
]
