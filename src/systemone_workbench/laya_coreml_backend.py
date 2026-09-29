"""Core ML model adapter for the existing System One-compatible API."""

from laya_coreml import common as coreml_common

from .laya_mlx_backend import LayaMLXBackend


class LayaCoreMLBackend(LayaMLXBackend):
    backend_name = "laya-coreml"
    common = coreml_common

    def __init__(self, checkpoint: str = "aac6fef/laya-multilingual-coreml",
                 agent=None) -> None:
        if agent is None:
            import laya_coreml
            agent = laya_coreml.load(checkpoint)
        self.agent = agent
        self.checkpoint = checkpoint
        self.model_name = checkpoint.rsplit("/", 1)[-1]
