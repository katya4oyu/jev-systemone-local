import pytest

from jev_systemone_local.laya_coreml_backend import LayaCoreMLBackend
from jev_systemone_local.laya_mlx_backend import InputTooLong


class Tokens:
    mask_token = "[MASK]"
    mask_token_id = 3
    cls_token_id = 1
    sep_token_id = 2

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [4] * len(text.split())}


class FakeCoreMLAgent:
    def __init__(self):
        self.tok = Tokens()
        self.cfg = {"max_len": 1024, "head_max_len": 192}
        self.shape = {"max_length": 96}

    def _to_internal(self, question):
        from laya_coreml.prompt import PromptMixin
        return PromptMixin._to_internal(question)

    def predict(self, state, questions):
        return {"answers": {"yes": {"noul": 0.7}},
                "usage": {"input_tokens": 15, "output_tokens": 0}}


def test_coreml_adapter_returns_model_identity_and_rejects_ane_overflow():
    agent = FakeCoreMLAgent()
    backend = LayaCoreMLBackend("aac6fef/laya-multilingual-coreml-ane", agent=agent)
    questions = {"yes": {"type": "noul", "instructions": "Is it urgent?"}}
    answer = backend.evaluate("short state", questions)
    assert answer["model"] == "laya-multilingual-coreml-ane"
    assert answer["answers"]["yes"] == {"type": "noul", "noul": 0.7}
    with pytest.raises(InputTooLong, match="only .* fit"):
        backend.evaluate("word " * 100, questions)
    assert backend.agent is agent
    assert agent.cfg["max_len"] == 1024
    assert agent.shape["max_length"] == 96
