import importlib
import sys
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

EVAL_DIR = Path(__file__).resolve().parents[1] / "eval" / "ja"
sys.path.insert(0, str(EVAL_DIR))
run_dialog_wordings = importlib.import_module("run_dialog_wordings")


class DialogWordingPolarityTest(unittest.TestCase):
    def test_continuation_question_is_inverted_to_finished_label(self):
        build = getattr(run_dialog_wordings, "noul_for_finished", None)
        if not callable(build):
            self.fail("negative-polarity noul needs a finished-label extractor")
        builder = cast(Callable[[str], Callable[[], tuple[dict[str, Any], Callable[[dict[str, float]], float]]]], build)
        question, extract = builder("Has the speaker finished?")()
        self.assertEqual(question, {"type": "noul", "instructions": "Has the speaker finished?"})
        self.assertAlmostEqual(extract({"noul": 0.2}), 0.8)
        self.assertAlmostEqual(extract({"noul": 0.9}), 0.1)


if __name__ == "__main__":
    unittest.main()
