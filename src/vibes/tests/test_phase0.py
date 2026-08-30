"""Phase 0 regressions: seed initialisation, dead-gradient detection, LLM accounting."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import agentic_recsys
from agentic_recsys.llm import (
    AuditedLLMClient, OpenAICompatibleClient, ScriptedLLMClient, normalize_usage,
)


SEED_DIR = Path(agentic_recsys.__file__).parent / "seed"


def load_seed_module(name):
    """Import a seed file the way the sandbox does: by path, with the seed dir importable."""
    sys.path.insert(0, str(SEED_DIR))
    try:
        spec = importlib.util.spec_from_file_location(f"seed_{name}", SEED_DIR / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SEED_DIR))


class SeedInitialisationTests(unittest.TestCase):
    def test_parameters_are_not_zero_initialised(self):
        model = load_seed_module("model").Model(64)
        self.assertTrue(np.any(model.weights != 0.0))
        self.assertTrue(np.all(np.isfinite(model.weights)))
        self.assertLess(float(np.abs(model.weights).max()), 1.0)

    def test_initialisation_is_seeded_and_reproducible(self):
        module = load_seed_module("model")
        self.assertTrue(np.array_equal(
            module.Model(32, seed=7).weights, module.Model(32, seed=7).weights
        ))
        self.assertFalse(np.array_equal(
            module.Model(32, seed=7).weights, module.Model(32, seed=8).weights
        ))

    def test_a_multiplicative_term_would_receive_gradient(self):
        """Two zero blocks in a product stay zero forever; two random blocks do not."""
        module = load_seed_module("model")
        left, right = module.Model(16, seed=1).weights, module.Model(16, seed=2).weights
        self.assertTrue(np.any(left * right != 0.0))

    def test_parameter_blocks_expose_every_trainable_array(self):
        model = load_seed_module("model").Model(16)
        blocks = model.parameter_blocks()
        self.assertIn("weights", blocks)
        self.assertNotIn("first_moment", blocks)
        self.assertNotIn("second_moment", blocks)

    def test_training_moves_the_seed_parameters(self):
        train = load_seed_module("train")
        model = load_seed_module("model").Model(8, learning_rate=0.05)
        before = train.parameter_snapshot(model)
        features = np.array([[0, 4], [1, 5], [2, 6]], dtype=np.int32)
        labels = np.array([1.0, 0.0, 1.0], dtype=np.float32)
        for _ in range(5):
            model.step(features, labels)
        train.assert_parameters_moved(model, before)


class DeadGradientTests(unittest.TestCase):
    class FrozenModel:
        def __init__(self):
            self.live = np.zeros(3, dtype=np.float32)
            self.inert = np.zeros(3, dtype=np.float32)

        def parameter_blocks(self):
            return {"live": self.live, "inert": self.inert}

    def test_an_inert_block_fails_loudly(self):
        train = load_seed_module("train")
        model = self.FrozenModel()
        before = train.parameter_snapshot(model)
        model.live += 0.1
        with self.assertRaises(RuntimeError) as caught:
            train.assert_parameters_moved(model, before)
        message = str(caught.exception)
        self.assertIn("inert", message)
        self.assertNotIn("'live'", message)
        # The message names model.py so the Experimentor routes the repair to the
        # model_designer rather than to the trainer.
        self.assertIn("model.py", message)

    def test_a_model_without_parameter_blocks_is_a_contract_error(self):
        train = load_seed_module("train")
        with self.assertRaises(RuntimeError) as caught:
            train.parameter_snapshot(object())
        self.assertIn("parameter_blocks", str(caught.exception))

    def test_dead_gradient_message_routes_to_the_model_designer(self):
        from agentic_recsys.experimentor import Experimentor
        from agentic_recsys.schemas import FailureKind

        train = load_seed_module("train")
        model = self.FrozenModel()
        before = train.parameter_snapshot(model)
        model.live += 0.1
        try:
            train.assert_parameters_moved(model, before)
        except RuntimeError as exc:
            stderr = f"Traceback (most recent call last):\nRuntimeError: {exc}"
        report = Experimentor.classify_process_failure(stderr, 1, 1)
        self.assertEqual(report.kind, FailureKind.SEMANTIC)
        self.assertEqual(report.responsible_agents, ["model_designer"])


class UsageNormalisationTests(unittest.TestCase):
    def test_responses_and_chat_shapes_normalise_alike(self):
        responses = normalize_usage({
            "input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
            "input_tokens_details": {"cached_tokens": 64},
            "output_tokens_details": {"reasoning_tokens": 8},
        })
        chat = normalize_usage({"prompt_tokens": 100, "completion_tokens": 20})
        self.assertEqual(responses["input_tokens"], chat["input_tokens"])
        self.assertEqual(responses["output_tokens"], chat["output_tokens"])
        self.assertEqual(responses["total_tokens"], 120)
        self.assertEqual(chat["total_tokens"], 120)
        self.assertEqual(responses["cached_input_tokens"], 64)
        self.assertEqual(responses["reasoning_tokens"], 8)
        self.assertNotIn("cached_input_tokens", chat)

    def test_missing_or_malformed_usage_is_empty_rather_than_wrong(self):
        self.assertEqual(normalize_usage(None), {})
        self.assertEqual(normalize_usage("47 tokens"), {})
        self.assertNotIn("input_tokens", normalize_usage({"unexpected": 1}))


class ProviderMetadataTests(unittest.TestCase):
    @staticmethod
    def _response(body):
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = json.dumps(body).encode("utf-8")
        return response

    @mock.patch("agentic_recsys.llm.urllib.request.urlopen")
    def test_responses_api_usage_and_served_model_are_captured(self, urlopen):
        urlopen.return_value = self._response({
            "id": "resp_1",
            "model": "gpt-test-2026-01-01",
            "usage": {"input_tokens": 11, "output_tokens": 3, "total_tokens": 14},
            "output": [{
                "type": "message",
                "content": [{"type": "output_text", "text": '{"ok": true}'}],
            }],
        })
        client = OpenAICompatibleClient("gpt-test", api_key="secret")
        client.complete_json(role="judge", system="s", payload={})
        metadata = client.last_metadata()
        self.assertEqual(metadata["model"], "gpt-test-2026-01-01")
        self.assertEqual(metadata["usage"]["input_tokens"], 11)
        self.assertEqual(metadata["usage"]["output_tokens"], 3)
        self.assertEqual(metadata["response_id"], "resp_1")

    @mock.patch("agentic_recsys.llm.urllib.request.urlopen")
    def test_chat_api_usage_is_captured(self, urlopen):
        urlopen.return_value = self._response({
            "model": "local-model",
            "usage": {"prompt_tokens": 7, "completion_tokens": 2},
            "choices": [{"message": {"content": '{"ok": true}'}}],
        })
        client = OpenAICompatibleClient("requested", api_key="k", api_mode="chat")
        client.complete_json(role="judge", system="s", payload={})
        self.assertEqual(client.last_metadata()["model"], "local-model")
        self.assertEqual(client.last_metadata()["usage"]["total_tokens"], 9)

    @mock.patch("agentic_recsys.llm.urllib.request.urlopen")
    def test_a_failed_call_does_not_report_the_previous_call_usage(self, urlopen):
        client = OpenAICompatibleClient("gpt-test", api_key="secret")
        urlopen.return_value = self._response({
            "model": "gpt-test",
            "usage": {"input_tokens": 40, "output_tokens": 5},
            "output": [{
                "type": "message",
                "content": [{"type": "output_text", "text": '{"ok": true}'}],
            }],
        })
        client.complete_json(role="judge", system="s", payload={})
        urlopen.side_effect = OSError("network down")
        with self.assertRaises(Exception):
            client.complete_json(role="judge", system="s", payload={})
        self.assertEqual(client.last_metadata()["usage"], {})


class AuditedUsageTests(unittest.TestCase):
    class MeteredClient:
        def __init__(self, usage):
            self.usage = usage

        def complete_json(self, *, role, system, payload):
            return {"ok": role}

        def last_metadata(self):
            return {"model": "metered-model", "usage": dict(self.usage)}

    class FailingClient:
        def complete_json(self, *, role, system, payload):
            raise RuntimeError("provider exploded")

        def last_metadata(self):
            return {"model": "metered-model", "usage": {}}

    def test_events_carry_model_usage_and_latency(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "llm_events.jsonl"
            client = AuditedLLMClient(
                self.MeteredClient({"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}),
                log,
            )
            client.complete_json(role="orchestrator", system="s", payload={})
            event = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(event["model"], "metered-model")
            self.assertEqual(event["usage"]["input_tokens"], 10)
            self.assertEqual(event["usage"]["output_tokens"], 4)
            self.assertIsInstance(event["duration_seconds"], float)
            self.assertEqual(event["role"], "orchestrator")

    def test_totals_accumulate_across_calls_and_roles(self):
        with tempfile.TemporaryDirectory() as directory:
            client = AuditedLLMClient(
                self.MeteredClient({"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}),
                Path(directory) / "llm_events.jsonl",
            )
            for role in ("evolution_judge", "consultant", "model_designer"):
                client.complete_json(role=role, system="s", payload={})
            totals = client.usage_report()
            self.assertEqual(totals["calls"], 3)
            self.assertEqual(totals["input_tokens"], 30)
            self.assertEqual(totals["output_tokens"], 12)
            self.assertEqual(totals["total_tokens"], 42)
            self.assertEqual(totals["models"], {"metered-model": 3})
            self.assertEqual(totals["calls_without_usage"], 0)

    def test_a_failed_call_is_still_counted_and_logged(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "llm_events.jsonl"
            client = AuditedLLMClient(self.FailingClient(), log)
            with self.assertRaises(RuntimeError):
                client.complete_json(role="trainer", system="s", payload={})
            event = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(event["status"], "error")
            self.assertEqual(event["model"], "metered-model")
            self.assertEqual(client.usage_report()["calls"], 1)
            self.assertEqual(client.usage_report()["calls_without_usage"], 1)

    def test_clients_that_cannot_meter_are_still_identified(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "llm_events.jsonl"
            client = AuditedLLMClient(ScriptedLLMClient([{"ok": True}]), log)
            client.complete_json(role="consultant", system="s", payload={})
            event = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(event["model"], "scripted")
            self.assertEqual(event["usage"], {})
            self.assertEqual(client.usage_report()["calls_without_usage"], 1)


if __name__ == "__main__":
    unittest.main()
