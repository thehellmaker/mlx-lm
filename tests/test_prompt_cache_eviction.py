# Copyright © 2026 Apple Inc.

import argparse
import logging
import unittest
from types import SimpleNamespace

from mlx_lm.models.cache import LRUPromptCache
from mlx_lm.server import ResponseGenerator


class FakeKV:
    """A minimal stand-in for a per-layer KV cache."""

    def __init__(self, nbytes):
        self.nbytes = nbytes

    def is_trimmable(self):
        # Quantized KV caches are not trimmable. This is the production
        # case that makes old entries pile up.
        return False


def make_generator(prompt_cache, max_bytes=None, bytes_per_token=None):
    """Build a ResponseGenerator without starting its threads."""
    generator = object.__new__(ResponseGenerator)
    generator.prompt_cache = prompt_cache
    generator.model_provider = SimpleNamespace(
        cli_args=argparse.Namespace(prompt_cache_bytes=max_bytes)
    )
    generator._kv_bytes_per_token = bytes_per_token
    return generator


def fill_cache(prompt_cache, entries):
    """Insert entries as (name, token_count, nbytes_per_token).

    Each entry gets a distinct token range so they do not replace
    each other in the cache trie.
    """
    for i, (name, tokens, nbytes_per_token) in enumerate(entries):
        cache = [FakeKV(nbytes_per_token * tokens)]
        prompt_cache.insert_cache(
            "model", list(range(i * tokens, (i + 1) * tokens)), cache
        )


class TestPrefillEviction(unittest.TestCase):
    def test_no_limit_means_no_eviction(self):
        cache = LRUPromptCache()
        fill_cache(cache, [("a", 1000, 1_000)])
        generator = make_generator(cache, max_bytes=None)
        generator._make_room_for_prompt(500_000)
        self.assertEqual(len(cache), 1)
        self.assertEqual(cache.nbytes, 1_000_000)

    def test_cache_within_budget_is_untouched(self):
        cache = LRUPromptCache()
        fill_cache(cache, [("a", 1000, 1_000)])
        generator = make_generator(cache, max_bytes=10_000_000, bytes_per_token=1)
        generator._make_room_for_prompt(1_000)
        self.assertEqual(len(cache), 1)

    def test_over_budget_cache_is_trimmed_to_target(self):
        cache = LRUPromptCache()
        fill_cache(cache, [("a", 1000, 1_000), ("b", 1000, 1_000)])
        generator = make_generator(cache, max_bytes=1_500_000, bytes_per_token=1)
        generator._make_room_for_prompt(1_000)
        # Target is 1.5 MB minus 1 KB for the incoming prompt.
        self.assertLessEqual(cache.nbytes, 1_499_000)
        self.assertEqual(len(cache), 1)

    def test_uncalibrated_estimate_uses_half_the_limit(self):
        cache = LRUPromptCache()
        fill_cache(cache, [("a", 1000, 2_000)])
        generator = make_generator(cache, max_bytes=4_000_000, bytes_per_token=None)
        generator._make_room_for_prompt(100)
        # Target is 4 MB minus 2 MB reserved for the first request.
        self.assertLessEqual(cache.nbytes, 2_000_000)

    def test_huge_prompt_empties_the_cache_without_error(self):
        cache = LRUPromptCache()
        fill_cache(cache, [("a", 1000, 1_000), ("b", 1000, 1_000)])
        generator = make_generator(cache, max_bytes=1_000_000, bytes_per_token=1)
        generator._make_room_for_prompt(10_000_000)
        # The estimate exceeds the limit, so the target clamps to zero.
        self.assertEqual(cache.nbytes, 0)
        self.assertEqual(len(cache), 0)

    def test_eviction_removes_least_recently_used_first(self):
        cache = LRUPromptCache()
        fill_cache(
            cache,
            [("oldest", 1000, 1_000), ("middle", 1000, 1_000), ("newest", 1000, 1_000)],
        )
        generator = make_generator(cache, max_bytes=3_500_000, bytes_per_token=2_000)
        generator._make_room_for_prompt(500)
        # Target is 3.5 MB minus 1 MB for the incoming prompt.
        self.assertLessEqual(cache.nbytes, 2_500_000)
        self.assertEqual(len(cache), 2)
        # The oldest entry is gone; the newest one is kept.
        oldest = cache.fetch_nearest_cache("model", list(range(1000)))[0]
        newest = cache.fetch_nearest_cache("model", list(range(2000, 3000)))[0]
        self.assertIsNone(oldest)
        self.assertIsNotNone(newest)

    def test_zero_token_prompt_never_evicts(self):
        cache = LRUPromptCache()
        fill_cache(cache, [("a", 1000, 1_000)])
        generator = make_generator(cache, max_bytes=1_000_000, bytes_per_token=1)
        generator._make_room_for_prompt(0)
        self.assertEqual(cache.nbytes, 1_000_000)

    def test_single_entry_larger_than_limit_is_evicted(self):
        cache = LRUPromptCache()
        fill_cache(cache, [("only", 1000, 5_000)])
        generator = make_generator(cache, max_bytes=1_000_000, bytes_per_token=1)
        generator._make_room_for_prompt(1_000)
        self.assertEqual(cache.nbytes, 0)


class TestKvBytesPerTokenCalibration(unittest.TestCase):
    def test_calibration_learns_real_cost(self):
        cache = LRUPromptCache()
        generator = make_generator(cache, max_bytes=1_000_000)
        generator._update_kv_bytes_per_token(list(range(1000)), [FakeKV(5_000)])
        self.assertEqual(generator._kv_bytes_per_token, 5.0)

    def test_empty_cache_key_is_skipped(self):
        cache = LRUPromptCache()
        generator = make_generator(cache, max_bytes=1_000_000, bytes_per_token=None)
        generator._update_kv_bytes_per_token([], [FakeKV(5_000)])
        self.assertIsNone(generator._kv_bytes_per_token)


class TestEstimatedKvBytes(unittest.TestCase):
    def test_calibrated_estimate_scales_with_tokens(self):
        generator = make_generator(LRUPromptCache(), max_bytes=1_000_000, bytes_per_token=3.0)
        self.assertEqual(generator._estimated_kv_bytes_for(2000), 6000)

    def test_uncalibrated_estimate_uses_half_the_limit(self):
        generator = make_generator(LRUPromptCache(), max_bytes=4_000_000, bytes_per_token=None)
        self.assertEqual(generator._estimated_kv_bytes_for(2000), 2_000_000)


if __name__ == "__main__":
    logging.disable(logging.CRITICAL)
    unittest.main()