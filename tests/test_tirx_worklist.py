"""CPU-only tests for the metadata contract shared with the TIRx kernel."""

from __future__ import annotations

import unittest

from blackwell_moe_tirx.config import (
    TIRxMoESpec,
    build_workload_plan,
    static_cta_assignments,
)


class TIRxWorklistTest(unittest.TestCase):
    def test_distributions_are_deterministic_and_conserve_tokens(self) -> None:
        spec = TIRxMoESpec(experts=16, tokens=1024, n=512, k=256)
        for distribution in ("uniform", "heavy_hitter", "sparse", "zipf"):
            first = build_workload_plan(spec, distribution, seed=2026)
            second = build_workload_plan(spec, distribution, seed=2026)
            self.assertEqual(first.tokens_per_expert, second.tokens_per_expert)
            self.assertEqual(sum(first.tokens_per_expert), spec.tokens)

    def test_compaction_and_tile_coverage(self) -> None:
        spec = TIRxMoESpec(experts=4, tokens=290, n=256, k=64)
        plan = build_workload_plan(spec, token_counts=(0, 1, 128, 161))

        self.assertEqual(plan.active_experts, 3)
        self.assertEqual(plan.m_capacity, 256)
        self.assertTrue(all(tile.expert_id != 0 for tile in plan.tiles))

        expected_tiles = (1 + 1 + 2) * (spec.n // spec.tile_n)
        self.assertEqual(len(plan.tiles), expected_tiles)
        for expert_id, count in enumerate(plan.tokens_per_expert):
            covered_rows = {
                tile.tile_m: tile.valid_m
                for tile in plan.tiles
                if tile.expert_id == expert_id and tile.tile_n == 0
            }
            self.assertEqual(sum(covered_rows.values()), count)

    def test_worklist_is_canonical_expert_major_order(self) -> None:
        spec = TIRxMoESpec(experts=2, tokens=129, n=256, k=64)
        plan = build_workload_plan(spec, token_counts=(128, 1))
        self.assertEqual(
            plan.device_worklist(),
            [[0, 0, 0], [0, 0, 128], [1, 0, 0], [1, 0, 128]],
        )

    def test_shape_contract_rejects_unsupported_tiles(self) -> None:
        with self.assertRaisesRegex(ValueError, "128x128x64"):
            TIRxMoESpec(tile_m=64).validate()

    def test_static_persistent_mapping_is_exactly_once(self) -> None:
        assignments = static_cta_assignments(tile_count=353, cta_count=148)
        flattened = [tile for cta in assignments for tile in cta]
        self.assertEqual(sorted(flattened), list(range(353)))
        self.assertLessEqual(
            max(map(len, assignments)) - min(map(len, assignments)),
            1,
        )


if __name__ == "__main__":
    unittest.main()
