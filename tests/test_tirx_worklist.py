"""CPU-only tests for the metadata contract shared with the TIRx kernel."""

from __future__ import annotations

import unittest

from blackwell_moe_tirx.config import (
    TIRxMoESpec,
    build_padding_aware_plans,
    build_workload_plan,
    chunked_claims,
    hybrid_claims,
    static_cta_assignments,
)
from blackwell_moe_tirx.kernels import get_descriptor, list_versions
from blackwell_moe_tirx.dispatch import routing_features, select_kernel


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
            [[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1]],
        )

    def test_shape_contract_accepts_layout_d_and_layout_f_tiles(self) -> None:
        TIRxMoESpec(tile_m=128).validate()
        TIRxMoESpec(tile_m=64).validate()
        with self.assertRaisesRegex(ValueError, "64x128x64 and 128x128x64"):
            TIRxMoESpec(tile_m=32).validate()

    def test_m64_worklist_uses_64_row_indices(self) -> None:
        spec = TIRxMoESpec(
            experts=1, tokens=129, n=128, k=64, tile_m=64
        )
        plan = build_workload_plan(spec, token_counts=(129,))
        self.assertEqual(
            plan.device_worklist(),
            [[0, 0, 0], [0, 1, 0], [0, 2, 0]],
        )

    def test_padding_aware_buckets_cover_rows_once_and_reduce_padding(self) -> None:
        spec = TIRxMoESpec(experts=4, tokens=450, n=256, k=64)
        bucketed = build_padding_aware_plans(
            spec, token_counts=(0, 63, 130, 257)
        )

        self.assertEqual(bucketed.large.tile_m, 128)
        self.assertEqual(bucketed.small.tile_m, 64)
        self.assertEqual(bucketed.large.m_capacity, bucketed.small.m_capacity)
        self.assertEqual(
            bucketed.logical_flops,
            2 * spec.tokens * spec.n * spec.k,
        )
        self.assertGreater(bucketed.padding_reduction, 0.0)

        covered: dict[tuple[int, int], int] = {}
        for plan in (bucketed.large, bucketed.small):
            for tile in plan.tiles:
                if tile.tile_n == 0:
                    key = (tile.expert_id, tile.tile_m)
                    self.assertNotIn(key, covered)
                    covered[key] = tile.valid_m
        for expert_id, count in enumerate((0, 63, 130, 257)):
            self.assertEqual(
                sum(
                    valid
                    for (owner, _offset), valid in covered.items()
                    if owner == expert_id
                ),
                count,
            )

    def test_static_persistent_mapping_is_exactly_once(self) -> None:
        assignments = static_cta_assignments(tile_count=353, cta_count=148)
        flattened = [tile for cta in assignments for tile in cta]
        self.assertEqual(sorted(flattened), list(range(353)))
        self.assertLessEqual(
            max(map(len, assignments)) - min(map(len, assignments)),
            1,
        )

    def test_chunked_claims_cover_each_tile_once(self) -> None:
        for claim_size in (1, 2, 4, 8):
            batches = chunked_claims(353, claim_size)
            flattened = [tile for batch in batches for tile in batch]
            self.assertEqual(flattened, list(range(353)))
            self.assertTrue(all(1 <= len(batch) <= claim_size for batch in batches))

    def test_hybrid_claims_use_fine_grained_tail(self) -> None:
        batches = hybrid_claims(353, main_claim_size=8, tail_tiles=37)
        flattened = [tile for batch in batches for tile in batch]
        self.assertEqual(flattened, list(range(353)))
        self.assertTrue(all(len(batch) == 1 for batch in batches[-37:]))
        self.assertTrue(all(len(batch) <= 8 for batch in batches[:-37]))

    def test_version_registry_has_complete_optimization_journey(self) -> None:
        self.assertEqual(
            list_versions(),
            (
                "v0_nonpersistent",
                "v0_5_persistent",
                "v1_static_ws",
                "v2_dynamic",
                "v3_chunked",
                "v4_hybrid",
                "v5_clc",
                "v6_small_m_ws",
            ),
        )

    def test_launch_and_queue_contracts(self) -> None:
        spec = TIRxMoESpec(experts=4, tokens=256, n=256, k=64)
        plan = build_workload_plan(spec, token_counts=(64, 64, 64, 64))
        self.assertEqual(
            get_descriptor("v0_nonpersistent").launch_ctas(spec, plan),
            len(plan.tiles),
        )
        self.assertEqual(
            get_descriptor("v1_static_ws").launch_ctas(spec, plan),
            spec.cta_count,
        )
        self.assertTrue(get_descriptor("v2_dynamic").requires_queue)
        self.assertFalse(get_descriptor("v5_clc").requires_queue)
        self.assertEqual(get_descriptor("v6_small_m_ws").tile_m, 64)

        hybrid_spec = TIRxMoESpec(
            experts=4,
            tokens=256,
            n=256,
            k=64,
            hybrid_tail_tiles=3,
        )
        hybrid_plan = build_workload_plan(
            hybrid_spec, token_counts=(64, 64, 64, 64)
        )
        self.assertEqual(
            get_descriptor("v4_hybrid").queue_initial_values(
                hybrid_spec, hybrid_plan
            ),
            (0, len(hybrid_plan.tiles) - 3),
        )

    def test_routing_features_and_bootstrap_dispatch_are_host_side(self) -> None:
        spec = TIRxMoESpec(experts=4, tokens=256, n=256, k=64, cta_count=2)
        uniform = build_workload_plan(spec, token_counts=(64, 64, 64, 64))
        features = routing_features(spec, uniform)
        self.assertEqual(features.cv_m, 0.0)
        self.assertEqual(features.inactive_expert_ratio, 0.0)
        self.assertEqual(features.expert_tile_cv, 0.0)
        self.assertEqual(select_kernel(spec, uniform), "v1_static_ws")

        low_parallelism_spec = TIRxMoESpec(
            experts=4, tokens=4, n=128, k=64, cta_count=148
        )
        low_parallelism = build_workload_plan(
            low_parallelism_spec, token_counts=(1, 1, 1, 1)
        )
        self.assertEqual(
            select_kernel(low_parallelism_spec, low_parallelism),
            "v0_nonpersistent",
        )


if __name__ == "__main__":
    unittest.main()
