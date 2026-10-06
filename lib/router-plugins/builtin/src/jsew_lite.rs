// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `dynamo-jsew-lite`: the JSEW policy behind one knob.
//!
//! [`crate::jsew`] exposes six tunables that were tuned jointly and are coupled: the objective
//! surface around them is stepped, the exponents sit near cliffs, and moving one without the
//! others lands on the wrong ridge. This policy collapses them into a single `intensity` dial
//! that interpolates every internal between Dynamo's default cost and the tuned JSEW point:
//!
//! ```text
//! decode_share_exponent  = 0.13  * intensity
//! decode_load_exponent   = 1 + 0.2 * intensity
//! decode_footprint_weight= 1 + 0.526 * intensity
//! decode_work_scale      = 17.601 * intensity
//! ```
//!
//! with the EMA horizon, the size-ratio exponent and the predictor fixed at the tuned values.
//! `intensity: 0` is byte-identical to the default policy; `intensity: 1` is byte-identical to
//! `dynamo-jsew` at its shipped defaults; values up to 1.1 push slightly further along the same ray. Both
//! anchors are tested by equality, so a one-dimensional sweep of this knob reuses everything
//! already measured for the two endpoints.

use std::sync::Arc;

use dynamo_kv_router::KvRouterConfig;
use dynamo_kv_router::plugins::worker_selection::WorkerSelectionPolicyFactory;
use dynamo_kv_router::plugins::{
    RouterPluginRegistry, WorkerSelectionPolicyParameters, WorkerSelectionPolicyProviderError,
    WorkerSelectionPolicyRegistryError,
};

use crate::jsew;

/// Policy type selected by `worker_selection.instances[].type`.
pub const POLICY_TYPE: &str = "dynamo-jsew-lite";

/// Tunables for [`POLICY_TYPE`]. One knob; unknown keys fail startup.
#[derive(Debug, Clone, Copy, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
struct Parameters {
    /// How far from the default cost toward the tuned JSEW point to go. `0.0` is the default
    /// policy, `1.0` the shipped `dynamo-jsew`. Must be finite and in `[0.0, 1.1]`; the upper
    /// bound keeps every derived internal inside the ranges `dynamo-jsew` validates
    /// (its work scale caps at 20, and 17.601 x 1.1 is the last step under it).
    intensity: f64,
}

impl Default for Parameters {
    fn default() -> Self {
        Self { intensity: 1.0 }
    }
}

impl Parameters {
    fn validate(&self) -> Result<(), WorkerSelectionPolicyProviderError> {
        if !self.intensity.is_finite() || !(0.0..=1.1).contains(&self.intensity) {
            return Err(WorkerSelectionPolicyProviderError::new(
                "intensity must be a finite number between 0.0 and 1.1",
            ));
        }
        Ok(())
    }
}

/// The full JSEW parameter set this intensity stands for.
fn expand(intensity: f64) -> jsew::Parameters {
    jsew::Parameters {
        effective_work_predictor: jsew::EffectiveWorkPredictor::EmaUncachedTokens,
        decode_work_scale: jsew::DEFAULT_DECODE_WORK_SCALE * intensity,
        decode_footprint_weight: 1.0 + (jsew::DEFAULT_DECODE_FOOTPRINT_WEIGHT - 1.0) * intensity,
        ema_alpha: jsew::DEFAULT_EMA_ALPHA,
        decode_work_exponent: jsew::DEFAULT_DECODE_WORK_EXPONENT,
        decode_load_exponent: 1.0 + (jsew::DEFAULT_DECODE_LOAD_EXPONENT - 1.0) * intensity,
        decode_share_exponent: jsew::DEFAULT_DECODE_SHARE_EXPONENT * intensity,
    }
}

fn provider(
    parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    let parameters: Parameters = parameters.deserialize()?;
    parameters.validate()?;
    let expanded = expand(parameters.intensity);
    expanded.validate()?;
    Ok(Arc::new(
        move |config: &KvRouterConfig, role, _partition| jsew::policy(config, expanded, role),
    ))
}

pub fn register(
    registry: &mut RouterPluginRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register_worker_selection(POLICY_TYPE, Arc::new(provider))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use dynamo_kv_router::plugins::worker_selection::WorkerSelectionPolicy;
    use dynamo_kv_router::protocols::{RoutingConstraints, WorkerConfigLike, WorkerWithDpRank};
    use dynamo_kv_router::scheduling::{OverlapSignals, ScheduleMode};
    use dynamo_kv_router::{
        SchedulingRequest, WorkerLoadProjection, WorkerSelectionInput, WorkerSelector, WorkerType,
    };

    use super::*;
    use crate::default::default_policy;

    const BLOCK_SIZE: u32 = 16;
    const A: u64 = 29;
    const B: u64 = 41;

    struct TestWorker;

    impl WorkerConfigLike for TestWorker {
        fn data_parallel_start_rank(&self) -> u32 {
            0
        }
        fn data_parallel_size(&self) -> u32 {
            1
        }
        fn max_num_batched_tokens(&self) -> Option<u64> {
            None
        }
        fn total_kv_blocks(&self) -> Option<u64> {
            Some(1_000_000)
        }
    }

    fn worker(id: u64) -> WorkerWithDpRank {
        WorkerWithDpRank::from_worker_id(id)
    }

    /// `(worker_id, overlap_blocks, active_decode_blocks, active_requests)`.
    type WorkerSpec = (u64, usize, usize, usize);

    fn request(workers: [WorkerSpec; 2]) -> SchedulingRequest {
        let mut request = SchedulingRequest {
            mode: ScheduleMode::QueryOnly { request_id: None },
            token_seq: None,
            isl_tokens: 160,
            lora_name: None,
            expected_output_tokens: None,
            affinity_target: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: RoutingConstraints::default(),
            router_config_override: None,
            track_prefill_tokens: true,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_context: None,
            overlap: OverlapSignals::default(),
            kv_transfer_candidates: None,
            retain_kv_transfer_chain: false,
            shared_cache_hits: None,
            worker_loads: Default::default(),
            resp_tx: None,
        };
        for (id, overlap_blocks, active_decode_blocks, active_requests) in workers {
            request
                .overlap
                .effective_overlap_blocks
                .insert(worker(id), overlap_blocks as f64);
            request
                .overlap
                .effective_cached_tokens
                .insert(worker(id), overlap_blocks * BLOCK_SIZE as usize);
            request.worker_loads.insert(
                worker(id),
                WorkerLoadProjection {
                    active_decode_blocks,
                    active_requests,
                    ..Default::default()
                },
            );
        }
        request
    }

    /// Route the same sequence through a policy; the EMA carries across the sequence.
    fn picks<const N: usize>(
        policy: &WorkerSelectionPolicy,
        sequence: [[WorkerSpec; 2]; N],
    ) -> Vec<WorkerWithDpRank> {
        sequence
            .into_iter()
            .map(|workers| {
                let request = request(workers);
                let configs: HashMap<u64, TestWorker> =
                    workers.iter().map(|&(id, ..)| (id, TestWorker)).collect();
                policy
                    .select_worker(WorkerSelectionInput::configured(
                        &configs,
                        &request,
                        request.eligibility(),
                        BLOCK_SIZE,
                    ))
                    .unwrap()
                    .worker
            })
            .collect()
    }

    /// Cases where cache overlap, footprint and sharing disagree, so any drift shows up.
    const SEQUENCE: [[WorkerSpec; 2]; 7] = [
        [(A, 0, 0, 1), (B, 8, 40, 1)],
        [(A, 6, 30, 2), (B, 0, 0, 1)],
        [(A, 10, 100, 5), (B, 2, 5, 1)],
        [(A, 3, 12, 1), (B, 4, 20, 6)],
        [(A, 0, 7, 1), (B, 5, 9, 1)],
        [(A, 0, 30, 1), (B, 5, 34, 1)],
        [(A, 0, 8, 1), (B, 5, 11, 3)],
    ];

    fn lite(intensity: f64) -> WorkerSelectionPolicy {
        jsew::policy(
            &KvRouterConfig::default(),
            expand(intensity),
            WorkerType::Aggregated,
        )
    }

    #[test]
    fn registers_its_policy_type() {
        let mut registry = RouterPluginRegistry::default();
        register(&mut registry).unwrap();
        assert_eq!(POLICY_TYPE, "dynamo-jsew-lite");
        assert!(register(&mut registry).is_err());
    }

    #[test]
    fn intensity_zero_is_the_default_policy() {
        let stock = default_policy(KvRouterConfig::default(), "aggregated");
        assert_eq!(picks(&lite(0.0), SEQUENCE), picks(&stock, SEQUENCE));
    }

    #[test]
    fn intensity_one_is_the_shipped_jsew() {
        let full = jsew::policy(
            &KvRouterConfig::default(),
            jsew::Parameters::default(),
            WorkerType::Aggregated,
        );
        assert_eq!(picks(&lite(1.0), SEQUENCE), picks(&full, SEQUENCE));
        // The two endpoints must differ somewhere, or the dial would be vacuous.
        let stock = default_policy(KvRouterConfig::default(), "aggregated");
        assert_ne!(picks(&lite(1.0), SEQUENCE), picks(&stock, SEQUENCE));
    }

    #[test]
    fn every_intensity_in_range_expands_to_valid_jsew_parameters() {
        for i in 0..=11 {
            let intensity = i as f64 / 10.0;
            expand(intensity)
                .validate()
                .unwrap_or_else(|e| panic!("intensity {intensity}: {e}"));
        }
    }

    #[test]
    fn rejects_out_of_range_intensity() {
        let check = |v| Parameters { intensity: v }.validate();
        assert!(check(-0.1).is_err() && check(1.2).is_err() && check(f64::NAN).is_err());
        assert!(check(0.0).is_ok() && check(1.1).is_ok());
    }
}
