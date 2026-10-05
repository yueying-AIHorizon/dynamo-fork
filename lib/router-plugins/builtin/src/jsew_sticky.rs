// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Join-shortest-expected-work with a session home (`dynamo-jsew-sticky`).
//!
//! An in-process port of the JSEW external router (`examples/router/jsew-external-router`
//! on the fork), which ran as a proxy in front of a `--router-mode direct` frontend. The rule is
//! the proxy's, with the proxy's own bookkeeping replaced by the host's signals:
//!
//! ```text
//! cost_k = active_prefill_tokens_k            // prefill still queued on worker k
//!        + uncached_tokens_k                  // this prompt minus the prefix k already holds
//!        + expected_output_tokens * active_requests_k   // decode work k still owes
//! pick   = the session's home worker, unless another worker's cost is lower by more than
//!          `hysteresis`; without a session, the minimum cost. Ties go to the lowest worker id.
//! ```
//!
//! What changed in the port:
//!
//! - Prefix knowledge comes from the router's KV-event indexer (device tier when reported, else
//!   the host's accounting estimate) instead of a proxy-side shadow LRU, so it is corrected by
//!   engine events and sees evictions.
//! - Queued prefill is the host's `active_prefill_tokens`, which covers every request on the
//!   worker, not only traffic that passed through a proxy.
//! - The proxy learned remaining decode length from completions, conditioned on each request's
//!   attained output. A plugin sees neither completions nor per-request progress, so the decode
//!   term is `active_requests` times an EMA of the output lengths requests *declare*
//!   (`expected_output_tokens`, i.e. `max_tokens` in production, the recorded length in replay).
//!   Requests that declare nothing leave the belief unchanged; until any request declares, the
//!   term is zero and the rule is join-shortest-prefill-plus-uncached.
//! - The request's own expected decode is the same for every worker and cancels out of the
//!   argmin, so it is not added. Per-worker capacity normalization is omitted: the plugin API
//!   does not expose capacity yet, and the proxy never set it either.
//! - The session home is a rendezvous hash of the session id over the eligible workers, so
//!   adding or removing a worker moves only the sessions it wins or held. When the host supplies
//!   an advisory affinity target (soft session affinity), that target is the home instead.
//!
//! Two parameters, no exponents. This is a different cost from Dynamo's default, so an A/B
//! against it compares two policies; use `dynamo-jsew` when the hypothesis under test must
//! reduce to the default at neutral parameters.

use std::sync::Arc;

use dynamo_kv_router::KvRouterConfig;
use dynamo_kv_router::plugins::worker_selection::{
    WorkerCacheInput, WorkerCandidates, WorkerInputView, WorkerInputs, WorkerPicker, WorkerScorer,
    WorkerSelectionContext, WorkerSelectionPolicy, WorkerSelectionPolicyError,
    WorkerSelectionPolicyFactory,
};
use dynamo_kv_router::plugins::{
    RouterPluginRegistry, WorkerSelectionPolicyParameters, WorkerSelectionPolicyProviderError,
    WorkerSelectionPolicyRegistryError,
};
use dynamo_kv_router::protocols::WorkerWithDpRank;

/// Policy type selected by `worker_selection.instances[].type`.
pub const POLICY_TYPE: &str = "dynamo-jsew-sticky";

/// The proxy kept a session on its home unless another worker was better by 5%.
const DEFAULT_HYSTERESIS: f64 = 0.05;
/// Smoothing for the declared-output-length belief; about ten requests of memory.
const DEFAULT_EMA_ALPHA: f64 = 0.1;

/// Tunables for [`POLICY_TYPE`]. Every field is optional; unknown keys fail startup.
#[derive(Debug, Clone, Copy, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
struct Parameters {
    /// Fraction by which another worker's cost must undercut the session home's cost before the
    /// session moves. `0.0` moves on any improvement; the home still wins exact ties.
    hysteresis: f64,
    /// Smoothing factor, in `(0, 1]`, of the EMA over declared output lengths that estimates the
    /// decode work each active request still owes.
    ema_alpha: f64,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            hysteresis: DEFAULT_HYSTERESIS,
            ema_alpha: DEFAULT_EMA_ALPHA,
        }
    }
}

impl Parameters {
    fn validate(&self) -> Result<(), WorkerSelectionPolicyProviderError> {
        if !self.hysteresis.is_finite() || !(0.0..1.0).contains(&self.hysteresis) {
            return Err(WorkerSelectionPolicyProviderError::new(
                "hysteresis must be a finite number in [0.0, 1.0)",
            ));
        }
        if !self.ema_alpha.is_finite() || self.ema_alpha <= 0.0 || self.ema_alpha > 1.0 {
            return Err(WorkerSelectionPolicyProviderError::new(
                "ema_alpha must be a finite number in (0.0, 1.0]",
            ));
        }
        Ok(())
    }
}

/// Prompt tokens worker already holds, on the basis the default scorer uses for its prefill
/// estimate: device-tier matches when the host reports tiers, else the accounting estimate.
fn cached_tokens(context: &WorkerSelectionContext<'_>, cache: WorkerCacheInput<'_>) -> usize {
    if cache.has_tier_matches() {
        (cache.device_overlap_blocks() * f64::from(context.block_size()))
            .round()
            .max(0.0) as usize
    } else {
        cache.accounting_cache_estimate().1
    }
}

struct JsewStickyScorer {
    ema_alpha: f64,
    /// EMA of declared output lengths, in tokens. `None` until a request declares one.
    expected_output_tokens: Option<f64>,
}

impl WorkerScorer for JsewStickyScorer {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::CACHE | WorkerInputs::LOAD
    }

    fn score(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        candidates: WorkerCandidates<'_>,
        costs: &mut [f64],
    ) -> Result<(), WorkerSelectionPolicyError> {
        // Fold the declaration once per request, before scoring, so every candidate is charged
        // against the same belief.
        if let Some(declared) = context.expected_output_tokens() {
            let declared = f64::from(declared);
            self.expected_output_tokens = Some(match self.expected_output_tokens {
                Some(previous) => self.ema_alpha * declared + (1.0 - self.ema_alpha) * previous,
                None => declared,
            });
        }
        let expected_output = self.expected_output_tokens.unwrap_or(0.0);
        for (candidate, cost) in candidates.iter().zip(costs) {
            let load = candidate
                .load()
                .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
            let cached = candidate
                .cache()
                .map_or(0, |cache| cached_tokens(context, cache));
            let uncached = context.prompt_tokens().saturating_sub(cached);
            *cost = load.active_prefill_tokens() as f64
                + uncached as f64
                + expected_output * load.active_requests() as f64;
        }
        Ok(())
    }
}

fn mix(mut value: u64) -> u64 {
    // SplitMix64 finalizer.
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn hash_session(session_id: &str) -> u64 {
    // FNV-1a over the id bytes, then mixed; stable across processes and replicas.
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in session_id.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    mix(hash)
}

/// Rendezvous (highest-random-weight) weight of `session` on `worker`. Ranking workers by this
/// maps a session consistently: adding or removing a worker moves only the sessions it wins or
/// held, which is what keeps a session's KV on one worker across membership changes.
fn rendezvous(session: u64, worker: WorkerWithDpRank) -> u64 {
    mix(session ^ mix(worker.worker_id ^ mix(u64::from(worker.dp_rank))))
}

struct JsewStickyPicker {
    hysteresis: f64,
}

impl WorkerPicker for JsewStickyPicker {
    fn pick(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        let candidates = input.candidates();
        if candidates.is_empty() {
            return Err(WorkerSelectionPolicyError::failed("no eligible worker"));
        }
        if context.pinned_worker().is_some() {
            return Ok(0);
        }
        // Minimum cost; ties go to the lowest worker id so selection does not depend on the
        // host's unspecified row order.
        let best = (0..candidates.len())
            .min_by(|&a, &b| {
                candidates[a]
                    .cost()
                    .total_cmp(&candidates[b].cost())
                    .then_with(|| {
                        order_key(candidates[a].worker()).cmp(&order_key(candidates[b].worker()))
                    })
            })
            .expect("non-empty candidates");

        let Some(home) = self.home_row(context, candidates) else {
            return Ok(best);
        };
        let home_cost = candidates[home].cost();
        let best_cost = candidates[best].cost();
        if best_cost < home_cost * (1.0 - self.hysteresis) {
            Ok(best)
        } else {
            Ok(home)
        }
    }
}

fn order_key(worker: WorkerWithDpRank) -> (u64, u32) {
    (worker.worker_id, worker.dp_rank)
}

impl JsewStickyPicker {
    /// The session's home row: the host's advisory affinity target when it is among the
    /// candidates, else the rendezvous winner for the session id. `None` without a session.
    fn home_row(
        &self,
        context: &WorkerSelectionContext<'_>,
        candidates: &[dynamo_kv_router::plugins::worker_selection::ScoredWorkerCandidate],
    ) -> Option<usize> {
        if let Some(target) = context.affinity_target() {
            let row = candidates.iter().position(|candidate| {
                let worker = candidate.worker();
                worker.worker_id == target.worker_id
                    && target.dp_rank.is_none_or(|rank| rank == worker.dp_rank)
            });
            if row.is_some() {
                return row;
            }
        }
        let session = hash_session(context.session_context()?.session_id());
        (0..candidates.len()).max_by_key(|&row| {
            let worker = candidates[row].worker();
            (
                rendezvous(session, worker),
                std::cmp::Reverse(order_key(worker)),
            )
        })
    }
}

fn policy(
    config: &KvRouterConfig,
    parameters: Parameters,
    role: dynamo_kv_router::WorkerType,
) -> WorkerSelectionPolicy {
    // Exclusive affinity stays off: a soft affinity target is the session home, and the
    // hysteresis band, not the host, decides whether the session stays on it.
    WorkerSelectionPolicy::new(
        config.clone(),
        role.default_selector_label(),
        vec![Box::new(JsewStickyScorer {
            ema_alpha: parameters.ema_alpha,
            expected_output_tokens: None,
        })],
        Box::new(JsewStickyPicker {
            hysteresis: parameters.hysteresis,
        }),
    )
}

fn provider(
    parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    let parameters: Parameters = parameters.deserialize()?;
    parameters.validate()?;
    Ok(Arc::new(
        move |config: &KvRouterConfig, role, _partition| policy(config, parameters, role),
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

    use dynamo_kv_router::protocols::{RoutingConstraints, WorkerConfigLike};
    use dynamo_kv_router::scheduling::{OverlapSignals, ScheduleMode};
    use dynamo_kv_router::{
        SchedulingRequest, SessionContext, WorkerLoadProjection, WorkerSelectionInput,
        WorkerSelector, WorkerType,
    };

    use super::*;

    const BLOCK_SIZE: u32 = 16;
    const PROMPT: usize = 160;
    const A: u64 = 29;
    const B: u64 = 41;
    const C: u64 = 53;

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

    /// One worker's signals: `(worker_id, cached_tokens, active_prefill_tokens, active_requests)`.
    type WorkerSpec = (u64, usize, usize, usize);

    fn request(
        workers: &[WorkerSpec],
        session: Option<&str>,
        expected_output_tokens: Option<u32>,
    ) -> SchedulingRequest {
        let mut request = SchedulingRequest {
            mode: ScheduleMode::QueryOnly { request_id: None },
            token_seq: None,
            isl_tokens: PROMPT,
            lora_name: None,
            expected_output_tokens,
            affinity_target: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: RoutingConstraints::default(),
            router_config_override: None,
            track_prefill_tokens: true,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_context: session
                .map(|id| SessionContext::new(id.to_string(), None, None, None)),
            overlap: OverlapSignals::default(),
            kv_transfer_candidates: None,
            retain_kv_transfer_chain: false,
            shared_cache_hits: None,
            worker_loads: Default::default(),
            resp_tx: None,
        };
        for &(id, cached, active_prefill_tokens, active_requests) in workers {
            request
                .overlap
                .effective_overlap_blocks
                .insert(worker(id), cached as f64 / f64::from(BLOCK_SIZE));
            request
                .overlap
                .effective_cached_tokens
                .insert(worker(id), cached);
            request.worker_loads.insert(
                worker(id),
                WorkerLoadProjection {
                    active_prefill_tokens,
                    active_requests,
                    ..Default::default()
                },
            );
        }
        request
    }

    fn policy(parameters: Parameters) -> WorkerSelectionPolicy {
        super::policy(
            &KvRouterConfig::default(),
            parameters,
            WorkerType::Aggregated,
        )
    }

    fn select(
        policy: &WorkerSelectionPolicy,
        workers: &[WorkerSpec],
        session: Option<&str>,
        expected_output_tokens: Option<u32>,
    ) -> WorkerWithDpRank {
        let request = request(workers, session, expected_output_tokens);
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
    }

    #[test]
    fn registers_its_policy_type() {
        let mut registry = RouterPluginRegistry::default();
        register(&mut registry).unwrap();
        assert_eq!(POLICY_TYPE, "dynamo-jsew-sticky");
        assert!(register(&mut registry).is_err());
    }

    #[test]
    fn without_a_session_the_least_queued_prefill_wins() {
        let policy = policy(Parameters::default());
        assert_eq!(
            select(&policy, &[(A, 0, 300, 1), (B, 0, 100, 1)], None, None),
            worker(B)
        );
    }

    #[test]
    fn cached_prefix_offsets_queued_prefill() {
        // B holds 128 of the 160 prompt tokens: its uncached 32 plus 100 queued beats A's 160 + 0.
        let policy = policy(Parameters::default());
        assert_eq!(
            select(&policy, &[(A, 0, 0, 1), (B, 128, 100, 1)], None, None),
            worker(B)
        );
        // Hold less of the prompt and the queue decides.
        assert_eq!(
            select(&policy, &[(A, 0, 0, 1), (B, 32, 100, 1)], None, None),
            worker(A)
        );
    }

    #[test]
    fn declared_output_lengths_charge_active_requests() {
        let policy = policy(Parameters {
            ema_alpha: 1.0,
            ..Parameters::default()
        });
        // Nothing declared yet: active requests cost nothing, so the shorter prefill queue wins.
        let workers = [(A, 0, 50, 4), (B, 0, 100, 1)];
        assert_eq!(select(&policy, &workers, None, None), worker(A));
        // A request declares 100 output tokens: A's four active requests now owe 400 tokens of
        // decode, and B's one request 100, so B is the shorter expected work.
        assert_eq!(select(&policy, &workers, None, Some(100)), worker(B));
        // The belief persists for requests that declare nothing.
        assert_eq!(select(&policy, &workers, None, None), worker(B));
    }

    #[test]
    fn a_session_stays_home_inside_the_hysteresis_band() {
        let policy = policy(Parameters {
            hysteresis: 0.05,
            ..Parameters::default()
        });
        let workers = [(A, 0, 100, 1), (B, 0, 100, 1), (C, 0, 100, 1)];
        let home = select(&policy, &workers, Some("session-7"), None);
        // Make the home 3% worse than the others: still home.
        let worse = |id: u64| -> Vec<WorkerSpec> {
            workers
                .iter()
                .map(|&(w, c, p, r)| {
                    if w == id {
                        (w, c, p + 8, r)
                    } else {
                        (w, c, p, r)
                    }
                })
                .collect()
        };
        assert_eq!(
            select(&policy, &worse(home.worker_id), Some("session-7"), None),
            home
        );
        // Make it 50% worse: the session moves to the cheapest worker.
        let much_worse: Vec<WorkerSpec> = workers
            .iter()
            .map(|&(w, c, p, r)| {
                if w == home.worker_id {
                    (w, c, p + 130, r)
                } else {
                    (w, c, p, r)
                }
            })
            .collect();
        assert_ne!(select(&policy, &much_worse, Some("session-7"), None), home);
    }

    #[test]
    fn the_home_is_stable_and_differs_across_sessions() {
        let policy = policy(Parameters::default());
        let workers = [(A, 0, 0, 0), (B, 0, 0, 0), (C, 0, 0, 0)];
        let homes: Vec<_> = (0..40)
            .map(|i| select(&policy, &workers, Some(&format!("session-{i}")), None))
            .collect();
        // Every session lands on the same worker each time it is routed ...
        for (i, &home) in homes.iter().enumerate() {
            assert_eq!(
                select(&policy, &workers, Some(&format!("session-{i}")), None),
                home
            );
        }
        // ... and the sessions spread over more than one worker.
        assert!(homes.iter().any(|&h| h != homes[0]));
        // Removing a worker moves only the sessions that lived on it.
        let fewer = [(A, 0, 0, 0), (B, 0, 0, 0)];
        for (i, &home) in homes.iter().enumerate() {
            if home != worker(C) {
                assert_eq!(
                    select(&policy, &fewer, Some(&format!("session-{i}")), None),
                    home
                );
            }
        }
    }

    #[test]
    fn ties_go_to_the_lowest_worker_id() {
        let policy = policy(Parameters::default());
        assert_eq!(
            select(
                &policy,
                &[(C, 0, 0, 0), (A, 0, 0, 0), (B, 0, 0, 0)],
                None,
                None
            ),
            worker(A)
        );
    }

    #[test]
    fn rejects_out_of_range_parameters() {
        let hysteresis = |v| {
            Parameters {
                hysteresis: v,
                ..Default::default()
            }
            .validate()
        };
        let alpha = |v| {
            Parameters {
                ema_alpha: v,
                ..Default::default()
            }
            .validate()
        };
        assert!(
            hysteresis(-0.1).is_err() && hysteresis(1.0).is_err() && hysteresis(f64::NAN).is_err()
        );
        assert!(hysteresis(0.0).is_ok());
        assert!(alpha(0.0).is_err() && alpha(1.1).is_err() && alpha(f64::NAN).is_err());
        assert!(Parameters::default().validate().is_ok());
    }
}
