# Fair timing of reconstruction pipelines: a reusable protocol

A proposed checklist for comparing the cost of reconstruction pipelines fairly. It is not a published standard and contains no results from this project.


## Contract and measurement

1. **Freeze the question before timing.** Register the method commits/configuration, independent input set, work unit, hardware, allowed inputs, time boundaries, number of attempts, quality metric, failure/timeout policy, estimator and confidence interval. Mark each published row *registered*, *descriptive*, or *modelled*; a projected result is not a measured wall time. Distinguish complete end-to-end work from a named component.
2. **Match resources and input.** Run arms on the same hardware class and configuration, CPU affinity/allotment and scheduling priority, software, precision and resource limits; log accelerator lock acquisition and observed foreign occupancy rather than asserting an idle machine. Record source-frame IDs, range, order, resize, encoding, masks and per-arm processed frames. If matching is impossible, label the difference and do not claim a like-for-like speedup. Count work at the agreed source-frame, processed-frame, window or clip basis and print both frame counts and wall times; normalize only when the denominator and input semantics are stated.
3. **Time the complete declared unit.** For a process-start-to-written-output measurement, acquire the exclusive resource first, then start a monotonic wall clock at the declared process start; stop only when all required output is written and the final device work has synchronized. Log lock wait separately, and report both excluding-lock time and end-user elapsed time if relevant. Include required decode, preprocessing, model loads, setup/compilation, reconstruction, tracking, refinement, output conversion, QC, retries, failures and fallback according to the frozen contract; list omissions. Never silently exclude work present in only one arm. Device seconds, CPU work and priced resource work are separate measurements, not substitutes for wall time.
4. **Report cold and warm, latency and throughput separately.** Cold uses a fresh process and includes model loading; warm means model-resident work in a specified pass. Compare equal regimes as primary; disclose and avoid presenting cross-regime ratios as matched. Latency is time per unit without overlapping units; throughput is completed units divided by complete pass wall, with pipeline overlap and startup/drain documented. Where completed counts can differ between arms, report the completion rule and each arm's completed units per full pass second, and compare these two rates: the ratio is (C_new / T_new) / (C_base / T_base), and pass walls alone give it only for equal positive counts. The ratio is defined only when the baseline rate C_base / T_base is positive and both pass walls are valid positive durations; otherwise report it as unavailable. If a pass cost must be allocated to units, specify the allocation rule, failed/refused work and sensitivity; never call the resulting per-unit amounts independently measured.
5. **Count every attempted unit and pair quality.** For sequential latency, sum each arm's measured seconds over the same prespecified attempted unit set and report baseline/alternative ratio-of-sums, alongside each arm's attempts, successful deliveries, failures, timeouts, and the joint-delivery intersection. For overlapping throughput passes use complete pass walls on both sides, or a declared allocation that sums to each complete pass; never sum overlapping unit latencies as elapsed pass wall. Keep time spent on failed/refused/retried work; for a timed-out unit keep the actual wall through its termination and record the timeout target separately; if a unit is not run or its cost missing, mark the estimate unavailable or use only a prespecified, disclosed censoring analysis—not zero. Report modelled and measured costs separately. Show jointly delivered analysis as secondary if registered; never quietly replace the intended denominator.
6. **Report quality and uncertainty at the intended population.** Pair on the same evaluated inputs and declare quality missingness, one- and two-sided failures, any tie rule, and uncertainty at the independent cluster (e.g. video or participant), including resampling method and seed. Quality-conditioned speed or differing delivery rates cannot establish unchanged quality. Specify the metric and resampling unit for each bootstrap interval. For a confirmatory noninferiority decision, prespecify the metric and margin. Joint-failure tie treatment is a declared study-specific rule, not universal.
7. **Keep external results on their own basis.** Quote outside numbers with their published device, software, frame count, task, setup and scope; use separate blocks for incomparable protocols and never infer a hardware conversion factor. Release a schema-valid attempt log, analysis scripts, configurations and non-sensitive input identifiers, plus disclosure of withheld artifacts. The registered owner decides what can be released.

| Pre-publication check | Required evidence |
|---|---|
| Match | Same eligible units/input manifest; per-arm resource, encoding, priority and device versions; documented deviations. |
| Timing | Clock boundaries, lock wait, synchronized wall, cold/warm, latency/throughput, included stages, all retries/fallback and device/CPU costs. |
| Estimand | Complete attempt manifest; ratio of sums and delivery counts; fail/timeout/missingness policy and cluster interval; separate measured and modelled rows. |
| Quality | Paired metric, sample/cluster identity, all attempts, confidence/noninferiority rules and uncertainty. |
| External / release | Original units and hardware for published baselines; reproducible scripts, configurations and permitted logs; unavailable fields explicit. |

## Minimal per-attempt log schema

One JSON object per **arm × unit × attempt**; every scheduled attempt, including failed or timed-out ones, gets a record. A separate manifest must establish the expected arm–unit attempts and per-pass shared allocation, or missing records cannot be distinguished from unattempted units. `null` means unavailable, not zero. Timestamps are RFC 3339 UTC; elapsed is measured with a monotonic clock. Resource costs can be charged at a later accounting step, with its rule/version retained. This schema checks types and required fields, not chronology, comparability, scientific quality or whether costs sum correctly. Record the full prespecified timing contract and stage/lock trace separately.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://example.org/fair-reconstruction-attempt.schema.json",
  "title": "Reconstruction timing attempt",
  "type": "object",
  "additionalProperties": false,
  "required": ["study_id", "unit_id", "cluster_id", "arm", "attempt_index", "input_manifest_id", "resource_config_id", "boundary_contract_id", "evidence_status", "regime", "start_utc", "end_utc", "wall_seconds", "outcome", "stages_included", "quality"],
  "properties": {
    "study_id": {"type": "string", "minLength": 1},
    "unit_id": {"type": "string", "minLength": 1},
    "cluster_id": {"type": "string", "minLength": 1},
    "arm": {"type": "string", "minLength": 1},
    "attempt_index": {"type": "integer", "minimum": 0},
    "input_manifest_id": {"type": "string", "minLength": 1},
    "resource_config_id": {"type": "string", "minLength": 1},
    "boundary_contract_id": {"type": "string", "minLength": 1},
    "evidence_status": {"enum": ["registered", "descriptive", "modelled"]},
    "pass_id": {"type": ["string", "null"]},
    "regime": {"enum": ["cold_latency", "warm_latency", "cold_throughput", "warm_throughput"]},
    "start_utc": {"type": ["string", "null"], "format": "date-time"},
    "end_utc": {"type": ["string", "null"], "format": "date-time"},
    "wall_seconds": {"type": ["number", "null"], "minimum": 0},
    "lock_wait_seconds": {"type": ["number", "null"], "minimum": 0},
    "gpu_seconds": {"type": ["number", "null"], "minimum": 0},
    "cpu_core_seconds": {"type": ["number", "null"], "minimum": 0},
    "allocated_shared_seconds": {"type": ["number", "null"], "minimum": 0},
    "shared_allocation_rule_id": {"type": ["string", "null"]},
    "outcome": {"enum": ["delivered", "failed", "timeout", "refused", "not_run"]},
    "failure_reason": {"type": ["string", "null"]},
    "stages_included": {"type": "array", "items": {"type": "string"}, "uniqueItems": true},
    "source_frame_count": {"type": ["integer", "null"], "minimum": 0},
    "processed_frame_count": {"type": ["integer", "null"], "minimum": 0},
    "quality": {
      "type": "object", "additionalProperties": false,
      "required": ["status", "metric_id", "value"],
      "properties": {
        "status": {"enum": ["scored", "failed", "missing", "not_applicable"]},
        "metric_id": {"type": ["string", "null"]},
        "value": {"type": ["number", "null"]}
      }
    },
    "notes": {"type": "string"}
  },
  "allOf": [
    {"if": {"properties": {"outcome": {"enum": ["delivered", "failed", "timeout", "refused"]}}},
     "then": {"properties": {"start_utc": {"type": "string", "format": "date-time"}, "end_utc": {"type": "string", "format": "date-time"}, "wall_seconds": {"type": "number", "minimum": 0}}}},
    {"if": {"properties": {"quality": {"properties": {"status": {"const": "scored"}}}}},
     "then": {"properties": {"quality": {"properties": {"metric_id": {"type": "string", "minLength": 1}, "value": {"type": "number"}}}}}}
  ]
}
```
