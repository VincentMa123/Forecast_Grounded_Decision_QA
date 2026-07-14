# Native PipeClaw teacher-trace evaluation

`teacher_quality.py` validates unsupported answer claims while a trace is being generated. `scorer.py` scores the compact record that is actually eligible for SFT.

Each generated master record keeps separate fields:

- `quality_flag`: `pass` or `needs_review`
- `quality_score`: numeric score from 0 to 100
- `quality_profile`: native evaluator profile
- `quality_failed_checks`: native checks that did not pass
- `quality_issues`: hard answer-grounding issues

Evaluate every record in the current master trace with:

```powershell
python pipeclaw/backend/evaluate_teacher_trace.py
```

Evaluate a different file or one scenario with:

```powershell
python pipeclaw/backend/evaluate_teacher_trace.py `
  --teacher-trace path/to/teacher_trace.jsonl `
  --scenario-id scenario_pipeformer_prediction_003
```

The default pass threshold is 85. Critical execution or grounding checks still block a pass regardless of the weighted score.

For causal or disturbance-impact questions, `run_pipeformer_forecast` accepts `include_baseline_comparison=true`. It runs one unchanged baseline in addition to the disturbed forecast and returns only a compact `counterfactual_comparison`.
