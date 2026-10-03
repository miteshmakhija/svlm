# {name} {tag}

| | |
|---|---|
| Owner | {owner} |
| Tier | {tier} |
| Task | {task} |
| Base model (student) | `{student}` |
| Teacher | `{teacher}` |
| Distillation | {method} |
| Release gate | **{verdict}** |
| Registered | {registered_at} |

## Intended use

{intended_use}

**Out of scope:** {out_of_scope}

## Results (held-out set, n = {n_heldout})

| Model | Exact match | Edit similarity | Parse rate |
|---|---|---|---|
| Teacher | {teacher_em} | {teacher_es} | {teacher_parse} |
| Student, untrained | {base_em} | {base_es} | {base_parse} |
| Student, SFT | {sft_em} | {sft_es} | {sft_parse} |
| **Student, final** | **{final_em}** | **{final_es}** | **{final_parse}** |

Latency (batch 1, {latency_device}): time to first token p50 {ttft_p50} ms, p90 {ttft_p90} ms.

HumanEval+: {humaneval}

### Gate checks

{gate_table}

## Training data

{data_table}

Decontamination: {decontam}

Teacher completions on train spans: {teacher_filter}

## Serving

| Artefact | File | Size |
|---|---|---|
{artefact_rows}

Recommended settings: temperature 0, max new tokens 64, stop on `<|endoftext|>` and FIM tokens.

## Lineage

| | |
|---|---|
| Git commit | `{git_commit}` |
| Config hash | `{config_hash}` |
| Pipeline run | `{run_dir}` |
| Data hashes | {data_hashes} |

## Limitations

{limitations}
