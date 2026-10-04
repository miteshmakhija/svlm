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

{results_section}

### Gate checks

{gate_table}

## Training data

{data_table}

Decontamination: {decontam}

Teacher output filter: {teacher_filter}

## Serving

| Artefact | File | Size |
|---|---|---|
{artefact_rows}

{serving_notes}

## Lineage

| | |
|---|---|
| Git commit | `{git_commit}` |
| Config hash | `{config_hash}` |
| Pipeline run | `{run_dir}` |
| Data hashes | {data_hashes} |

## Limitations

{limitations}
