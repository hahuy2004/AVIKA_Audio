# Enriched Text-Video Retrieval Evaluation

Implementation of enriched query evaluation for text-video retrieval based on ICLR 2025 paper "Bridging Information Asymmetry in Text-Video Retrieval: A Data-Centric Approach".

## Overview

This module implements a two-phase evaluation approach:

### Phase 1: Offline Pre-processing (Query Generation)
Generate enriched query variations using GPT-4 LLM.

### Phase 2: Online Evaluation
- **FQS (Farthest Query Selection)**: Select k most diverse queries
- **Majority Voting**: Aggregate results using Reciprocal Rank Fusion

## File Structure

```
enriched_eval/
├── __init__.py                 # Module initialization
├── query_generator.py          # GPT-4 query generation (Phase 1)
├── fqs_selector.py            # Farthest Query Selection algorithm
├── voting_aggregator.py       # Majority Voting with RRF/Borda Count
└── enriched_dataloader.py     # Data loader for enriched queries
```

## Usage

### Step 1: Generate Enriched Queries (Offline)

First, prepare your test set captions in JSON format:
```json
{
  "video_0": "a man is playing guitar",
  "video_1": "a woman is cooking in kitchen",
  ...
}
```

Then run the query generator:

```bash
python generate_enriched_queries.py \
  --input_json data/test_captions.json \
  --output_json data/enriched_queries.json \
  --api_key "your-openai-api-key" \
  --n_variations 10 \
  --model gpt-4
```

**API Key**: Use your OpenAI API key: `sk-proj-...`

This generates a JSON file with 11 sentences per video (1 original + 10 variations).

### Step 2: Run Enriched Evaluation

#### Normal Evaluation (eval_mode=1)
```bash
python main_task_retrieval.py \
  --do_eval \
  --eval_mode 1 \
  --datatype msrvtt \
  --init_model checkpoints/model.bin \
  --output_dir results/
```

#### Enriched Evaluation (eval_mode=2)
```bash
python main_task_retrieval.py \
  --do_eval \
  --eval_mode 2 \
  --enriched_queries_path data/enriched_queries.json \
  --fqs_k 2 \
  --voting_top_k 100 \
  --voting_method rrf \
  --datatype msrvtt \
  --init_model checkpoints/model.bin \
  --output_dir results/
```

## Parameters

### Enriched Evaluation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--eval_mode` | 1 | Evaluation mode: 1=normal, 2=enriched |
| `--enriched_queries_path` | None | Path to enriched queries JSON file |
| `--fqs_k` | 2 | Number of enriched queries to select (total = k+1 with original) |
| `--voting_top_k` | 100 | Top-K candidates for voting aggregation |
| `--voting_method` | 'rrf' | Voting method: 'rrf' (Reciprocal Rank Fusion) or 'borda' |

## Algorithm Details

### 1. Farthest Query Selection (FQS)

Selects k most diverse queries from enriched set:

```
1. Initialize: S = {Q_original}
2. Loop k times:
   - Find query Q with maximum distance to all queries in S
   - Add Q to S
3. Return: S (contains k+1 queries)
```

### 2. Majority Voting Aggregation

Combines retrieval results using Reciprocal Rank Fusion:

```
For each candidate video v:
  RRF_score(v) = Σ 1/(k + rank_i(v))
  where rank_i(v) is the rank of v in the i-th query's results
```

## Example Workflow

```python
# 1. Generate enriched queries (offline)
from enriched_eval.query_generator import generate_enriched_queries

enriched_data = generate_enriched_queries(
    input_captions={"video_0": "a man playing guitar"},
    output_json_path="enriched_queries.json",
    api_key="your-api-key",
    n_variations=10
)

# 2. Use in evaluation (automatically handled by main_task_retrieval.py)
# Just set eval_mode=2 and provide enriched_queries_path
```

## Testing Individual Modules

### Test FQS Algorithm
```bash
cd enriched_eval
python fqs_selector.py
```

### Test Voting Aggregation
```bash
cd enriched_eval
python voting_aggregator.py
```

## Notes

- **API Rate Limits**: The query generator includes sleep time between API calls to avoid rate limits
- **Resumable**: If generation is interrupted, it will resume from where it stopped
- **Fallback**: If enriched data is missing for a video, the system uses the original query
- **GPU Memory**: Enriched evaluation requires more memory due to multiple query embeddings

## Citation

If you use this implementation, please cite:

```bibtex
@inproceedings{iclr2025_enriched_retrieval,
  title={Bridging Information Asymmetry in Text-Video Retrieval: A Data-Centric Approach},
  booktitle={ICLR},
  year={2025}
}
```

## Troubleshooting

### Issue: "No enriched data for video_X"
**Solution**: Ensure your enriched_queries.json contains entries for all test videos.

### Issue: Out of memory during enriched evaluation
**Solution**: Reduce batch_size_val or voting_top_k parameter.

### Issue: OpenAI API rate limit
**Solution**: Increase sleep_time in query_generator.py or use a higher tier API plan.
