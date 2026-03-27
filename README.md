# Track A - LLM Sentiment Benchmarking Study

## Hypothesis
RoBERTa (tweet-trained) will outperform distilBERT and VADER on social media sentiment
because domain alignment matters more than model size.

## Models Benchmarked
- VADER (rule-based baseline)
- distilBERT (SST-2 fine-tuned)
- RoBERTa (Twitter fine-tuned)
- LLaMA 3.1 8B (zero-shot via Groq)
- Gemini 1.5 Flash (zero-shot via Google)

## Dataset
Sentiment140 - 1.6M tweets, stratified 10K sample (5K positive, 5K negative)

## Results
Current `results/results.csv` values:

| model | accuracy | precision | recall | f1 | latency_seconds | cost_per_1k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| VADER | 0.6530 | 0.6646 | 0.6178 | 0.6403 | 2.2427 | 0.0 |
| distilBERT | 0.7093 | 0.7568 | 0.6168 | 0.6797 | 246.9646 | 0.0 |
| RoBERTa | 0.7103 | 0.7775 | 0.5892 | 0.6704 | 499.3073 | 0.0 |

LLaMA and Gemini rows will appear here after `GROQ_API_KEY` and `GOOGLE_API_KEY` are set and `benchmark.py` is rerun.

## Key Finding
Among the completed runs so far, distilBERT achieved the best F1 score, so the hypothesis is not yet confirmed even though RoBERTa slightly led on accuracy.

## How to Run
```bash
pip install -r requirements.txt
python benchmark.py
python visualize.py
```

## Environment Variables Required
GROQ_API_KEY - from console.groq.com
GOOGLE_API_KEY - from aistudio.google.com
HF_TOKEN - optional, for HuggingFace authenticated access
