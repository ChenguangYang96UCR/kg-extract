# kg-extract

Convert structured fields from an NSF Awards CSV file into knowledge graph triples. The project uses a hybrid design:

1. Deterministic rules process structured CSV fields for stable and traceable results.
2. The `Abstract` is preserved as a source-text triple and can optionally be processed by KGGen or UIE.
3. The extractor produces a review-friendly CSV file and standards-compliant RDF N-Triples.

## Why not send the entire CSV to an LLM?

Fields such as `AwardNumber`, PI, organization, amount, and dates already contain structured facts. Asking an LLM to extract them again increases cost and introduces hallucination risk. A text extraction model is only needed for implicit relations in `Abstract`, such as methods, problems, and application domains.

Methodological references:

- Dimou et al. (2014), [RML: A Generic Language for Integrated RDF Mappings of Heterogeneous Data](https://ruben.verborgh.org/publications/dimou_ldow_2014/)
- W3C, [Generating RDF from Tabular Data on the Web](https://www.w3.org/TR/csv2rdf/)
- Mo et al. (2025), [KGGen: Extracting Knowledge Graphs from Plain Text with Language Models](https://arxiv.org/abs/2502.09956)

## Requirements

Python 3.10 or later. The core extractor has no third-party runtime dependencies.

```bash
python3 -m pip install -e .
```

Install an optional abstract extraction backend only when needed:

```bash
# Open-schema extraction with an API or local model
python -m pip install -e src/kg-gen

# Controlled-schema extraction with PaddleNLP
python -m pip install paddlepaddle
python -m pip install -e src/PaddleNLP
```

You can also run the project without installing it:

```bash
PYTHONPATH=src python3 -m kg_extract --help
```

## Usage

```bash
kg-extract /Users/chenguangyang/Downloads/Awards.csv --output-dir output
```

Without installation:

```bash
PYTHONPATH=src python3 -m kg_extract \
  /Users/chenguangyang/Downloads/Awards.csv \
  --output-dir output
```

The command generates:

- `output/triples.csv`
- `output/triples.nt`

When an Abstract backend is enabled, the command also writes Abstract-only debug outputs:

- `output/abstract_triples.csv`
- `output/abstract_triples.nt`

When keyword extraction is enabled, the command also writes keyword debug outputs:

- `output/keywords.csv`
- `output/keyword_triples.csv`
- `output/keyword_triples.nt`

Email addresses, phone numbers, and street addresses are excluded by default. Include them only when needed:

```bash
kg-extract /path/to/Awards.csv --output-dir output --include-contact
```

To change the generated entity URI prefix:

```bash
kg-extract /path/to/Awards.csv \
  --output-dir output \
  --base-uri https://example.org/nsf/
```

## Triple schema

Each CSV row is represented as an Award entity connected to reusable Person, Organization, and Program entities. For example:

```text
award/2541748  rdf:type                   schema:MonetaryGrant
award/2541748  schema:name                "CAREER: A Safety-Aware..."
award/2541748  kg:principalInvestigator   person/chenguang-wang
award/2541748  kg:recipientOrganization   organization/university-of-california-santa-cruz
award/2541748  kg:belongsToProgram        program/info-integration-informatics
award/2541748  schema:amount              "349875.00"^^xsd:decimal
```

Entity URIs use readable slugs without hash suffixes. Labels that normalize to the
same slug are therefore represented by the same graph node.

`triples.csv` also includes `award_number`, `source_column`, `evidence`, `confidence`, and `extractor`. These fields distinguish deterministic facts from model-generated relations and trace each triple back to its source.

## Tests

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python3 -m unittest discover -s tests -v
```

## Abstract relation extraction

The default backend is `none`; therefore, running the base command never invokes a model. Model-generated triples are added to the same CSV and N-Triples outputs when a backend is explicitly selected.

### KGGen

KGGen performs open-schema extraction. It supports API and local models through LiteLLM. The model name follows LiteLLM's `provider/model` convention.

```bash
export OPENAI_API_KEY=your-key

kg-extract /path/to/Awards.csv \
  --output-dir output-kggen \
  --abstract-backend kggen \
  --abstract-model openai/gpt-4o \
  --api-key-env OPENAI_API_KEY \
  --abstract-limit 5
```

Start with `--abstract-limit 5` to review quality and cost before processing all records. Remove the option to process every non-empty abstract. KGGen's built-in deduplication/clustering is disabled by default because some `kg-gen` and `semhash` version combinations are incompatible. Use `--kggen-deduplicate` only after confirming your installed KGGen dependencies support it. A local Ollama model can be selected without an API key:

```bash
kg-extract /path/to/Awards.csv \
  --output-dir output-kggen-local \
  --abstract-backend kggen \
  --abstract-model ollama_chat/deepseek-r1:14b \
  --abstract-limit 5
```

### UIE

PaddleNLP UIE performs controlled-schema extraction and returns span-level confidence values. The included schema is [config/uie_award_schema.json](config/uie_award_schema.json):

```text
develops
addresses problem
uses method
targets domain
evaluated in
has expected outcome
mitigates risk
```

Run UIE with the English model:

```bash
kg-extract /path/to/Awards.csv \
  --output-dir output-uie \
  --abstract-backend uie \
  --abstract-model uie-base-en \
  --uie-schema config/uie_award_schema.json \
  --min-confidence 0.50 
```


```bash
kg-extract /Users/chenguangyang/Desktop/ucr_work/kg-extract/data/Quantum/filtered/awards_start_2024.csv \
  --output-dir output-uie \
  --abstract-backend uie \
  --abstract-model uie-base-en \
  --uie-schema config/uie_award_schema.json \
  --min-confidence 0.50 

```

The first UIE run downloads model weights. A custom schema must use PaddleNLP's hierarchical JSON structure, for example:

```json
[
  {
    "Project": ["develops", "uses method", "evaluated in"]
  }
]
```

Relations whose subject is `this project`, `the project`, or `project` are grounded directly to the corresponding Award URI. Other extracted terms become Concept entities and are linked to the Award with `kg:mentions`.

Model output is not treated as ground truth. Review evidence and confidence, normalize entities, and evaluate a human-annotated sample before using generated relations downstream.

## Abstract keyword extraction

Keyword extraction links awards through reusable Keyword nodes. This is useful when two awards do not have an explicit extracted relation but share topics such as `digital tutor`, `machine learning`, or `cybersecurity education`.

The generated pattern is:

```text
award/2541748  kg:hasKeyword  keyword/digital-tutor
keyword/digital-tutor  rdf:type     kg:Keyword
keyword/digital-tutor  schema:name  "digital tutor"
```

The keyword stage removes NSF's repeated statutory-mission boilerplate before scoring candidates, because that sentence appears in many award abstracts and would otherwise create noisy shared keywords.

A dependency-free baseline is available for quick debugging:

```bash
kg-extract /path/to/Awards.csv \
  --output-dir output-keywords \
  --keyword-backend simple \
  --keyword-top-k 8 \
  --keyword-ngram-min 1 \
  --keyword-ngram-max 3
```

For better semantic keywords, install KeyBERT and run:

```bash
python3 -m pip install -e '.[keywords]'

kg-extract /path/to/Awards.csv \
  --output-dir output-keybert \
  --keyword-backend keybert \
  --keyword-model sentence-transformers/all-MiniLM-L6-v2 \
  --keyword-top-k 8 \
  --keyword-ngram-min 1 \
  --keyword-ngram-max 3
```

To merge near-duplicate keyword nodes, add embedding clustering:

```bash
kg-extract /path/to/Awards.csv \
  --output-dir output-keybert-clustered \
  --keyword-backend keybert \
  --keyword-model sentence-transformers/all-MiniLM-L6-v2 \
  --keyword-top-k 8 \
  --keyword-ngram-min 2 \
  --keyword-ngram-max 3 \
  --keyword-noun-filter \
  --keyword-cluster \
  --keyword-cluster-threshold 0.82
```

The helper script uses the same recommended settings and automatically places the
dataset name and year in the output folder when the second argument is omitted:

```bash
bash scripts/run_keybert_keywords.sh "data/Quantum/filtered/awards_start_2024.csv"
# writes to output-keybert-clean/quantum/2024

bash scripts/run_keybert_keywords.sh "data/digital twins/filtered/awards_start_2025_2026.csv"
# writes to output-keybert-clean/digital-twins/2025_2026
```

```bash
kg-extract 'data/Quantum/filtered/awards_start_2024.csv' \
  --output-dir output-keybert \
  --keyword-backend keybert \
  --keyword-model sentence-transformers/all-MiniLM-L6-v2 \
  --keyword-top-k 8 \
  --keyword-ngram-min 1 \
  --keyword-noun-filter \
  --keyword-cluster \
  --keyword-ngram-max 3

```


Keyword normalization performs lowercasing, punctuation removal, stopword filtering,
basic singularization, common acronym expansion, and slug-based URI generation.
`--keyword-noun-filter` additionally removes or trims verb-led candidates so phrases
such as `developing groundbreaking chips` become `groundbreaking chip`, while
relation-like phrases such as `led university michigan` are discarded.
Embedding clustering runs after this rule-based normalization so labels such as
`language model` and `large language model` can be mapped to one Keyword node when
their sentence-transformer embeddings are similar enough.

YAKE is also supported as a lightweight unsupervised backend:

```bash
python3 -m pip install -e '.[yake]'

kg-extract /path/to/Awards.csv \
  --output-dir output-yake \
  --keyword-backend yake \
  --keyword-top-k 8
```

### Local LLM keyword extraction

Local causal language models can be used as another keyword backend. This branch
only asks the model to propose candidate noun phrases; the shared keyword
normalization, noun filtering, clustering, and triple generation steps still run
afterward.

The default local model path is `/data/cyang314/kg/Qwen3-8B`:

```bash
python3 -m pip install -e '.[llm]'

kg-extract /path/to/Awards.csv \
  --output-dir output-llm-keywords \
  --keyword-backend llm \
  --keyword-llm-model-path /data/cyang314/kg/Qwen3-8B \
  --keyword-llm-repo-id Qwen/Qwen3-8B \
  --keyword-top-k 8 \
  --keyword-ngram-min 2 \
  --keyword-ngram-max 3 \
  --keyword-noun-filter \
  --keyword-cluster \
  --keyword-cluster-threshold 0.88
```

The LLM backend loads model weights with `local_files_only=True`, so the model
is always loaded from the local directory. If the directory is missing or
incomplete, the backend first downloads `Qwen/Qwen3-8B` from Hugging Face into
that directory. Use `--keyword-llm-no-download` to require a fully offline local
load.

Do not use `deepseek-ai/dspark_qwen3_8b_block7` as the standalone keyword LLM.
That repository is a DeepSpec/DSpark draft checkpoint for speculative decoding
and does not include the tokenizer or full target model needed for direct text
generation. Use the target model, such as `Qwen/Qwen3-8B`, for this backend.

### Ollama / LiteLLM keyword extraction

Ollama models can be used through LiteLLM. Start Ollama and pull the model first:

```bash
ollama pull deepseek-r1:14b
```

Then run:

```bash
python3 -m pip install -e '.[litellm]'

kg-extract /path/to/Awards.csv \
  --output-dir output-ollama-keywords \
  --keyword-backend litellm \
  --keyword-litellm-model ollama_chat/deepseek-r1:14b \
  --keyword-litellm-api-base http://localhost:11434 \
  --keyword-top-k 8 \
  --keyword-ngram-min 2 \
  --keyword-ngram-max 3 \
  --keyword-noun-filter \
  --keyword-cluster \
  --keyword-cluster-threshold 0.88
```

The helper script [scripts/run_llm_keywords.sh](scripts/run_llm_keywords.sh)
uses this Ollama/LiteLLM backend by default.

`keywords.csv` records `award_number`, raw `keyword`, `canonical_keyword`, `score`, `extractor`, and `evidence`. `keyword_triples.csv` and `keyword_triples.nt` contain only the triples produced by the keyword stage, while `triples.csv` and `triples.nt` include them together with the structured and optional Abstract relation triples.

## Privacy

The input CSV may contain personal contact details. The repository ignores `Awards.csv` and the generated `output/` directory. Contact triples are opt-in through `--include-contact`.
