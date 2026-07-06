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
award/2541748  kg:principalInvestigator   person/chenguang-wang-...
award/2541748  kg:recipientOrganization   organization/university-of-california-santa-cruz-...
award/2541748  kg:belongsToProgram        program/info-integration-informatics-...
award/2541748  schema:amount              "349875.00"^^xsd:decimal
```

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
  --min-confidence 0.70 \
  --abstract-limit 5
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

## Privacy

The input CSV may contain personal contact details. The repository ignores `Awards.csv` and the generated `output/` directory. Contact triples are opt-in through `--include-contact`.
