# Private source material

The repository does not redistribute copyrighted books or credentials.

## The Finders

The local research workspace contains a privately supplied copy named
`assets/The Finders PDF.pdf` (Jeffery A. Martin, *The Finders*, 431 pages). It is
excluded from Git. Researchers who are lawfully entitled to use the book may put
their own copy at that exact path. Code and annotations must never depend on page
images; derived quotations must be short, cited, and independently auditable.

Before using a local copy, record its SHA-256 in the run manifest without
committing the file itself:

```bash
shasum -a 256 "assets/The Finders PDF.pdf"
```

The freely distributed companion PDF may be obtained from its official public
URL and is retained under `output/pdf/` for provenance:

`https://s3.amazonaws.com/kajabi-storefronts-production/sites/18817/themes/689414/downloads/nTFv42aaSXiI6UI6JU3b_How-To-Reach-Fundamental-Wellbeing-v2.pdf`

## Secrets

API keys belong in `.env`, the cluster secret store, or a private environment
module. `.env` and `.streamlit/secrets.toml` are excluded from Git. Never put a
token in an sbatch file, log, manifest, or command-line argument.
