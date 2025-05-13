# Implementation Catalog Scripts

This directory contains automation scripts for maintaining the Vector Institute Implementation Catalog.

## Sync README to Docs

The `sync_readme_to_docs.py` script automates the process of keeping the documentation site in sync with the implementations listed in the main README.md file.

### How It Works

The script:

1. Parses the implementation table from README.md
2. Extracts all implementation details including:
   - Repository name and link
   - Description
   - Algorithms/techniques used
   - Dataset information
   - Year of publication
3. Generates HTML cards for each implementation grouped by type
4. Updates the docs/index.md file with the generated cards
5. Preserves the overall structure of docs/index.md

### Usage

To manually run the script:

```bash
python scripts/sync_readme_to_docs.py
```

### GitHub Actions Integration

This script is automatically executed by the GitHub Actions workflow whenever changes are made to README.md. The workflow:

1. Runs the sync script
2. Checks if there are any changes to docs/index.md
3. Commits the changes back to the repository if needed
4. For pull requests, it adds a comment indicating that the docs will be updated when merged

See the workflow configuration in `.github/workflows/sync_readme.yml` for details.

## Adding New Scripts

When adding new automation scripts to this directory:

1. Create a well-documented Python script
2. Update this README.md to document the script
3. Make sure to add appropriate tests
4. If needed, create a GitHub Actions workflow to run the script automatically
