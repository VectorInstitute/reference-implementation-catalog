# Contributing to Reference Implementation Catalog

Thank you for your interest in contributing to the Vector Institute Reference Implementation Catalog!

## Adding a New Implementation

The catalog is automatically synchronized between the README.md and the documentation site. If you are a Vector researcher or engineer and would like to add your implementation to this catalog, please follow these steps:

1. **Add your implementation to the table in README.md** following the existing format
2. Make sure to include:
   - Repository link
   - Short description
   - Tags for relevant technologies/algorithms
   - Information about datasets used
   - Year of publication

The documentation site (docs/index.md) will be automatically updated when your changes to README.md are merged.

## Automatic Synchronization

When you update README.md, a GitHub Actions workflow (`sync-readme.yml`) automatically:

1. Extracts the reference implementation table from README.md
2. Updates the cards in docs/index.md to match
3. Commits the changes to the repository

This ensures the documentation website always stays in sync with the README.md content.

## Manual Synchronization

If needed, you can manually run the synchronization script:

```bash
python scripts/sync_readme_to_docs.py
```

## Submitting Changes

To submit changes to this catalog:

1. Fork the repository
2. Create a feature branch
3. Make your changes to README.md
4. Submit a pull request

When submitting a PR, please fill out the PR template. If the PR fixes an issue, don't forget to link the PR to the issue!

## Issue Templates

We have templates available for:
- Bug reports
- Feature requests

Please use these templates when opening new issues to ensure all necessary information is provided.

## Building Documentation Locally

To build and preview the documentation locally:

```bash
# Install dependencies using uv
uv sync --group docs

# Serve the documentation
uv run mkdocs serve
```

Then visit http://localhost:8000 in your browser.

## Additional Information

For questions about contributing to the Reference Implementation Catalog, please reach out to the AI Engineering team at Vector Institute.
