#!/usr/bin/env python3
"""Sync repos from YAML files to docs/index.md.

This script reads repository information from YAML files in the repositories/ directory
and updates the cards in docs/index.md. It automatically groups implementations
by type and ensures changes in YAML files are reflected in the documentation.
"""

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

TYPE_TAB_PATTERN = re.compile(r'=== "([^"]*)"')
BROWSE_HEADING_PATTERN = r"## Browse Implementations by Type"
# Pattern to match Markdown links [text][reference]
MARKDOWN_LINK_PATTERN = re.compile(r"\[(.*?)\]\[(.*?)\]")
# Pattern to match simple Markdown links [text]
SIMPLE_MARKDOWN_LINK_PATTERN = re.compile(r"\[(.*?)\]")


def format_datasets(datasets_list) -> str:
    """Format dataset text into a nicer representation.

    Parameters
    ----------
    datasets_list : List
        List of dataset names or dataset objects from YAML file

    Returns
    -------
    str
        Formatted dataset HTML

    """
    if not datasets_list:
        return "<em>No public datasets available</em>"

    # Create dataset tags for each dataset in the list
    formatted_text = ""
    for dataset in datasets_list:
        # Handle both string datasets and dict datasets with name/url fields
        if isinstance(dataset, dict) and "name" in dataset:
            dataset_name = dataset["name"].strip()
            dataset_url = dataset.get("url", "")

            if dataset_url:
                formatted_text += f'<a href="{dataset_url}" class="dataset-tag" target="_blank">{dataset_name}</a>  '
            else:
                formatted_text += f'<span class="dataset-tag">{dataset_name}</span>  '
        else:
            dataset_text = str(dataset).strip()
            if dataset_text:
                formatted_text += f'<span class="dataset-tag">{dataset_text}</span>  '

    # Add a final check to ensure no broken spans
    if "<span" in formatted_text and formatted_text.count(
        "<span"
    ) != formatted_text.count("</span"):
        # Try to repair broken spans
        formatted_text = re.sub(
            r"<span([^>]*?)>([^<]*?)(?!<\/span>)(\s|$)",
            r"<span\1>\2</span>\3",
            formatted_text,
        )

    return formatted_text.rstrip()


def parse_yaml_repositories() -> Dict[str, List[Dict]]:
    """Parse the implementations from YAML files in repositories/ directory.

    Returns
    -------
    Dict[str, List[Dict]]
        Dictionary with types as keys and lists of implementation details as values.

    """
    repos_dir = Path("repositories")
    if not repos_dir.exists():
        raise FileNotFoundError(
            f"repositories/ directory not found at {repos_dir.absolute()}"
        )

    # Find all YAML files in the repositories directory
    yaml_files = list(repos_dir.glob("*.yaml")) + list(repos_dir.glob("*.yml"))

    if not yaml_files:
        raise FileNotFoundError(f"No YAML files found in {repos_dir.absolute()}")

    # Extract repository information from each YAML file
    implementations_by_type = defaultdict(list)

    for yaml_file in yaml_files:
        with open(yaml_file, "r", encoding="utf-8") as f:
            repo_data = yaml.safe_load(f)

        # Add the repository to the appropriate type list
        implementations_by_type[repo_data["type"]].append(repo_data)

    return implementations_by_type


def generate_card_html(impl: Dict) -> str:
    """Generate HTML for a single implementation card.

    Parameters
    ----------
    impl : Dict
        Dictionary containing implementation details

    Returns
    -------
    str
        HTML string for the implementation card

    """
    # Extract up to 5 algorithms to display as tags
    algorithms = impl.get("algorithms", [])
    tag_html = ""

    # Create algorithm tags (limit to 5 for display)
    display_algos = algorithms[:5]
    for algo in display_algos:
        algo_text = str(algo).strip()
        if algo_text:
            tag_html += f'        <span class="tag" data-tippy="{algo_text}">{algo_text}</span>\n'

    # Format datasets
    formatted_datasets = format_datasets(impl.get("public_datasets", []))

    # Get the repository URL - either from github_url or construct from repo_id
    if "github_url" in impl:
        repo_url = impl["github_url"]
    else:
        repo_id = impl["repo_id"].replace("-repo", "")
        repo_url = f"https://github.com/VectorInstitute/{repo_id}"

    # Add BibTeX citation button if available
    bibtex_html = ""
    if "bibtex" in impl:
        bibtex_id = impl["bibtex"]
        bibtex_html = f'<a href="#" class="bibtex-button" data-bibtex-id="{bibtex_id}" title="View Citation">Cite</a>'

    # Add paper link if available
    paper_html = ""
    if "paper_url" in impl:
        paper_url = impl["paper_url"]
        paper_html = f'<a href="{paper_url}" class="paper-link" title="View Paper" target="_blank">Paper</a>'

    # Combine citation links
    citation_html = ""
    if bibtex_html or paper_html:
        citation_html = f"""    <div class="citation-links">
        {bibtex_html}
        {paper_html}
    </div>"""

    # Create the card HTML with proper indentation
    return f"""    <div class="card" markdown>
    <div class="header">
        <h3><a href="{repo_url}" title="Go to Repository">{impl["name"]}</a></h3>
        <span class="tag year-tag">{impl["year"]}</span>
        <span class="tag type-tag">{impl["type"]}</span>
    </div>
    <p>{impl["description"]}</p>
    <div class="tag-container">
{tag_html.rstrip() if tag_html else "        <!-- No tags available -->"}
    </div>
    <div class="datasets">
        <strong>Datasets:</strong> {formatted_datasets}
    </div>
{citation_html if citation_html else ""}
    </div>
"""


def get_type_sections(content: str) -> Dict[str, Tuple[int, int, str]]:
    """Extract type sections from the markdown content.

    Parameters
    ----------
    content : str
        The markdown content

    Returns
    -------
    Dict[str, Tuple[int, int, str]]
        Dictionary mapping types to their (start_pos, end_pos, section_content)

    """
    type_sections = {}
    type_matches = list(TYPE_TAB_PATTERN.finditer(content))

    for i, match in enumerate(type_matches):
        type_value = match.group(1)
        section_start = match.start()

        # Find the end of this section
        if i + 1 < len(type_matches):
            section_end = type_matches[i + 1].start()
        else:
            section_end = len(content)

        section_content = content[section_start:section_end]
        type_sections[type_value] = (section_start, section_end, section_content)

    return type_sections


def generate_type_section(type_value: str, implementations: List[Dict]) -> str:
    """Generate a complete type section with all implementations.

    Parameters
    ----------
    type_value : str
        The type for this section
    implementations : List[Dict]
        List of implementation details

    Returns
    -------
    str
        Formatted type section

    """
    section = f'=== "{type_value}"\n\n    <div class="grid cards" markdown>\n'

    for impl in implementations:
        section += generate_card_html(impl)

    section += "\n    </div>\n\n"
    return section


def rebuild_document(
    original_content: str, implementations_by_type: Dict[str, List[Dict]]
) -> str:
    """Completely rebuild the document with all type sections.

    Parameters
    ----------
    original_content : str
        The original markdown content
    implementations_by_type : Dict[str, List[Dict]]
        Dictionary with implementations grouped by type

    Returns
    -------
    str
        Updated markdown content

    """
    # Find the position after the "Browse Implementations by Type" heading
    heading_match = re.search(BROWSE_HEADING_PATTERN, original_content)
    if not heading_match:
        raise ValueError("Could not find 'Browse Implementations by Type' heading")

    heading_end = heading_match.end()

    # Get content before the first type section
    pre_content = original_content[:heading_end]

    # Find the first type section
    type_match = re.search(TYPE_TAB_PATTERN, original_content)
    if type_match:
        pre_content = original_content[: type_match.start()]

    # Build the new content with the pre-content
    new_content = pre_content + "\n\n"

    # Define the desired type order
    type_order = ["bootcamp", "tool", "applied-research"]

    # Add sections in the specified order, followed by any other types alphabetically
    for type_value in type_order:
        if type_value in implementations_by_type:
            new_content += generate_type_section(
                type_value, implementations_by_type[type_value]
            )

    # Add any remaining types that weren't in the predefined order
    for type_value in sorted(
        [t for t in implementations_by_type if t not in type_order]
    ):
        new_content += generate_type_section(
            type_value, implementations_by_type[type_value]
        )

    return new_content


def update_docs_index(implementations_by_type: Dict[str, List[Dict]]) -> None:
    """Update the docs/index.md file with cards for all implementations.

    Parameters
    ----------
    implementations_by_type : Dict[str, List[Dict]]
        Dictionary of implementations grouped by type

    """
    docs_index_path = Path("docs/index.md")
    if not docs_index_path.exists():
        raise FileNotFoundError(
            f"docs/index.md not found at {docs_index_path.absolute()}"
        )

    original_content = docs_index_path.read_text(encoding="utf-8")

    # Ensure we have CSS for dataset tags, type tags, year tags, and hero section
    css_for_tags = """
<style>
.hero-section {
  position: relative;
  padding: 5rem 4rem;
  text-align: center;
  color: white;
  background-color: var(--md-primary-fg-color);
  background-image: linear-gradient(rgba(0, 0, 0, 0.35), rgba(0, 0, 0, 0.35)), url('assets/splash.png');
  background-size: cover;
  background-position: center;
  display: flex;
  justify-content: center;
  align-items: center;
  margin: 0;
  padding: 0;
  width: 100%;
  position: relative;
  min-height: 70vh;
}

.hero-content {
  max-width: 800px;
  z-index: 10;
}

.hero-content h1 {
  font-size: 3rem;
  margin-bottom: 1rem;
  text-shadow: 0 2px 8px rgba(0,0,0,0.7);
  font-weight: 600;
  letter-spacing: 0.5px;
  color: #ffffff;
  font-family: 'Roboto', sans-serif;
}

.hero-content p {
  font-size: 1.5rem;
  margin-bottom: 2rem;
  text-shadow: 0 2px 6px rgba(0,0,0,0.7);
  max-width: 700px;
  margin-left: auto;
  margin-right: auto;
  line-height: 1.4;
  color: #f8f8f8;
  font-family: 'Roboto', sans-serif;
  font-weight: 300;
}

.card {
  box-shadow: 0 3px 6px rgba(0, 0, 0, 0.12) !important;
  border-left: 3px solid var(--md-accent-fg-color) !important;
  background-image: linear-gradient(to bottom right, rgba(255, 255, 255, 0.05), rgba(72, 192, 217, 0.05)) !important;
  transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}

.card:hover {
  transform: translateY(-3px);
  box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
  border-left: 3px solid #48c0d9 !important;
}

.dataset-tag {
  display: inline-block;
  background-color: #6a5acd;
  color: white;
  padding: 0.1rem 0.4rem;
  border-radius: 0.8rem;
  margin-right: 0.2rem;
  margin-bottom: 0.2rem;
  font-size: 0.7rem;
  font-weight: 500;
  white-space: nowrap;
}

.type-tag {
  display: inline-block;
  background-color: #2e8b57;
  color: white;
  padding: 0.1rem 0.4rem;
  border-radius: 0.8rem;
  margin-right: 0.2rem;
  margin-bottom: 0.2rem;
  font-size: 0.7rem;
  font-weight: 500;
  white-space: nowrap;
}

.year-tag {
  background-color: #48c0d9; /* Vector teal accent color */
  color: white;
  float: right;
  font-weight: 600;
}

.citation-links {
  margin-top: 0.75rem;
  display: flex;
  gap: 0.75rem;
}

.citation-links a {
  display: inline-block;
  padding: 0.25rem 0.5rem;
  background-color: #f0f0f0;
  border-radius: 4px;
  font-size: 0.8rem;
  text-decoration: none;
  color: #333;
  transition: background-color 0.2s;
}

.citation-links a:hover {
  background-color: #e0e0e0;
}
</style>
"""

    # Check if the CSS is already in the document
    if "<style>" in original_content:
        # No need to replace CSS as it's already there
        pass
    else:
        # Add the CSS after the header area
        header_end_marker = "</div>"
        header_section_end = original_content.find(
            header_end_marker, original_content.find("catalog-stats")
        )

        if header_section_end > 0:
            original_content = (
                original_content[: header_section_end + len(header_end_marker)]
                + css_for_tags
                + original_content[header_section_end + len(header_end_marker) :]
            )

    # Update the heading from "by Year" to "by Type"
    original_content = original_content.replace(
        "## Browse Implementations by Year", "## Browse Implementations by Type"
    )

    # Create an entirely new document
    updated_content = rebuild_document(original_content, implementations_by_type)

    # Write the updated content back to docs/index.md
    docs_index_path.write_text(updated_content, encoding="utf-8")

    # Get existing and new types for reporting
    existing_sections = set(get_type_sections(original_content).keys())
    new_types = set(implementations_by_type.keys()) - existing_sections

    # Print summary
    print(
        f"Updated {docs_index_path} with {sum(len(impls) for impls in implementations_by_type.values())} repositories"
    )
    if new_types:
        print(
            f"Added {len(new_types)} new type sections: {', '.join(sorted(new_types))}"
        )


def main() -> None:
    """Run main function to sync YAML repositories to docs/index.md.

    This function orchestrates the entire synchronization process from YAML files to docs/index.md.
    """
    print("Syncing implementations from YAML files to docs/index.md...")
    implementations_by_type = parse_yaml_repositories()

    if not implementations_by_type:
        print("No repositories found in YAML files. Nothing to update.")
        return

    total_count = sum(len(impls) for impls in implementations_by_type.values())
    print(
        f"Found {total_count} repositories across {len(implementations_by_type)} types"
    )

    update_docs_index(implementations_by_type)
    print("Sync complete!")


if __name__ == "__main__":
    main()
