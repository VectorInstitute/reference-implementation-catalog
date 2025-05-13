#!/usr/bin/env python3
"""Sync repos from README.md to docs/index.md.

This script extracts the repositories list from README.md and
updates the cards in docs/index.md. It automatically groups implementations
by type and ensures changes in README.md are reflected in the documentation.
"""

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# Regular expressions to parse README.md table rows
REPO_ROW_PATTERN = re.compile(
    r"\| \[(.*?)\]\[(.*?-repo)\] \| (.*?) \| (.*?) \| (.*?) \| (.*?) \| (.*?) \| (\d{4}) \|"
)
TYPE_TAB_PATTERN = re.compile(r'=== "([^"]*)"')
BROWSE_HEADING_PATTERN = r"## Browse Implementations by Type"
# Pattern to match Markdown links [text][reference]
MARKDOWN_LINK_PATTERN = re.compile(r"\[(.*?)\]\[(.*?)\]")
# Pattern to match simple Markdown links [text]
SIMPLE_MARKDOWN_LINK_PATTERN = re.compile(r"\[(.*?)\]")


def format_datasets(datasets_text: str) -> str:
    """Format dataset text into a nicer representation.

    Parameters
    ----------
    datasets_text : str
        Raw dataset text from README.md

    Returns
    -------
    str
        Formatted dataset HTML

    """
    if not datasets_text or datasets_text == "-":
        return "<em>No public datasets available</em>"

    # If it contains numbers with text, e.g. "3 datasets including..."
    if re.match(r"^\d+\s+", datasets_text):
        count_match = re.match(r"^(\d+)\s+[^,]+", datasets_text)
        if count_match:
            count = count_match.group(1)
            # Remove the count part and format the rest
            main_text = (
                datasets_text.split(",", 1)[-1].strip() if "," in datasets_text else ""
            )
            if main_text:
                return f"{count} public datasets including {main_text}"
            return f"{count} public datasets"

    # Format markdown links to look like tags
    formatted_text = datasets_text

    # Clean up any malformed brackets first
    formatted_text = formatted_text.replace("][", "] [")

    # Replace markdown links with spans
    def replace_markdown_link(match):
        text = match.group(1)
        # Clean any remaining brackets that might break HTML
        text = text.replace("[", "").replace("]", "")
        return f'<span class="dataset-tag">{text}</span>'

    # First check for [text][reference] format
    formatted_text = MARKDOWN_LINK_PATTERN.sub(replace_markdown_link, formatted_text)

    # Then check for simple [text] format
    formatted_text = SIMPLE_MARKDOWN_LINK_PATTERN.sub(
        replace_markdown_link, formatted_text
    )

    # Clean up any remaining markdown formatting
    formatted_text = formatted_text.replace("[", "").replace("]", "")

    # Replace commas and <br> with proper separators
    formatted_text = formatted_text.replace(",", " ")
    formatted_text = formatted_text.replace("<br>", " ")

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

    return formatted_text


def parse_readme_table() -> Dict[str, List[Dict]]:
    """Parse the implementations table from README.md.

    Returns
    -------
    Dict[str, List[Dict]]
        Dictionary with types as keys and lists of implementation details as values.

    """
    readme_path = Path("README.md")
    if not readme_path.exists():
        raise FileNotFoundError(f"README.md not found at {readme_path.absolute()}")

    readme_content = readme_path.read_text(encoding="utf-8")

    # Extract all table rows with repository information
    implementations_by_type = defaultdict(list)

    for match in REPO_ROW_PATTERN.finditer(readme_content):
        (
            repo_name,
            repo_id,
            description,
            algorithms,
            datasets_count,
            public_datasets,
            type_value,
            year,
        ) = match.groups()

        # Extract algorithms as a list
        algo_list = []
        for algo_ in algorithms.split(","):
            algo = algo_.strip()
            if algo:
                algo_list.append(algo)

        implementations_by_type[type_value.strip()].append(
            {
                "name": repo_name,
                "repo_id": repo_id,
                "description": description.strip(),
                "algorithms": algo_list,
                "datasets_count": datasets_count.strip(),
                "public_datasets": public_datasets.strip(),
                "type": type_value.strip(),
                "year": year,
            }
        )

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
    algorithms = impl["algorithms"]
    tag_html = ""

    # Create algorithm tags (limit to 5 for display)
    display_algos = algorithms[:5]
    for algo_ in display_algos:
        algo = algo_.strip()
        if algo:
            tag_html += f'        <span class="tag" data-tippy="{algo}">{algo}</span>\n'

    # Format datasets
    formatted_datasets = format_datasets(impl["public_datasets"])

    # Create the card HTML with proper indentation
    return f"""    <div class="card" markdown>
    <div class="header">
        <h3><a href="https://github.com/VectorInstitute/{impl["repo_id"].replace("-repo", "")}" title="Go to Repository">{impl["name"]}</a></h3>
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
            new_content += generate_type_section(type_value, implementations_by_type[type_value])

    # Add any remaining types that weren't in the predefined order
    for type_value in sorted([t for t in implementations_by_type.keys() if t not in type_order]):
        new_content += generate_type_section(type_value, implementations_by_type[type_value])

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

    # Ensure we have CSS for dataset tags and type tags
    css_for_tags = """
<style>
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
</style>
"""

    # Check if the CSS is already in the document
    if "<style>" in original_content:
        # Replace the existing style block to include both dataset and type tags
        style_start = original_content.find("<style>")
        style_end = original_content.find("</style>", style_start) + len("</style>")
        original_content = original_content[:style_start] + css_for_tags + original_content[style_end:]
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
        "## Browse Implementations by Year",
        "## Browse Implementations by Type"
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
    """Run main function to sync README.md implementations to docs/index.md.

    This function orchestrates the entire synchronization process from README.md to docs/index.md.
    """
    print("Syncing implementations from README.md to docs/index.md...")
    implementations_by_type = parse_readme_table()

    if not implementations_by_type:
        print("No repositories found in README.md. Nothing to update.")
        return

    total_count = sum(len(impls) for impls in implementations_by_type.values())
    print(
        f"Found {total_count} repositories across {len(implementations_by_type)} types"
    )

    update_docs_index(implementations_by_type)
    print("Sync complete!")


if __name__ == "__main__":
    main()
