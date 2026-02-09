from pathlib import Path

import pytest

# These imports are part of the jupyter ecosystem
nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")


# Get the path to the examples directory
EXAMPLES_DIR = Path(__file__).parent.parent / "docs/examples"


def get_notebook_paths():
    """Discover all notebook files in the examples directory."""
    if not EXAMPLES_DIR.exists():
        return []
    return sorted(EXAMPLES_DIR.glob("*.ipynb"))


def get_notebook_ids():
    """Get notebook names for test IDs."""
    return [nb.stem for nb in get_notebook_paths()]


@pytest.mark.notebooks
@pytest.mark.parametrize(
    "notebook_path",
    get_notebook_paths(),
    ids=get_notebook_ids(),
)
def test_notebook_execution(notebook_path):
    """Test that a notebook executes without errors.

    Parameters
    ----------
    notebook_path : Path
        Path to the notebook file to test.
    """
    # Read the notebook
    with open(notebook_path, encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)

    # Create a client to execute the notebook
    # Set timeout to 600 seconds (10 minutes) per cell for complex computations
    client = nbclient.NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(notebook_path.parent)}},
    )

    # Execute the notebook - this will raise an exception if any cell fails
    try:
        client.execute()
    except nbclient.exceptions.CellExecutionError as e:
        pytest.fail(f"Notebook {notebook_path.name} failed during execution:\n{e}")
