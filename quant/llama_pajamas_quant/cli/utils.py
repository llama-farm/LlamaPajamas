"""Shared CLI utilities."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(verbose: bool = False, quiet: bool = False):
    """Setup logging configuration.

    Args:
        verbose: Enable verbose output
        quiet: Suppress non-error output
    """
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(message)s' if not verbose else '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )


def validate_path(path: Path, must_exist: bool = True, file_type: Optional[str] = None) -> Path:
    """Validate path exists and optionally check file type.

    Args:
        path: Path to validate
        must_exist: Whether path must exist
        file_type: Optional file extension to check (e.g., '.gguf', '.txt')

    Returns:
        Validated path

    Raises:
        ValueError: If validation fails
    """
    path = Path(path)

    if must_exist and not path.exists():
        raise ValueError(f"Path not found: {path}")

    if file_type and path.suffix != file_type:
        raise ValueError(f"Expected {file_type} file, got: {path.suffix}")

    return path


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable size.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted string (e.g., "4.68 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def print_section(title: str, char: str = "=", width: int = 60):
    """Print a section header.

    Args:
        title: Section title
        char: Character to use for border
        width: Width of border
    """
    print()
    print(char * width)
    print(title.upper())
    print(char * width)


def confirm_action(message: str, default: bool = False) -> bool:
    """Ask user to confirm action.

    Args:
        message: Confirmation message
        default: Default response if user just presses enter

    Returns:
        True if user confirmed
    """
    if default:
        prompt = f"{message} [Y/n]: "
    else:
        prompt = f"{message} [y/N]: "

    response = input(prompt).strip().lower()

    if not response:
        return default

    return response in ['y', 'yes']
