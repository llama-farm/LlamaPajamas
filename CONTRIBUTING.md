# Contributing to LlamaPajamas

Thank you for your interest in contributing to LlamaPajamas! This document provides guidelines for contributing to the project.

## Code of Conduct

Be respectful, inclusive, and constructive in all interactions.

## How to Contribute

### Reporting Issues

- Check existing issues before creating a new one
- Provide detailed reproduction steps
- Include environment details (OS, Python version, hardware)
- Share error messages and logs

### Suggesting Features

- Open an issue with the "enhancement" label
- Describe the use case and expected behavior
- Consider implementation complexity and compatibility

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes**
4. **Test thoroughly**: Run existing tests and add new ones
5. **Commit**: Use clear, descriptive commit messages
6. **Push**: `git push origin feature/your-feature-name`
7. **Create PR**: Open a pull request with a detailed description

### Code Style

- Follow PEP 8 for Python code
- Use type hints where applicable
- Add docstrings for functions and classes
- Keep functions focused and modular

### Testing

```bash
# Run tests
cd quant
uv run pytest

# Run specific test
uv run pytest tests/test_quantize.py
```

### Documentation

- Update README.md if adding user-facing features
- Add examples for new functionality
- Update DEPLOYMENT.md for deployment-related changes

## Development Setup

```bash
# Clone and setup
git clone https://github.com/llama-farm/LlamaPajamas.git
cd LlamaPajamas

# Install quantization pipeline
cd quant
uv sync

# Install runtimes (as needed)
cd ../run && uv sync
cd ../run-coreml && uv sync
```

## Areas for Contribution

### High Priority
- [ ] Additional model architectures support
- [ ] Performance optimizations
- [ ] More comprehensive tests
- [ ] Documentation improvements

### Good First Issues
- [ ] Add example scripts
- [ ] Improve error messages
- [ ] Add logging
- [ ] Fix typos/formatting

### Advanced
- [ ] New quantization methods
- [ ] Additional backend support
- [ ] Distributed quantization
- [ ] Model optimization techniques

## Questions?

Open an issue with the "question" label.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
