# Contributing to xLSTM ECG Classification

Thank you for your interest in contributing to this project! We welcome contributions from the community.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- A clear, descriptive title
- Steps to reproduce the issue
- Expected vs. actual behavior
- Your environment (OS, Python version, library versions)
- Any relevant error messages or logs

### Suggesting Enhancements

Enhancement suggestions are welcome! Please open an issue describing:
- The feature you'd like to see
- Why it would be useful
- How it might work

### Pull Requests

1. **Fork the repository** and create your branch from `main`
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clean, readable code
   - Follow PEP 8 style guidelines
   - Add comments for complex logic
   - Update documentation as needed

3. **Test your changes**
   - Ensure existing tests pass
   - Add new tests for new functionality
   - Test with different ECG formats

4. **Commit your changes**
   ```bash
   git commit -m "Add: Brief description of changes"
   ```
   
   Commit message prefixes:
   - `Add:` New features
   - `Fix:` Bug fixes
   - `Update:` Updates to existing features
   - `Docs:` Documentation changes
   - `Refactor:` Code refactoring
   - `Test:` Adding or updating tests

5. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request**
   - Provide a clear description of changes
   - Reference any related issues
   - Explain the motivation for changes

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and concise
- Use type hints where appropriate

## Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/xLSTM-ECG-Signals-Multilabel-Classification.git
cd xLSTM-ECG-Signals-Multilabel-Classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Testing

Before submitting a PR:
- Test the training notebook with a small dataset
- Test the deployment app with sample data
- Verify all file formats work correctly
- Check that documentation is up to date

## Areas for Contribution

We especially welcome contributions in:

- **Model improvements**: New architectures, hyperparameter tuning
- **Data augmentation**: Advanced signal augmentation techniques
- **Preprocessing**: Additional filtering or normalization methods
- **Visualization**: Enhanced ECG plotting and analysis tools
- **Documentation**: Tutorials, examples, API documentation
- **Testing**: Unit tests, integration tests
- **Deployment**: Docker support, cloud deployment guides
- **Performance**: Optimization, inference speed improvements

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on what is best for the project
- Show empathy towards other contributors

## Questions?

Feel free to open an issue for:
- Questions about the codebase
- Clarification on how to contribute
- Discussion of potential features

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing to xLSTM ECG Classification! 🎉
