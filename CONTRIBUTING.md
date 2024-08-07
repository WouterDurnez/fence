# Contributing to [Your Project Name]

Thank you for considering contributing to [Your Project Name]! We welcome contributions from everyone. Below are some guidelines to help you get started.

## Table of Contents

1. [How Can I Contribute?](#how-can-i-contribute)
   - [Reporting Bugs](#reporting-bugs)
   - [Suggesting Features](#suggesting-features)
   - [Improving Documentation](#improving-documentation)
   - [Contributing Code](#contributing-code)
2. [Setting Up Your Development Environment](#setting-up-your-development-environment)
3. [Style Guides](#style-guides)
   - [Python Style Guide](#python-style-guide)
   - [Commit Messages](#commit-messages)
4. [Submitting Your Changes](#submitting-your-changes)
5. [Acknowledgements](#acknowledgements)

## How Can I Contribute?

### Reporting Bugs

If you find a bug, please create an issue on [GitHub Issues](https://github.com/WouterDurnez/fence/issues) and provide detailed information about the bug. Include steps to reproduce, expected behavior, and screenshots if applicable.

### Suggesting Features

If you have a suggestion for a new feature, please open an issue on [GitHub Issues](https://github.com/WouterDurnez/fence/issues) and describe the feature in detail, including its benefits and any potential drawbacks.

### Improving Documentation

We always appreciate improvements to our documentation. Feel free to suggest changes by opening an issue or directly submitting a pull request (PR) with your improvements.

### Contributing Code

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature-name`).
3. Make your changes.
4. Ensure your changes pass all tests and follow the [style guides](#style-guides).
5. Commit your changes (`git commit -m '✨ Add some feature'`).
6. Push to the branch (`git push origin feature/your-feature-name`).
7. Open a pull request on the original repository.

## Setting Up Your Development Environment

1. Fork and clone the repository:

   ```sh
   git clone https://github.com/WouterDurnez/fence.git
   ```

2. Install Poetry if you haven't already:

   ```sh
   pip install poetry
   ```

3. Install the dependencies:

   ```sh
   poetry install
   ```

4. Run the tests to make sure everything is working:

   ```sh
   poetry run pytest
   ```

## Style Guides

### Python Style Guide

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/).
- Use type hints as much as possible.
- Document your code thoroughly using docstrings.
- There's a pre-commit hook that will run `black` and `isort` on your code before you commit. You can also run it manually with `poetry run pre-commit run --all`.

### Commit Messages

- Use the present tense ("Add feature" not "Added feature").
- Capitalize the first letter.
- Limit the first line to 72 characters or less.
- Reference issues and pull requests liberally.
- [GitMoji](https://gitmoji.dev/) is encouraged for commit messages.

## Submitting Your Changes

1. Ensure your code follows the [style guides](#style-guides).
2. Ensure all tests pass.
3. Submit a pull request through the GitHub UI.

## Acknowledgements

Thank you to all the contributors who have helped make this project better!
