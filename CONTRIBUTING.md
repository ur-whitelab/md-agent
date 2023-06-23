# Contributor's Guide

Welcome to the md-agent repository! We're thrilled that you're interested in contributing. This guide will help you understand how you can contribute to the project effectively.

## Setup Instructions
To get started with contributing to md-agent, follow these steps:

### Setting up the Repository and Environment
```
git clone https://github.com/ur-whitelab/md-agent.git
cd md-agent
conda env create -n mdagent -f environment.yaml
conda activate mdagent
conda install -c conda-forge openmm pdbfixer
```

### Installing the Package and Dependencies and Configuring Pre-commit Hooks
```
pip install -e .
pip install -r dev-requirements.txt
pre-commit install
```

## Code Guidelines

- Follow the [programming language] style guide for code formatting.
- Use meaningful variable and function names.
- Maintain consistency with the existing codebase.
- Write clear and concise comments to explain your code.
- Run pre-commit before committing your changes to ensure code quality and pass the automated checks.

## Feature Development Guidelines

When developing new features for md-agent, please follow these guidelines:

- If your feature uses a new package, ensure that you add the package to the project's setup and also include it in the ignore list in the `mypy.ini` file to avoid type checking errors.

- If your feature requires the use of API keys or other sensitive information, follow the appropriate steps:
  - Add a placeholder or example entry for the required information in the `.env.example` file.
  - Open an issue to discuss and coordinate the addition of the actual keys or secrets to the project's secure environment.

- New features should include the following components:
  - Implement the feature functionality in the codebase.
  - Write unit test functions to ensure proper functionality.
  - If applicable, create a notebook demonstration or example showcasing the usage and benefits of the new feature.

These guidelines help maintain consistency, security, and thoroughness in the development of new features for the project.


## Pull Request Process

1. Fork the repository to your own GitHub account.
2. Create a new branch: `git checkout -b my-feature-branch`
3. Make your changes, following the code guidelines mentioned above.
4. Test your changes thoroughly.
5. Commit your changes: `git commit -am 'Add new feature'`
6. Push your branch to your forked repository: `git push origin my-feature-branch`
7. Submit a pull request, providing a detailed description of your changes and their purpose.
8. Request a review from the project maintainers or assigned reviewers.
9. Address any feedback or comments received during the review process.
10. Once your changes are approved, they will be merged into the main branch.


## Issue Guidelines

If you encounter any bugs or have feature requests, please follow these guidelines when submitting an issue:

1. Before creating a new issue, search the existing issues to avoid duplicates.
2. Use a clear and descriptive title.
3. Provide steps to reproduce the issue (if applicable).
4. Include any relevant error messages or logs.
5. Explain the expected behavior and the actual behavior you encountered.

In addition, if you have any questions, need help, or want to discuss ideas, please submit an issue.


## Code Review Etiquette

When participating in code reviews, please follow these guidelines:


1. Be respectful and constructive in your feedback.
2. Provide specific and actionable feedback.
3. Explain the reasoning behind your suggestions.
4. Be open to receiving feedback and engage in discussions.
5. Be responsive and timely in addressing comments.

By adhering to these guidelines, we can foster a positive and collaborative environment for code reviews and contribute to the project's success.


## Acknowledgment

We value and appreciate all contributions to md-agent. Your efforts are highly valued and have a positive impact on the project's development.
