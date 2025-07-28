# Manual Setup Required

## GitHub Repository Configuration

### 1. Branch Protection Rules
```
Settings > Branches > Add rule for 'main'
- Require pull request reviews (1+ reviewers)
- Require status checks to pass
- Restrict pushes to main branch
```

### 2. Security Features
```
Settings > Security & analysis
- Enable Dependency graph
- Enable Dependabot alerts
- Enable Dependabot security updates
- Configure CodeQL analysis
```

### 3. GitHub Actions Workflows
Create these workflow files manually:
- `.github/workflows/ci.yml` - CI/CD pipeline
- `.github/workflows/security.yml` - Security scanning
- `.github/workflows/release.yml` - Release automation

### 4. Repository Settings
```
Settings > General
- Add description and topics
- Configure homepage URL
- Enable/disable features as needed
```

## Development Environment

### Pre-commit Setup
```bash
pip install pre-commit
pre-commit install
```

### Development Dependencies
```bash
npm install  # For documentation tooling
```

## Resources
- [Repository Setup Guide](https://docs.github.com/en/repositories/creating-and-managing-repositories/quickstart-for-repositories)
- [GitHub Actions Setup](https://docs.github.com/en/actions/quickstart)