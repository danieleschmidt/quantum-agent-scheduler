# Manual Setup Required

This document outlines the manual setup steps required to complete the SDLC implementation due to GitHub App permission limitations.

## 🚨 Required GitHub Actions Setup

The Terragon autonomous SDLC implementation has created comprehensive workflow templates, but GitHub App permissions prevent automatic workflow creation. Repository maintainers must manually create these workflow files.

## 📋 Required Actions

### 1. Create GitHub Actions Workflows

Copy the workflow templates from `docs/workflows/templates/` to `.github/workflows/`:

```bash
# Create workflows directory
mkdir -p .github/workflows

# Copy all workflow templates
cp docs/workflows/templates/*.yml .github/workflows/

# Verify workflows are in place
ls -la .github/workflows/
```

### Required Workflow Files:

#### Core CI/CD Workflows
- **`ci.yml`** - Pull request validation, testing, security scanning
- **`release.yml`** - Automated releases with semantic versioning
- **`docs.yml`** - Documentation building and deployment

#### Security Workflows  
- **`security.yml`** - Comprehensive security scanning (SAST, DAST, dependency scan)
- **`container-security.yml`** - Container image vulnerability scanning

#### Specialized Workflows
- **`quantum-integration.yml`** - Quantum backend integration testing
- **`performance.yml`** - Performance benchmarking and regression testing
- **`health-check.yml`** - Production health monitoring

### 2. Configure Repository Settings

#### Branch Protection Rules
Enable branch protection for main branch via GitHub UI:
- Require pull request reviews (2 reviewers)
- Require status checks to pass before merging
- Require branches to be up to date before merging
- Restrict pushes to main branch
- Include administrators in restrictions

#### Repository Secrets
Configure the following secrets in repository settings:

**Quantum Provider Credentials:**
```
AWS_ACCESS_KEY_ID          # AWS Braket access
AWS_SECRET_ACCESS_KEY      # AWS Braket secret
QISKIT_IBM_TOKEN          # IBM Quantum token
DWAVE_API_TOKEN           # D-Wave API token
```

**Container Registry:**
```
REGISTRY_USERNAME         # Container registry username
REGISTRY_PASSWORD         # Container registry password/token
```

**Security Scanning:**
```
SNYK_TOKEN               # Snyk security scanning
SONAR_TOKEN              # SonarCloud code quality
```

### 3. Environment Configuration

#### Development Environment
```bash
# Copy environment template
cp .env.example .env

# Essential variables to configure:
# - Database connection string
# - Quantum provider credentials
# - API keys and secrets
```

### 4. Validation Steps

#### Verify CI/CD Pipeline
```bash
# Create test branch and PR
git checkout -b test-setup
git commit --allow-empty -m "test: verify CI/CD pipeline"
git push origin test-setup
```

#### Test Quantum Backends
```bash
# Run quantum backend validation
npm run quantum-test
```

## 🚀 Going Live Checklist

- [ ] All GitHub Actions workflows created and functional
- [ ] Repository secrets configured
- [ ] Branch protection rules enabled
- [ ] Environment variables configured
- [ ] End-to-end tests passing
- [ ] Documentation updated

## 📞 Support

For setup issues:
- **Repository Issues**: Create an issue in this repository
- **Documentation**: Check `/docs` directory for detailed guides

## 🔄 Post-Setup Tasks

After completing setup:
1. Update README with actual endpoints
2. Configure monitoring alerts
3. Set up regular backups
4. Review security settings

---

**Note**: This setup is required due to GitHub App permission limitations. All templates are provided in `docs/workflows/templates/`.