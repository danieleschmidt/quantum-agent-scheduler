# Software Bill of Materials (SBOM)

## Overview

This document outlines the Software Bill of Materials (SBOM) generation and management practices for the Quantum Agent Scheduler project.

## SBOM Generation

### Automated Generation

SBOMs are automatically generated during the release process using:

```yaml
# In GitHub Actions Release workflow
- name: Generate SBOM
  uses: anchore/sbom-action@v0
  with:
    path: .
    format: spdx-json
    output-file: sbom.spdx.json
```

### Manual Generation

For local SBOM generation:

```bash
# Install syft
curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin

# Generate SBOM
syft packages . -o spdx-json=sbom.spdx.json
syft packages . -o cyclonedx-json=sbom.cyclonedx.json
```

## SBOM Formats

We support multiple SBOM formats:

- **SPDX**: Primary format for compliance and legal review
- **CycloneDX**: Alternative format for vulnerability scanning
- **Syft JSON**: Detailed format for internal tooling

## SBOM Contents

Our SBOMs include:

### Direct Dependencies
- Production Python packages from poetry.lock
- Development dependencies (marked as dev-only)
- Optional quantum backend dependencies

### System Dependencies
- Base Docker image components
- Operating system packages
- System libraries

### Build Dependencies
- Poetry build system
- Python interpreter version
- Build tools and compilers

## Compliance and Security

### SLSA (Supply Chain Levels for Software Artifacts)

We aim for **SLSA Level 3** compliance:

- ✅ Source integrity verified
- ✅ Build process documented
- ✅ Provenance generation
- ✅ Isolated build environment

### Security Scanning

SBOMs are used for:

- **Vulnerability Assessment**: Automated scanning with Trivy/Grype
- **License Compliance**: OSS license analysis
- **Supply Chain Security**: Dependency risk assessment

## SBOM Verification

### Digital Signatures

Release SBOMs are signed with:

```bash
# Cosign signature verification
cosign verify-blob --certificate sbom.spdx.json.cert \
  --signature sbom.spdx.json.sig \
  sbom.spdx.json
```

### Integrity Checks

```bash
# SHA256 checksums
sha256sum sbom.spdx.json > sbom.spdx.json.sha256
```

## Integration Points

### CI/CD Pipeline
- SBOM generation on every release
- Vulnerability scanning of SBOM contents
- License compliance checks

### Package Distribution
- SBOMs published with GitHub releases
- PyPI integration for package metadata
- Docker image layer analysis

### Monitoring
- Continuous dependency monitoring
- Automated security alerts
- License compliance tracking

## Tools and Standards

### SBOM Tools
- **Syft**: Primary SBOM generation
- **CycloneDX CLI**: Alternative generator
- **SPDX Tools**: Validation and analysis

### Standards Compliance
- **SPDX 2.3**: Industry standard format
- **CycloneDX 1.4**: OWASP standard
- **NIST SSDF**: Secure software development

## Repository Integration

```bash
# Add to Makefile
sbom: ## Generate SBOM
	syft packages . -o spdx-json=sbom.spdx.json
	syft packages . -o cyclonedx-json=sbom.cyclonedx.json

sbom-verify: ## Verify SBOM integrity
	spdx-tools validate sbom.spdx.json
```

## Quantum-Specific Considerations

### Quantum Backend Dependencies
- IBM Qiskit: Open source, Apache 2.0
- AWS Braket SDK: Open source, Apache 2.0
- D-Wave Ocean SDK: Open source, Apache 2.0
- Azure Quantum: Proprietary client libraries

### Export Control Compliance
- No ITAR-controlled quantum algorithms
- Classical optimization components only
- Open source quantum computing libraries

## Maintenance

- **Weekly**: Automated dependency updates
- **Monthly**: SBOM accuracy review
- **Quarterly**: Compliance audit
- **Release**: Full SBOM regeneration

## References

- [NIST SSDF Framework](https://csrc.nist.gov/Projects/ssdf)
- [SLSA Framework](https://slsa.dev/)
- [SPDX Specification](https://spdx.github.io/spdx-spec/)
- [CycloneDX Specification](https://cyclonedx.org/specification/overview/)