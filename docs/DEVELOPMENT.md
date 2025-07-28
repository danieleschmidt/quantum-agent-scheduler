# Development Guide

## Prerequisites
- Node.js (latest LTS)
- Git
- Your preferred IDE/editor

## Quick Start
```bash
git clone <repository-url>
cd <project-name>
npm install
npm run dev
```

## Available Scripts
- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run test` - Run test suite
- `npm run lint` - Check code style

## Project Structure
- `/src` - Source code
- `/docs` - Documentation
- `/tests` - Test files
- `/.github` - GitHub workflows and templates

## Code Style
- Follow .editorconfig settings
- Use ESLint and Prettier
- Run pre-commit hooks

## Testing
Run the full test suite before submitting PRs.

## Resources
- [Project Architecture](../ARCHITECTURE.md)
- [ADR Documentation](./adr/README.md)
- [Contributing Guidelines](../CONTRIBUTING.md)