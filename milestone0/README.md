# Milestone 0: Reproducible ML Environments

## 1. Setup Steps & Reproducibility Principles

### Setup Steps

1. **Create isolated environment**: `python -m venv venv`
2. **Activate environment**: `source venv/bin/activate`
3. **Install dependencies**: `pip install -r requirements.txt`
4. **Run smoke tests**: `pytest tests/`

### Key Reproducibility Principles Applied

- **Isolated Environments**: Python `venv` keeps project dependencies isolated from system packages
- **Version Pinning**: `requirements.txt` specifies exact package versions (e.g., `pytest==9.0.2`) ensuring deterministic, consistent installs
- **Automated CI Testing**: GitHub Actions validates setup in a clean environment, catching issues before deployment

---

## 2. CI Badge Status

[![Milestone0-CI](https://github.com/santhoshprathik1530/IDS568-MLOPS-Assignments/actions/workflows/ci.yml/badge.svg)](https://github.com/santhoshprathik1530/IDS568-MLOPS-Assignments/actions/workflows/ci.yml)

---

## 3. ML Lifecycle & Deployment Reliability

Environment reproducibility is fundamental to ML system reliability. During development, training, and evaluation phases, code behavior depends entirely on specific library versions and Python runtime—mismatches between environments cause models to behave unpredictably. By pinning exact dependency versions and using isolated environments, we ensure identical conditions across machines.

Our CI workflow validates this by installing dependencies in a completely clean environment and running automated smoke tests, simulating real deployment pipelines. This catches configuration issues early, preventing the costly "it works locally but fails in production" problem. Reproducible environments support the full ML lifecycle: training produces consistent models, evaluation results are trustworthy, and deployment succeeds with confidence.

Environment management directly translates to deployment success by ensuring development, testing, and production stages operate under aligned configurations. This practice builds production-grade ML systems that are reliable, maintainable, and traceable—qualities essential for enterprise ML operations.

