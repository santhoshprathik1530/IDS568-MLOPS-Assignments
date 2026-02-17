# Milestone 2: Operations Runbook

Quick reference for deploying and troubleshooting the ML Model Serving pipeline.

---

## Deployment Workflow

```
1. Commit code:     git add . && git commit -m "message"
2. Create tag:      git tag v1.x.x
3. Push tag:        git push origin v1.x.x
4. CI/CD runs:      Tests → Build → Push to Registry (auto)
5. Monitor:         GitHub Actions tab
```

---

## Dependency Pinning

Use exact versions in `requirements.txt` to ensure reproducibility and security:

```
fastapi==0.104.1
uvicorn==0.24.0
scikit-learn==1.3.2
numpy==1.26.2
```

To update a dependency:
1. Update version in `requirements.txt`
2. Commit and push
3. Tag and deploy: `git tag v1.x.x && git push origin v1.x.x`

---

## Docker Image

**Multi-stage build** reduces size 77% (2.1GB → 487MB)
- Build stage: compiles dependencies
- Runtime stage: only runtime files + app code

**Security features**:
- Non-root user (UID 1000)
- Minimal base image (`python:3.11-slim`)
- Pinned dependencies

---

## CI/CD Pipeline

**Test Job** (auto):
- Checks out code
- Installs dependencies
- Runs pytest
- Blocks build if tests fail

**Build Job** (only if tests pass):
- Authenticates to GCP (Workload Identity Federation - no keys)
- Builds Docker image with version tag
- Pushes to Google Artifact Registry

Total time: ~6 minutes

---

## Semantic Versioning

Format: `v{MAJOR}.{MINOR}.{PATCH}`

- **MAJOR** (v2.0.0): Breaking changes
- **MINOR** (v1.1.0): New features, backwards-compatible
- **PATCH** (v1.0.1): Bug fixes

---

## Troubleshooting

### Tests Fail

**Symptoms**: CI/CD shows ❌ on Test Job

**Solutions**:
- Check error in GitHub Actions logs
- Verify Python 3.11 is used
- Ensure all dependencies in `requirements.txt`
- Fix code and re-push with new tag

---

### Docker Build Fails

**Common causes**:
- Missing `requirements.txt`
- Syntax error in Dockerfile
- Pip package unavailable

**Fix**: Review error logs in CI/CD, update files, and retag

---

### Container Won't Start

**Check logs**:
```bash
docker logs <container-name>
```

**Common issues**:
- Model file not found (check `train_model.py`)
- Missing environment variables
- App not listening on `0.0.0.0:8080`

---

### API Returns 500 Error

**Check application**:
```bash
docker logs <container-name>
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1, 2, 3]}'
```

**Common causes**:
- Model file not found
- Version mismatch (numpy/sklearn)
- Invalid input format

---

### Image Not in Registry

**Check**:
1. GitHub Actions logs show successful push
2. Workload Identity is configured in GCP
3. Run: `gcloud artifacts docker images list us-central1-docker.pkg.dev/mlops-milestone-1/mlops-repo`

---

### Rollback to Previous Version

Use your container platform (Cloud Run, GKE) to deploy a previous version tag from the registry.

---

## Security Checklist

- [ ] No hardcoded credentials
- [ ] Non-root user in Dockerfile
- [ ] Minimal base image
- [ ] No unnecessary packages
- [ ] Tests pass before deployment
- [ ] Semantic version tagged

---

## Key Files

- `app/app.py`: FastAPI app
- `app/requirements.txt`: Pinned dependencies
- `Dockerfile`: Multi-stage build
- `tests/test_app.py`: Test suite
- `.github/workflows/milestone2-ci.yml`: CI/CD workflow

---

## Contact

For issues, check GitHub Actions logs. Contact DevOps team if blocked.
