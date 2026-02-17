# Milestone 2: Containerized ML Service Deployment Pipeline

## Project Overview

This milestone demonstrates a **production-ready ML service deployment system** that integrates containerization best practices with automated CI/CD workflows. You will build a complete end-to-end solution that emphasizes **security, reproducibility, and automation**—fundamental skills for modern ML engineering roles.

### Key Competencies Demonstrated

1. **Multi-Stage Docker Containerization**
   - Optimized Docker image layers for reduced size and improved build performance
   - Security best practices including non-root user execution and minimal base images
   - Reproducible builds that work consistently across environments

2. **Automated CI/CD Pipeline**
   - Automated testing before deployment
   - Secure authentication to cloud infrastructure using Workload Identity Federation (no keys stored in code)
   - Automated container image building and publishing to a registry
   - Version-tagged releases for reproducibility

3. **Operations & Deployment**
   - Complete documentation for running and managing the service
   - Clear workflows for developers and operators
   - Production-ready configuration and best practices

---

## Architecture

### Service Overview

This project deploys a trained ML model as a **FastAPI microservice**, allowing users to make predictions via a RESTful API. The service is containerized using Docker and automatically deployed to Google Cloud infrastructure through CI/CD.

```
Developer
    ↓ (pushes version tag)
GitHub Actions CI/CD Pipeline
    ├→ Test Job (runs pytest, validates code quality)
    ├→ Build Job (creates Docker image, authenticates to GCP, pushes to registry)
    ↓
Google Artifact Registry (image storage)
    ↓
Deployment Environment (Cloud Run, GKE, etc.)
```

---

## Project Structure

```
milestone2/
├── app/
│   ├── app.py                 # FastAPI application with /predict endpoint
│   └── requirements.txt        # Production dependencies
├── tests/
│   ├── test_app.py           # Test suite for API endpoints
│   └── __pycache__/
├── train_model.py            # Model training script
├── Dockerfile                # Multi-stage Docker build configuration
├── pytest.ini                # Pytest configuration
├── requirements-dev.txt      # Development and testing dependencies
└── README.md                 # This file
```

---

## API Specification

### Prediction Endpoint

**POST** `/predict`

Accepts input features and returns a model prediction.

**Request Body:**
```json
{
  "features": [1, 2, 3]
}
```

**Response:**
```json
{
  "prediction": 0.95,
  "model_version": "v1.0.0"
}
```

**Status Codes:**
- `200 OK`: Prediction successful
- `400 Bad Request`: Invalid input format
- `500 Internal Server Error`: Model loading or prediction error

---

## Docker Containerization

### Multi-Stage Build

The `Dockerfile` implements a multi-stage build strategy to optimize image size and security:

1. **Build Stage**: Installs all dependencies and prepares the application
2. **Runtime Stage**: Contains only the necessary runtime components, reducing attack surface

### Security Considerations

- Uses non-root user for container execution (prevents privilege escalation)
- Minimal base image reduces attack surface
- Dependencies are pinned to specific versions for reproducibility

---

## Testing

The test suite (`tests/test_app.py`) validates:
- API endpoint functionality
- Input validation and error handling
- Model prediction correctness
- Edge cases and boundary conditions

Tests are automatically run on the GitHub Actions server as part of the CI/CD pipeline before deployment.

---

## CI/CD Pipeline

### Workflow Overview

The GitHub Actions workflow (`.github/workflows/milestone2-ci.yml`) automates the entire deployment process when you push a version tag.

### Workflow Architecture

```
┌─────────────────────────────────┐
│  Push Version Tag (v1.0.0)      │
└────────────┬────────────────────┘
             │
             ↓
    ┌────────────────────┐
    │   TEST JOB         │
    ├────────────────────┤
    │ • Checkout code    │
    │ • Setup Python     │
    │ • Install deps     │
    │ • Run pytest       │
    └────────┬───────────┘
             │
        ✓ All tests pass?
             │
             ↓ YES
    ┌─────────────────────────────────┐
    │   BUILD JOB                     │
    ├─────────────────────────────────┤
    │ • Authenticate to GCP (WIF)     │
    │ • Setup gcloud                  │
    │ • Configure Docker auth         │
    │ • Build Docker image            │
    │ • Push to Artifact Registry     │
    └─────────────────────────────────┘
             │
             ↓
    ┌─────────────────────────────────┐
    │   Image in Registry Ready       │
    │ (us-central1-docker.pkg.dev/...) │
    └─────────────────────────────────┘
```

### Test Job

When a version tag is pushed:
1. **Code Checkout**: Retrieves the repository code
2. **Environment Setup**: Installs Python 3.11
3. **Dependency Installation**: Installs from `app/requirements.txt` and `requirements-dev.txt`
4. **Test Execution**: Runs pytest with verbose output
5. **Validation**: Confirms code quality before proceeding to build

**Why This Matters**: Tests catch bugs early and prevent broken code from being deployed.

### Build Job

Runs only if the test job succeeds:
1. **Secure Authentication**: Uses Workload Identity Federation (WIF) to authenticate to Google Cloud without storing credentials
2. **GCP Setup**: Initializes Google Cloud SDK
3. **Docker Registry Auth**: Configures authentication to Google Artifact Registry
4. **Docker Build**: Creates an optimized image with the version tag
5. **Image Publishing**: Uploads the image to Google Artifact Registry for storage and deployment

**Security Feature**: WIF is a keyless authentication method—no service account keys are stored in GitHub, reducing security risk.

### Triggering the Pipeline

To trigger the CI/CD pipeline and deploy your service:

```bash
# Create a version tag
git tag v1.0.0

# Push the tag to GitHub
git push origin v1.0.0
```

**What Happens Next:**
- GitHub Actions automatically detects the tag
- Tests run and validate the code
- If tests pass, Docker image is built and published
- The image is tagged as `milestone2:v1.0.0` in the registry

### Deployed Images

Once published to Google Artifact Registry, images are ready for deployment to production servers using container orchestration platforms like Cloud Run, GKE, or Docker Swarm.

---

## Operations Runbook

### Deployment Checklist

Before pushing a new release:
- [ ] Code follows project standards
- [ ] Update version in code/docs if needed
- [ ] Create and push version tag: `git tag v1.x.x && git push origin v1.x.x`
- [ ] Monitor GitHub Actions for successful test and build jobs

### Monitoring & Troubleshooting

**Check CI/CD Status:**
- Visit the [GitHub Actions](https://github.com) tab in your repository
- View logs for any failed jobs
- Check test output for specific failures

**Common Issues:**

| Issue | Solution |
|-------|----------|
| Tests fail locally but pass in CI | Ensure Python version and dependencies match CI environment |
| Docker build fails | Check `Dockerfile` syntax and ensure all dependencies are in `requirements.txt` |
| GCP authentication fails | Verify Workload Identity Provider is properly configured |
| Image doesn't run | Ensure the FastAPI app listens on `0.0.0.0:8080` |

### Rolling Back a Release

If an issue is discovered after deployment, use your container orchestration platform (Cloud Run, GKE, etc.) to redeploy a previous version from Google Artifact Registry.

---

## Best Practices Applied

✅ **Reproducibility**: Pinned dependencies, multi-stage Docker builds  
✅ **Security**: Non-root user, minimal base image, keyless authentication (WIF)  
✅ **Automation**: Fully automated testing and deployment pipeline  
✅ **Testing**: Comprehensive test coverage before production deployment  
✅ **Documentation**: Clear runbooks for operations teams  
✅ **Versioning**: Semantic versioning with Git tags for release management  

---

## Next Steps

- Monitor deployed images in Google Artifact Registry
- Set up alerts for deployment failures
- Implement automated rollbacks for failed deployments
- Extend CI/CD for additional environments (staging, production)
- Add integration tests for deployed services
