Milestone 1 – Model Serving with Cloud Run and Cloud Functions

============================================================

Project Name: mlops-milestone-1
Course: IDS 568 – MLOps

------------------------------------------------------------
1. Project Overview
------------------------------------------------------------

This assignment demonstrates how a trained machine learning model can be served
as a web API and deployed to the cloud using two different approaches:

1. Google Cloud Run (container-based deployment)
2. Google Cloud Functions (serverless deployment)

The goal of this milestone is to understand model serving, deployment options,
runtime behavior, and trade-offs between container-based and serverless systems.

------------------------------------------------------------
2. Machine Learning Model
------------------------------------------------------------

- Model type: Linear Regression
- Library used: scikit-learn
- Input: List of numerical features
- Output: Single numerical prediction

The model is trained offline using Python and saved as:

model.pkl

This trained model artifact is reused for:
- Local FastAPI testing
- Cloud Run deployment
- Cloud Function deployment

------------------------------------------------------------
3. Local FastAPI Service
------------------------------------------------------------

A FastAPI application is implemented to serve predictions locally before
deploying to the cloud.

API details:
- Endpoint: /predict
- Method: POST
- Input format: JSON
- Output format: JSON

Example request:
{
  "features": [1, 2, 3]
}

Example response:
{
  "prediction": 10.0,
  "version": "v1.0"
}

Important implementation details:
- The trained model (model.pkl) is loaded once at application startup.
- The model is not reloaded for every request.
- Pydantic is used for request and response validation.

------------------------------------------------------------
4. Setup and Deployment Summary
------------------------------------------------------------

The following steps summarize how this project was set up and deployed:

1. Train the machine learning model using train_model.py.
2. Verify predictions locally using the FastAPI application.
3. Create a Dockerfile for the FastAPI service.
4. Use Cloud Build to build a Docker image.
5. Store the Docker image in Artifact Registry.
6. Deploy the containerized application to Cloud Run.
7. Deploy the same model using Google Cloud Functions.
8. Verify public access and inference using curl.

------------------------------------------------------------
5. Cloud Run Deployment (Container-Based)
------------------------------------------------------------

The FastAPI application is deployed using Google Cloud Run.

How Cloud Run works:
- The application runs inside a Docker container.
- Containers start only when requests arrive.
- Containers automatically stop when idle (scale-to-zero).
- No servers are manually managed.

Container build process:
1. Cloud Build reads the Dockerfile and builds a Docker image.
2. The image is stored in Artifact Registry.
3. Cloud Run pulls the image and runs containers from it.

Cloud Run details:
- Project: mlops-milestone-1
- Region: us-central1
- Public access: Enabled (unauthenticated)

Cloud Run endpoint:
POST https://milestone1-api-938213003679.us-central1.run.app/predict

Public access verification:
- The endpoint was tested using curl without authentication.
- HTTP 200 response confirms the service is publicly accessible.

------------------------------------------------------------
6. Cloud Functions Deployment (Serverless)
------------------------------------------------------------

The same trained model is also deployed using Google Cloud Functions.

How Cloud Functions work:
- Code runs only when a request is received.
- Each invocation is stateless.
- The model is loaded during cold starts.
- No container or server management is required.

Cloud Function details:
- Project: mlops-milestone-1
- Runtime: Python
- Region: us-central1
- Trigger type: HTTP

Cloud Function endpoint:
POST https://us-central1-mlops-milestone-1.cloudfunctions.net/predict

------------------------------------------------------------
7. Cold Start vs Warm Request Behavior
------------------------------------------------------------

Latency was observed under two conditions:

Cold start:
- First request after the service has been idle.
- Slower due to container or function initialization.

Warm request:
- Subsequent requests while the service is active.
- Faster because the runtime is already initialized.

Observations:
- Cloud Functions generally have higher cold start latency.
- Cloud Run performs better on warm requests because containers
  remain alive briefly after handling traffic.

------------------------------------------------------------
8. Cloud Run vs Cloud Functions Comparison
------------------------------------------------------------

Cloud Run:
- Runs applications inside Docker containers.
- Semi-stateful (model loaded once per container).
- Strong reproducibility due to containerization.
- Suitable for APIs and long-running services.

Cloud Functions:
- Runs individual functions.
- Fully stateless.
- Higher cold start overhead.
- Suitable for event-driven tasks.

------------------------------------------------------------
9. Reproducibility Considerations
------------------------------------------------------------

- All dependencies are pinned using requirements.txt.
- Cloud Run provides strong reproducibility because the entire runtime
  environment is packaged inside a Docker image.
- Cloud Functions rely on managed runtimes provided by Google.

------------------------------------------------------------
10. Final Inference Verification (curl Test)
------------------------------------------------------------

Cloud Run curl test:

curl -X POST https://milestone1-api-938213003679.us-central1.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{"features":[1,2,3]}'

Expected response format:
{
  "prediction": 10.0,
  "version": "v1.0"
}

This confirms:
- Public access is enabled
- No authentication is required
- The model loads correctly and returns predictions

------------------------------------------------------------
11. Project Structure
------------------------------------------------------------

milestone1/
├── main.py
├── model.pkl
├── train_model.py
├── requirements.txt
├── Dockerfile
├── cloud_function/
│   ├── main.py
│   ├── model.pkl
│   └── requirements.txt
└── README.txt

------------------------------------------------------------
12. Checklist Confirmation
------------------------------------------------------------

- FastAPI container vs Cloud Function comparison included
- Lifecycle differences (stateful vs stateless) explained
- Artifact loading strategies compared
- Latency characteristics documented (cold starts, warm requests)
- Reproducibility considerations discussed
- Setup and deployment steps included
- API usage examples provided
- Model-API interaction clearly described
- Deployment URLs included and publicly accessible
- Final curl-based inference verification included

------------------------------------------------------------
13. Final Summary
------------------------------------------------------------

In this milestone, a trained machine learning model was successfully served
as a web API, deployed using Cloud Run and Cloud Functions, and evaluated
for runtime behavior and deployment trade-offs.

This assignment demonstrates practical MLOps concepts including model serving,
containerization, serverless deployment, and cloud-based inference.