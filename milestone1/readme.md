Milestone 1 – Model Serving with Cloud Run and Cloud Functions

------------------------------------------------------------
1. Project Overview
------------------------------------------------------------

This assignment demonstrates how a trained machine learning model can be served
as a web API and deployed to the cloud using two different approaches:

1. Google Cloud Run (container-based deployment)
2. Google Cloud Functions (serverless deployment)

The focus of this milestone is model serving, deployment, and runtime behavior,
not model accuracy or retraining.

------------------------------------------------------------
2. Machine Learning Model
------------------------------------------------------------

- Model type: Linear Regression
- Library used: scikit-learn
- Input: A list of numerical features
- Output: A single numerical prediction

The model is trained offline using Python and saved as:

model.pkl

This trained model file is reused across:
- Local FastAPI testing
- Cloud Run deployment
- Cloud Function deployment

------------------------------------------------------------
3. Local FastAPI Service
------------------------------------------------------------

A FastAPI application is created to serve predictions locally before cloud
deployment.

API details:
- Endpoint: /predict
- Method: POST
- Input format: JSON
- Output format: JSON

Example input:
{
  "features": [1, 2, 3]
}

Example output:
{
  "prediction": 9.999999999999998,
  "version": "v1.0"
}

Important implementation details:
- The model (model.pkl) is loaded once at application startup.
- The model is NOT reloaded for every request.
- Pydantic is used for both request and response validation.

------------------------------------------------------------
4. Cloud Run Deployment (Container-Based)
------------------------------------------------------------

The FastAPI application is deployed using Google Cloud Run.

How Cloud Run works:
- The application is packaged into a Docker container.
- Containers start only when requests are received.
- Containers automatically stop when idle (scale-to-zero).
- No servers are manually managed.

Container build process:
1. A Dockerfile defines the runtime environment.
2. Cloud Build uses the Dockerfile to build a Docker image.
3. The image is stored in Artifact Registry.
4. Cloud Run pulls the image and runs containers from it.

Cloud Run service:
- Region: us-central1
- Public access: Enabled (unauthenticated)

Cloud Run endpoint:
POST https://<CLOUD-RUN-SERVICE-URL>/predict

Public access verification:
- The endpoint was tested using curl in an unauthenticated environment.
- HTTP 200 response confirms public accessibility.

Note:
- The /predict endpoint is POST-only and cannot be accessed directly via browser.

------------------------------------------------------------
5. Cloud Functions Deployment (Serverless)
------------------------------------------------------------

The same trained model is also deployed using Google Cloud Functions.

How Cloud Functions work:
- Code runs only when a request is received.
- Each invocation is stateless.
- The model is loaded during cold starts.
- No container or server configuration is required.

Cloud Function details:
- Runtime: Python
- Region: us-central1
- Trigger type: HTTP

Cloud Function endpoint:
POST https://us-central1-<PROJECT-ID>.cloudfunctions.net/predict

The input and output format is identical to the Cloud Run API.

------------------------------------------------------------
6. Cold Start vs Warm Request Behavior
------------------------------------------------------------

Latency was observed under two conditions:

Cold start:
- First request after the service has been idle.
- Slower due to container/function initialization.

Warm request:
- Subsequent requests while the service is active.
- Faster because the runtime is already initialized.

Observations:
- Cloud Functions show higher cold start latency.
- Cloud Run performs better on warm requests because containers remain alive
  briefly after handling traffic.

------------------------------------------------------------
7. Cloud Run vs Cloud Functions Comparison
------------------------------------------------------------

Cloud Run:
- Runs applications inside Docker containers.
- Semi-stateful (model loaded once per container).
- Better reproducibility due to containerization.
- Suitable for APIs and long-running services.

Cloud Functions:
- Runs individual functions.
- Fully stateless.
- Higher cold start overhead.
- Suitable for event-driven tasks.

------------------------------------------------------------
8. Reproducibility
------------------------------------------------------------

- All dependencies are pinned using requirements.txt.
- Cloud Run ensures strong reproducibility by packaging the entire runtime
  environment inside a Docker image.
- Cloud Functions rely on managed runtimes provided by Google.

------------------------------------------------------------
9. Project Structure
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
10. Checklist Confirmation
------------------------------------------------------------

- FastAPI service implemented with /predict endpoint
- Pydantic request and response models used
- Model artifact (model.pkl) included
- Dependencies pinned
- Cloud Run deployed with public access
- Artifact Registry used for Docker image storage
- Cloud Function deployed and tested
- Public inference verified using curl
- Cold start vs warm behavior documented
- Cloud Run vs Cloud Functions comparison included

------------------------------------------------------------
11. Final Summary
------------------------------------------------------------

In this milestone, a trained machine learning model was successfully served
as an API, deployed using Cloud Run and Cloud Functions, and evaluated for
runtime behavior and deployment trade-offs.

This assignment demonstrates real-world MLOps concepts including model serving,
containerization, serverless deployment, and cloud-based inference.