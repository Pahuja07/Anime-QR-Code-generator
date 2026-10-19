# AniQR CI/CD setup

Workflow: `.github/workflows/ci-cd.yml`

## What runs

- Pull requests to `main`: install lightweight test dependencies, run the Python
  suite, and compile the application modules.
- Pushes to `main`: after checks pass, build and publish immutable SHA tags plus
  `latest` for both the RunPod worker and FastAPI API images to GHCR.
- RunPod rollout: after image publishing, update the configured Serverless
  template to the new immutable worker image through RunPod REST API v2.
- `workflow_dispatch` on `main`: manually rerun publish and deployment.

## One-time GitHub and RunPod setup

1. Push this project to GitHub on a `main` branch. This workspace currently has
   no `.git` directory or remote, so Actions cannot run until it is connected to
   a GitHub repository.
2. In GitHub repository **Settings → Secrets and variables → Actions**, add:
   - Secret `RUNPOD_API_KEY`: a RunPod API key with template update permissions.
   - Variable `RUNPOD_TEMPLATE_ID`: the ID of the existing Serverless template
     used by the AniQR endpoint.
3. Set up that RunPod template and endpoint once in RunPod. Give the template
   access to the worker image. GHCR packages are private by default: either make
   the worker package public, or configure registry credentials for the private
   package on the RunPod template.
4. On the first push, GitHub publishes packages named
   `ghcr.io/<owner>/<repo>-worker` and `ghcr.io/<owner>/<repo>-api`. If keeping
   them private, grant the RunPod template pull access before the first deploy.
5. Configure `RUNPOD_API_KEY` and `RUNPOD_ENDPOINT_ID` on the API host so the
   FastAPI service can call the GPU worker. Keep Razorpay credentials in the API
   host's secret store as well.

The worker update uses RunPod's v2 template update route and an immutable
`sha-<commit>` image tag, so an earlier workflow run can be redeployed to roll
back. The API image is published as a deployable artifact; deploying it to a
public web host needs that host's service configuration and credentials, which
are not present in this project workspace.

## Local image builds

```powershell
docker build -t aniqr-worker .
docker build -f Dockerfile.api -t aniqr-api .
```

The worker uses the CUDA/PyTorch image. The API has a separate Python image and
calls the worker remotely when RunPod credentials are configured. Start the API
with port `8000` exposed and set `ANIQR_ALLOWED_ORIGINS` to the frontend's
browser origin.
