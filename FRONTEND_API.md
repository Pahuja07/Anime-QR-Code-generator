# AniQR frontend API

## Local endpoint

Run the FastAPI service with Docker:

```powershell
docker build -f Dockerfile.api -t aniqr-api .
docker run --rm -p 8000:8000 --env-file .env aniqr-api
```

The local base URL is `http://localhost:8000`. Interactive API docs are at
`http://localhost:8000/docs`; health check: `GET /api/v1/health`.

Set `ANIQR_ALLOWED_ORIGINS` in `.env` to the exact origin(s) of the frontend,
comma-separated, for example `https://aniqr.example.com`. The API must be
reachable from the browser; `localhost` only works when frontend and API run on
the same computer.

## Frontend calls

First create a guest session:

```js
const API_BASE_URL = "http://localhost:8000"; // replace with deployed API origin

const guestResponse = await fetch(`${API_BASE_URL}/api/v1/auth/guest`, {
  method: "POST",
});
if (!guestResponse.ok) throw new Error("Could not create AniQR session");
const { guest_token, credits } = await guestResponse.json();
```

Generate an image (the uploaded image is optional):

```js
const response = await fetch(`${API_BASE_URL}/api/v1/generate`, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    Authorization: `Bearer ${guest_token}`,
  },
  body: JSON.stringify({
    content: "https://example.com",
    content_type: "url", // url | text | phone
    style: "iron_man",
    reference_image_base64: null, // optional image data URL from FileReader
  }),
});
const result = await response.json();
if (!response.ok) throw new Error(result.detail ?? "QR generation failed");
// Display the generated PNG directly:
imageElement.src = result.image_base64;
```

Generation returns `image_base64` as a PNG data URL, `decoded_content`,
`is_scannable`, `attempts`, and `credits_remaining`. `POST /api/v1/credits`
is not used; check the balance with `GET /api/v1/credits` and the same Bearer
token. The payment routes are documented in `/docs`.

## Public URL

This repository does not have a public hostname or hosting credentials, so no
public URL can be assigned from this workspace. Deploy the `api` Docker target
to a host with HTTPS, configure the RunPod API credentials and
`ANIQR_ALLOWED_ORIGINS`, then use that host's origin as `API_BASE_URL` in the
frontend. Keep Razorpay and RunPod secrets on the server; never put them in
browser code.
