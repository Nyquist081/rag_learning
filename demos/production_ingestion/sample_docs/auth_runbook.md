# Authentication Runbook

## Token Expired

When an API request returns `401 Token Expired`, refresh the access token and retry the request once.
Do not retry indefinitely. If refresh fails, invalidate the session and ask the user to sign in again.

## Retry Policy

| Error | Retry | Action |
| --- | --- | --- |
| `401 Token Expired` | Once | Refresh token, then retry |
| `403 Forbidden` | No | Check user permissions |
| `429 Too Many Requests` | Yes | Use exponential backoff |

## Example

```python
def refresh_and_retry(request):
    token = refresh_access_token()
    return request.with_token(token).send()
```

