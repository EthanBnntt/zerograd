# Agent notes

## Lambda Labs GPU cloud

This machine can talk to Lambda Cloud. Credentials and ops live in:

- `.cursor/rules/lambda-labs.mdc` (always-applied Cursor rule)
- `~/.config/lambda/README.md` (local credential paths)

Quick start:

```bash
export LAMBDA_API_KEY="$(cat ~/.config/lambda/api_key)"
curl -sH "Authorization: Bearer $LAMBDA_API_KEY" \
  https://cloud.lambdalabs.com/api/v1/instances
ssh -i ~/.ssh/lambda.pem ubuntu@<INSTANCE-IP>
```

Launch with SSH key name `Lambda SSH` so `~/.ssh/lambda.pem` works. Do not commit API keys or PEM files.
