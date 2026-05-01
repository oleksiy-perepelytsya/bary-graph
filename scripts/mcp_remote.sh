#!/usr/bin/env bash
# Remote MCP server — pipes stdio through SSH to the VPS.
# Uses the gcloud-generated SSH alias (run `gcloud compute config-ssh` once to set up).

VPS_HOST="bary-mcp.europe-central2-b.barygraph-494511"
VPS_PATH="${BARYGRAPH_VPS_PATH:-~/bary-vector}"

exec ssh -T -o BatchMode=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=6 "${VPS_HOST}" \
  "cd ${VPS_PATH} && MONGO_URI='mongodb://localhost:27117/?directConnection=true' .venv/bin/python -m scripts.mcp_server"
