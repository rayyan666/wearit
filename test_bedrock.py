#!/usr/bin/env python3
"""
test_bedrock_models.py
----------------------
Lists all Anthropic Claude models available in your AWS Bedrock region,
then runs a quick inference test on each one that is accessible.

Usage:
    python test_bedrock_models.py [--region us-east-1] [--profile default]

Requirements:
    pip install boto3
    AWS credentials configured (IAM role on EC2, or ~/.aws/credentials)
"""

import argparse
import json
import sys
import time

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
except ImportError:
    print("❌  boto3 not found. Run: pip install boto3")
    sys.exit(1)


# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Test Anthropic models on AWS Bedrock")
parser.add_argument("--region",  default="us-east-1",
                    help="AWS region (default: us-east-1). Try ap-south-1 for Mumbai.")
parser.add_argument("--profile", default=None,
                    help="AWS CLI profile name (default: instance role / env vars)")
parser.add_argument("--test-prompt", default="Reply with exactly one word: working",
                    help="Prompt sent to each model for the inference test")
args = parser.parse_args()

REGION  = args.region
PROFILE = args.profile
PROMPT  = args.test_prompt

print(f"\n{'='*60}")
print(f"  AWS Bedrock — Anthropic Model Checker")
print(f"  Region : {REGION}")
print(f"  Profile: {PROFILE or '(instance role / env)'}")
print(f"{'='*60}\n")

# ── boto3 session ─────────────────────────────────────────────────────────────
try:
    session = boto3.Session(profile_name=PROFILE, region_name=REGION)
    sts     = session.client("sts")
    identity = sts.get_caller_identity()
    print(f"✅  AWS identity: {identity['Arn']}\n")
except NoCredentialsError:
    print("❌  No AWS credentials found.")
    print("    On EC2: attach an IAM role with AmazonBedrockFullAccess.")
    print("    Locally: run 'aws configure' or set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY.")
    sys.exit(1)
except ClientError as e:
    print(f"❌  AWS auth error: {e}")
    sys.exit(1)


# ── list foundation models ────────────────────────────────────────────────────
bedrock = session.client("bedrock", region_name=REGION)

print("📋  Fetching foundation model list from Bedrock ...\n")
try:
    response = bedrock.list_foundation_models(byProvider="Anthropic")
    all_models = response.get("modelSummaries", [])
except ClientError as e:
    print(f"❌  Could not list models: {e}")
    sys.exit(1)

if not all_models:
    print(f"⚠️   No Anthropic models returned for region '{REGION}'.")
    print("    Try --region us-east-1 or us-west-2 where Bedrock has broadest coverage.")
    sys.exit(0)

# Sort by model ID
all_models.sort(key=lambda m: m.get("modelId", ""))

print(f"Found {len(all_models)} Anthropic model(s) in {REGION}:\n")
print(f"  {'MODEL ID':<55} {'STATUS':<15} {'INPUT'}")
print(f"  {'-'*55} {'-'*15} {'-'*20}")

for m in all_models:
    mid    = m.get("modelId", "?")
    status = m.get("modelLifecycle", {}).get("status", m.get("modelAvailability", "?"))
    modes  = ", ".join(m.get("inputModalities", []))
    print(f"  {mid:<55} {status:<15} {modes}")

print()


# ── inference test ────────────────────────────────────────────────────────────
runtime = session.client("bedrock-runtime", region_name=REGION)

# Only test models that are ACTIVE / AVAILABLE and support TEXT input
testable = [
    m for m in all_models
    if "TEXT" in m.get("inputModalities", [])
    and m.get("modelLifecycle", {}).get("status", "").upper() in ("ACTIVE", "")
    or m.get("modelAvailability", "").upper() == "AVAILABLE"
]

# De-dupe (cross-region inference profiles can duplicate)
seen = set()
unique_testable = []
for m in testable:
    mid = m.get("modelId", "")
    if mid not in seen:
        seen.add(mid)
        unique_testable.append(m)

print(f"🧪  Running inference test on {len(unique_testable)} model(s) ...\n")
print(f"  Prompt: \"{PROMPT}\"\n")

results = []

for m in unique_testable:
    model_id = m.get("modelId", "")
    print(f"  ▶  {model_id}")

    # Build request body — all Claude models use the Messages API on Bedrock
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": PROMPT}
        ]
    })

    t0 = time.time()
    try:
        resp = runtime.invoke_model(
            modelId=model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        result_body = json.loads(resp["body"].read())
        elapsed = round(time.time() - t0, 2)

        # Extract text from response
        content = result_body.get("content", [])
        text = " ".join(c.get("text", "") for c in content if c.get("type") == "text").strip()
        tokens_in  = result_body.get("usage", {}).get("input_tokens",  "?")
        tokens_out = result_body.get("usage", {}).get("output_tokens", "?")

        print(f"     ✅  Response: \"{text}\"  ({elapsed}s | in:{tokens_in} out:{tokens_out})\n")
        results.append({"model": model_id, "status": "OK", "response": text, "latency": elapsed})

    except ClientError as e:
        code = e.response["Error"]["Code"]
        msg  = e.response["Error"]["Message"]
        elapsed = round(time.time() - t0, 2)

        if code == "AccessDeniedException":
            reason = "NOT ENABLED — request model access in Bedrock console"
        elif code == "ValidationException":
            reason = f"Validation error: {msg}"
        elif code == "ThrottlingException":
            reason = "Throttled — try again later"
        else:
            reason = f"{code}: {msg}"

        print(f"     ❌  {reason}\n")
        results.append({"model": model_id, "status": "FAILED", "reason": reason})

    time.sleep(0.5)  # be polite to the API


# ── summary ───────────────────────────────────────────────────────────────────
ok     = [r for r in results if r["status"] == "OK"]
failed = [r for r in results if r["status"] == "FAILED"]

print(f"\n{'='*60}")
print(f"  SUMMARY  —  {len(ok)} passed, {len(failed)} failed")
print(f"{'='*60}")

if ok:
    print("\n  ✅  Working models:")
    for r in ok:
        print(f"      {r['model']}  ({r['latency']}s)")

if failed:
    print("\n  ❌  Failed models:")
    for r in failed:
        print(f"      {r['model']}")
        print(f"         └─ {r['reason']}")

print()

# ── hint for server.py ────────────────────────────────────────────────────────
if ok:
    best = ok[0]["model"]
    # Prefer claude-3 or claude-sonnet if available
    for r in ok:
        if "claude-3" in r["model"] or "sonnet" in r["model"]:
            best = r["model"]
            break

    print(f"💡  Recommended model for detect_garment_category() in server.py:")
    print(f"    model_id = \"{best}\"")
    print()
    print("    Update server.py to use boto3 bedrock-runtime instead of anthropic SDK:")
    print("""
    import boto3, json

    bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")

    async def detect_garment_category(garment_b64, mime_type="image/jpeg"):
        try:
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 10,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": garment_b64
                        }},
                        {"type": "text", "text":
                            "Classify this garment. Reply with exactly one word: "
                            "'upper' (shirts/tops/jackets), 'lower' (pants/skirts), "
                            "or 'overall' (dresses/jumpsuits/rompers)."}
                    ]
                }]
            })
            resp = bedrock_runtime.invoke_model(
                modelId=\"""" + best + """\",
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            result = json.loads(resp["body"].read())
            text = result["content"][0]["text"].strip().lower()
            return text if text in ("upper", "lower", "overall") else "upper"
        except Exception as e:
            print(f"[detect-category] ERROR: {e}")
            return "upper"
    """)