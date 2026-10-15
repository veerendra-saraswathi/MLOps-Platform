"""Register a model from an MLflow run ID via the control plane API."""

import argparse
import httpx


def main():
    parser = argparse.ArgumentParser(description="Register a model via the MLOps control plane")
    parser.add_argument("--name", required=True, help="Model name")
    parser.add_argument("--run-id", required=True, help="MLflow run ID")
    parser.add_argument("--stage", default="Staging", help="Stage (Staging / Production)")
    parser.add_argument("--api", default="http://localhost:8000", help="Control plane base URL")
    args = parser.parse_args()

    url = f"{args.api}/models/register"
    payload = {
        "name": args.name,
        "run_id": args.run_id,
        "stage": args.stage,
    }

    with httpx.Client(timeout=30.0) as client:
        resp = client.post(url, json=payload)
        if resp.status_code != 200:
            print(f"Error {resp.status_code}: {resp.text}")
            return
        print(resp.json())


if __name__ == "__main__":
    main()