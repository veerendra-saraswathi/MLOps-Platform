from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import mlflow
from mlflow.tracking import MlflowClient

app = FastAPI(
    title="MLOps Platform Control Plane",
    version="0.1.0",
    description="Lightweight control plane for model registry, serving, and GPU orchestration hooks",
)

mlflow.set_tracking_uri("http://localhost:5000")


class ModelRegisterRequest(BaseModel):
    name: str
    run_id: str
    stage: str = "Staging"
    description: Optional[str] = None


class TrainJobRequest(BaseModel):
    name: str
    image: str
    command: list[str]
    gpu_count: int = 1
    gpu_memory_gb: Optional[int] = None


@app.get("/health")
def health():
    try:
        uri = mlflow.get_tracking_uri()
        return {"status": "ok", "mlflow_tracking_uri": uri}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


@app.post("/models/register")
def register_model(req: ModelRegisterRequest):
    client = MlflowClient()
    try:
        mv = client.create_model_version(
            name=req.name,
            source=f"runs:/{req.run_id}/model",
            run_id=req.run_id,
            description=req.description,
        )
        client.transition_model_version_stage(
            name=req.name,
            version=mv.version,
            stage=req.stage,
        )
        return {
            "model": req.name,
            "version": mv.version,
            "stage": req.stage,
            "status": "registered",
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/models/{name}")
def get_model(name: str):
    client = MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{name}'")
        return {
            "name": name,
            "versions": [
                {"version": v.version, "stage": v.current_stage, "run_id": v.run_id}
                for v in versions
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/jobs/train")
def submit_train_job(req: TrainJobRequest):
    return {
        "job_name": req.name,
        "status": "accepted",
        "gpu_count": req.gpu_count,
        "note": "Wire this to Kubernetes + KAI Scheduler in the next iteration",
    }