import json
from datetime import datetime, timezone
from pathlib import Path


REGISTRY_DIR = Path("models/registry")
PRODUCTION_FILE = REGISTRY_DIR / "production.json"


def load_metadata(version):
    """Carga los metadatos de una versión registrada."""
    path = REGISTRY_DIR / version / "metadata.json"

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def should_promote(baseline_metrics, candidate_metrics):
    """
    Política de promoción.

    El candidato debe mantener o mejorar:
    - F1
    - Recall
    """
    f1_ok = candidate_metrics["f1"] >= baseline_metrics["f1"]
    recall_ok = (
        candidate_metrics["recall"]
        >= baseline_metrics["recall"]
    )

    return f1_ok and recall_ok


def update_candidate_status(version, status):
    """Actualiza el estado de una versión registrada."""
    path = REGISTRY_DIR / version / "metadata.json"

    with open(path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    metadata["status"] = status

    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def register_approved_candidate(version):
    """
    Registra qué modelo fue aprobado por el pipeline.

    No sustituye todavía el modelo físico utilizado por FastAPI.
    Esto evita modificar producción automáticamente hasta que
    exista un mecanismo persistente de despliegue de artefactos.
    """
    with open(PRODUCTION_FILE, "r", encoding="utf-8") as f:
        production = json.load(f)

    production["approved_candidate"] = {
        "version": version,
        "model_file": f"models/registry/{version}/modelo.pkl",
        "columns_file": f"models/registry/{version}/columnas.pkl",
        "approved_at": datetime.now(timezone.utc).isoformat(),
        "deployment_status": "pending",
    }

    with open(PRODUCTION_FILE, "w", encoding="utf-8") as f:
        json.dump(production, f, indent=2)


def main():
    baseline_version = "v1.0.0-baseline"
    candidate_version = "v1.1.0"

    baseline = load_metadata(baseline_version)
    candidate = load_metadata(candidate_version)

    baseline_metrics = baseline["metrics"]
    candidate_metrics = candidate["metrics"]

    print("=== POLÍTICA DE PROMOCIÓN ===")

    print("\nF1:")
    print(f"Baseline : {baseline_metrics['f1']:.4f}")
    print(f"Candidate: {candidate_metrics['f1']:.4f}")

    print("\nRecall:")
    print(f"Baseline : {baseline_metrics['recall']:.4f}")
    print(f"Candidate: {candidate_metrics['recall']:.4f}")

    promote = should_promote(
        baseline_metrics,
        candidate_metrics,
    )

    print("\n=== DECISIÓN ===")

    if promote:
        update_candidate_status(
            candidate_version,
            "approved",
        )

        register_approved_candidate(candidate_version)

        print("Candidato APROBADO.")
        print("Registrado como candidato para despliegue.")
        print("Estado de despliegue: PENDING.")
    else:
        update_candidate_status(
            candidate_version,
            "rejected",
        )

        print("Candidato RECHAZADO.")
        print("Se mantiene el modelo actual en producción.")

    return promote


if __name__ == "__main__":
    main()
