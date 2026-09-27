import json
from pathlib import Path


REGISTRY_DIR = Path("models/registry")


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
    """Actualiza el estado del modelo candidato."""
    path = REGISTRY_DIR / version / "metadata.json"

    with open(path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    metadata["status"] = status

    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


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

        print("Candidato APROBADO para promoción.")
    else:
        update_candidate_status(
            candidate_version,
            "rejected",
        )

        print("Candidato RECHAZADO.")
        print("Se mantiene el modelo actual en producción.")


if __name__ == "__main__":
    main()

