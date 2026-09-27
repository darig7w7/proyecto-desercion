import json
import joblib
import shutil
from datetime import datetime, timezone
from pathlib import Path


REGISTRY_DIR = Path("models/registry")
PRODUCTION_FILE = REGISTRY_DIR / "production.json"

ACTIVE_MODEL = Path("models/modelo_desercion.pkl")
ACTIVE_COLUMNS = Path("models/columnas.pkl")

BACKUP_DIR = Path("models/backups")


def load_metadata(version):
    """Carga los metadatos de una versión del registry."""
    metadata_path = REGISTRY_DIR / version / "metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"No existe metadata para la versión {version}"
        )

    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_active_artifacts():
    """
    Verifica que los artefactos activados puedan cargarse
    y sean compatibles con la API de predicción.
    """
    model = joblib.load(ACTIVE_MODEL)
    columns = joblib.load(ACTIVE_COLUMNS)

    if not hasattr(model, "predict"):
        raise ValueError(
            "El modelo activado no implementa predict()."
        )

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "El modelo activado no implementa predict_proba()."
        )

    if len(columns) == 0:
        raise ValueError(
            "El archivo de columnas está vacío."
        )

    return True


def validate_candidate(version):
    """
    Verifica que el candidato pueda ser desplegado.
    """
    metadata = load_metadata(version)

    if metadata.get("status") != "approved":
        raise ValueError(
            f"La versión {version} no está aprobada para despliegue."
        )

    version_dir = REGISTRY_DIR / version
    model_path = version_dir / "modelo.pkl"
    columns_path = version_dir / "columnas.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"No existe el modelo de la versión {version}"
        )

    if not columns_path.exists():
        raise FileNotFoundError(
            f"No existe el archivo de columnas de {version}"
        )

    return model_path, columns_path


def create_backup():
    """
    Guarda una copia del modelo actualmente activo
    antes de reemplazarlo.
    """
    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    backup_dir = BACKUP_DIR / timestamp
    backup_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(
        ACTIVE_MODEL,
        backup_dir / "modelo_desercion.pkl",
    )

    shutil.copy2(
        ACTIVE_COLUMNS,
        backup_dir / "columnas.pkl",
    )

    shutil.copy2(
        PRODUCTION_FILE,
        backup_dir / "production.json",
    )

    return backup_dir


def update_production_registry(version):
    """
    Actualiza el registro de la versión activa.
    """
    production = {
        "production_version": version,
        "model_file": "models/modelo_desercion.pkl",
        "columns_file": "models/columnas.pkl",
        "deployed_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(PRODUCTION_FILE, "w", encoding="utf-8") as f:
        json.dump(
            production,
            f,
            indent=2,
        )


def activate(version):
    """
    Activa una versión previamente aprobada.

    Flujo:
    1. Valida que el candidato esté aprobado.
    2. Crea un backup del modelo actual.
    3. Copia los nuevos artefactos.
    4. Valida que puedan cargarse correctamente.
    5. Actualiza el registro de producción.
    6. Si ocurre un error, restaura el backup.
    """
    print(f"Validando versión {version}...")

    model_path, columns_path = validate_candidate(version)

    print("Creando backup del modelo actual...")
    backup_dir = create_backup()

    try:
        print("Activando nuevo modelo...")

        shutil.copy2(
            model_path,
            ACTIVE_MODEL,
        )

        shutil.copy2(
            columns_path,
            ACTIVE_COLUMNS,
        )

        print("Validando artefactos activados...")
        validate_active_artifacts()

        update_production_registry(version)

    except Exception:
        print(
            "Error durante la activación. "
            "Ejecutando rollback..."
        )

        shutil.copy2(
            backup_dir / "modelo_desercion.pkl",
            ACTIVE_MODEL,
        )

        shutil.copy2(
            backup_dir / "columnas.pkl",
            ACTIVE_COLUMNS,
        )

        shutil.copy2(
            backup_dir / "production.json",
            PRODUCTION_FILE,
        )

        raise

    print(
        f"Versión {version} activada correctamente."
    )
    print(
        f"Backup disponible en: {backup_dir}"
    )

    return backup_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Activar una versión aprobada del modelo."
        )
    )

    parser.add_argument(
        "version",
        help=(
            "Versión del registry a activar. "
            "Ejemplo: v1.1.0"
        ),
    )

    args = parser.parse_args()

    activate(args.version)
