from maintenance.prepare_data import main as prepare_data
from maintenance.train import main as train_models
from maintenance.promotion import main as evaluate_promotion


def main():
    print("=" * 60)
    print("PIPELINE AUTOMATIZADO DE MANTENIMIENTO DEL MODELO")
    print("=" * 60)

    print("\n[1/3] PREPARACIÓN Y VALIDACIÓN DE DATOS")
    print("-" * 60)
    prepare_data()

    print("\n[2/3] REENTRENAMIENTO Y EVALUACIÓN")
    print("-" * 60)
    train_models()

    print("\n[3/3] POLÍTICA DE PROMOCIÓN")
    print("-" * 60)
    evaluate_promotion()

    print("\n" + "=" * 60)
    print("PIPELINE DE MANTENIMIENTO FINALIZADO")
    print("=" * 60)


if __name__ == "__main__":
    main()
