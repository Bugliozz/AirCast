"""
AriaPulita - Launcher

Uso:
    python main.py step1       # EDA
    python main.py step2       # Ingestione (MySQL + Neo4j + OpenSearch)
    python main.py step3       # Collector Flask server
    python main.py step3 collect  # Raccolta dati one-shot
    python main.py step4       # Training regressione PM10
    python main.py step5       # Training classificazione 4 classi
"""

import sys


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    step = sys.argv[1].lower()

    if step == "step1":
        from step_1.eda import run
        run()

    elif step == "step2":
        from step_2.ingest import run
        run()

    elif step == "step3":
        from step_3.main import main as step3_main
        step3_main()

    elif step == "step4":
        from step_4.train import run
        run()

    elif step == "step5":
        from step_5.train import run
        run()

    else:
        print(f"Step sconosciuto: {step}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
