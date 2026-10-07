"""Command-line interface: ``heartrisk <command> --help``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import __version__
from .config import load_config

DEFAULT_BUNDLE = "models/heartrisk.joblib"


def _cfg(args) -> dict:
    cfg = load_config(args.config, args.set)
    if getattr(args, "quick", False):
        from .study import quick_config

        cfg = quick_config(cfg)
    return cfg


def cmd_study(args) -> int:
    from .study import run_study

    run_study(_cfg(args), args.out, args.results, args.bundle, n_jobs=args.jobs, reuse=args.reuse)
    return 0


def cmd_evaluate(args) -> int:
    from .data import load_dataset
    from .evaluation import cross_validate
    from .study import md_table

    cfg = _cfg(args)
    X, y, spec = load_dataset(args.dataset or cfg["dataset"], cfg.get("data_dir", "data"))
    models = args.models.split(",") if args.models else cfg["evaluation"]["models"]
    res = cross_validate(X, y, spec, cfg, models, n_jobs=args.jobs, verbose=args.verbose)
    res.save(args.out)
    s = res.summary(["roc_auc", "brier", "ece", "f1", "sensitivity", "specificity"])
    cols = [
        "model",
        "roc_auc",
        "roc_auc_lo",
        "roc_auc_hi",
        "brier",
        "ece",
        "f1",
        "sensitivity",
        "specificity",
    ]
    print(md_table(s[cols]))
    base = cfg["evaluation"]["baseline"]
    if base in models and len(models) > 1:
        print("\nPaired corrected t-test vs", base)
        print(md_table(res.compare(base)[["model", "mean_diff", "ci_lo", "ci_hi", "p", "p_holm"]], ".4f"))
    return 0


def cmd_train(args) -> int:
    from .train import train_bundle

    cfg = _cfg(args)
    bundle = train_bundle(cfg, args.model, args.calibration)
    path = bundle.save(args.out)
    print(
        json.dumps(
            {"bundle": str(path), "threshold": bundle.threshold, **bundle.performance}, indent=2, default=str
        )
    )
    return 0


def _read_records(path: str) -> pd.DataFrame:
    if path == "-":
        payload = json.load(sys.stdin)
    elif path.endswith(".json"):
        payload = json.loads(Path(path).read_text())
    else:
        return pd.read_csv(path)
    return pd.DataFrame(payload if isinstance(payload, list) else [payload])


def cmd_predict(args) -> int:
    from .bundle import ModelBundle
    from .predict import predict_frame

    bundle = ModelBundle.load(args.bundle)
    df = _read_records(args.input)
    out = pd.concat([df, predict_frame(bundle, df)], axis=1)
    if args.output:
        out.to_csv(args.output, index=False)
        print(f"wrote {len(out)} predictions to {args.output}")
    else:
        print(out[["probability", "prediction", "risk_band", "warnings"]].to_string())
    return 0


def cmd_explain(args) -> int:
    from .bundle import ModelBundle
    from .explain import explain_patient
    from .predict import prepare

    bundle = ModelBundle.load(args.bundle)
    df = _read_records(args.input).iloc[[0]]
    table = explain_patient(bundle, prepare(bundle, df))
    print(f"baseline {table.attrs['base_value']:.3f} -> prediction {table.attrs['probability']:.3f}")
    print(table.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    from .api import create_app
    from .bundle import ModelBundle

    uvicorn.run(create_app(ModelBundle.load(args.bundle)), host=args.host, port=args.port)
    return 0


def cmd_info(args) -> int:
    from .bundle import ModelBundle

    print(json.dumps(ModelBundle.load(args.bundle).card(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="heartrisk", description=__doc__)
    p.add_argument("--version", action="version", version=f"heartrisk {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, jobs=True):
        sp.add_argument("--config", help="YAML merged over the packaged defaults")
        sp.add_argument(
            "--set",
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help="override a config value, e.g. --set evaluation.repeats=10",
        )
        sp.add_argument("--quick", action="store_true", help="tiny settings for smoke tests")
        if jobs:
            sp.add_argument("-j", "--jobs", type=int, default=-1, help="parallel folds (default: all cores)")

    sp = sub.add_parser("study", help="run the full study and write reports/REPORT.md")
    common(sp)
    sp.add_argument("--out", default="reports")
    sp.add_argument("--results", default="results")
    sp.add_argument("--bundle", default=DEFAULT_BUNDLE)
    sp.add_argument(
        "--reuse", action="store_true", help="reuse cached nested-CV results in --results if they match"
    )
    sp.set_defaults(func=cmd_study)

    sp = sub.add_parser("evaluate", help="nested repeated CV of selected models")
    common(sp)
    sp.add_argument("--dataset", help="heart | cleveland | framingham")
    sp.add_argument("--models", help="comma-separated, e.g. logreg,xgb")
    sp.add_argument("--out", default="results/evaluate")
    sp.add_argument("-v", "--verbose", type=int, default=0)
    sp.set_defaults(func=cmd_evaluate)

    sp = sub.add_parser("train", help="fit the deployable bundle on all data")
    common(sp, jobs=False)
    sp.add_argument("--model", help="model key (default: deploy.model)")
    sp.add_argument("--calibration", choices=["none", "sigmoid", "isotonic"])
    sp.add_argument("--out", default=DEFAULT_BUNDLE)
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("predict", help="score a CSV / JSON file ('-' for stdin)")
    sp.add_argument("input")
    sp.add_argument("--bundle", default=DEFAULT_BUNDLE)
    sp.add_argument("-o", "--output")
    sp.set_defaults(func=cmd_predict)

    sp = sub.add_parser("explain", help="Shapley explanation for the first record of a file")
    sp.add_argument("input")
    sp.add_argument("--bundle", default=DEFAULT_BUNDLE)
    sp.set_defaults(func=cmd_explain)

    sp = sub.add_parser("serve", help="start the REST API")
    sp.add_argument("--bundle", default=DEFAULT_BUNDLE)
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8000)
    sp.set_defaults(func=cmd_serve)

    sp = sub.add_parser("info", help="print the model card of a bundle")
    sp.add_argument("--bundle", default=DEFAULT_BUNDLE)
    sp.set_defaults(func=cmd_info)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
