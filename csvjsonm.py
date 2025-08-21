import json
import csv
import numpy as np

def log_results(mode: str, params: dict, results: dict):
    import logging
    safe_params={k: str(v) for k, v in params.items()}
    safe_results={k: str(v) for k, v in results.items()}
    logging.info(f"Mode: {mode} | Params: {safe_params} | Results: {safe_results}")

def _to_native(x):
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x

def _normalize_dict(d: dict) -> dict:
    return {str(k): _to_native(v) for k, v in d.items()}

def save_results(result: dict, filename: str, fmt: str = "csv"):
    safe = _normalize_dict(result)

    if fmt.lower() == "csv":
        with open(f"{filename}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Metric", "Value"])
            for k, v in safe.items():
                w.writerow([k, v])
        print(f"Results saved to {filename}.csv")

    elif fmt.lower() == "json":
        with open(f"{filename}.json", "w", encoding="utf-8") as f:
            json.dump(safe, f, indent=4)
        print(f"Results saved to {filename}.json")

    else:
        raise ValueError("Unsupported format. Use 'csv' or 'json'.")