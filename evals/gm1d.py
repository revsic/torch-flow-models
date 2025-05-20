import json
from datetime import timedelta, timezone, datetime
from pathlib import Path

from tqdm.auto import tqdm

from testbed import TestGaussianMixture1D
from testutils import inherit_testbed


MODELS = inherit_testbed(TestGaussianMixture1D)


def main():
    PATH = Path("./test.gm1d")
    STAMP = datetime.now(timezone(timedelta(hours=9))).strftime("%Y.%m.%d.KST%H:%M:%S")

    path = PATH / STAMP
    path.mkdir(exist_ok=True, parents=True)

    results = {}
    with tqdm(MODELS.items(), total=len(MODELS)) as pbar:
        for name, suite in pbar:
            pbar.set_description_str(name)
            p = path / name
            p.mkdir(exist_ok=True)
            result = suite.test(p)
            results[name] = result

    with open(path / "result.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
