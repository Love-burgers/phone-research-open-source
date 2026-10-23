"""Run the source checkout's six offline acceptance groups and save evidence."""

import argparse
from datetime import datetime, timezone
from importlib.resources import files
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


def summarize(report_path, categories, pytest_exit_code):
    """Count actual parameterized pytest cases, not labels or question samples."""
    root = ET.parse(report_path).getroot()
    rows = []
    unexpected = []
    for testcase in root.iter("testcase"):
        selector = testcase.get("classname", "").replace(".", "/") + ".py::" + testcase.get("name", "").split("[", 1)[0]
        owners = [group["id"] for group in categories if selector in group["tests"]]
        if len(owners) != 1:
            unexpected.append(selector)
        status = "passed"
        for tag in ("failure", "error", "skipped"):
            if testcase.find(tag) is not None:
                status = tag
                break
        rows.append({"category": owners[0] if len(owners) == 1 else None,
                     "selector": selector, "name": testcase.get("name"), "status": status})
    groups = []
    for group in categories:
        cases = [row for row in rows if row["category"] == group["id"]]
        observed = {row["selector"] for row in cases}
        missing = sorted(set(group["tests"]) - observed)
        groups.append({"id": group["id"], "name": group["name"], "expectation": group["expectation"],
                       "tests": len(cases), "passed_tests": sum(row["status"] == "passed" for row in cases),
                       "missing_selectors": missing,
                       "passed": bool(cases) and not missing and all(row["status"] == "passed" for row in cases)})
    return {"schema_version": 1, "suite": "phone_assistant_offline_acceptance",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "pytest_exit_code": pytest_exit_code, "category_count": len(groups),
            "test_count": len(rows), "passed_tests": sum(row["status"] == "passed" for row in rows),
            "passed": pytest_exit_code == 0 and bool(groups) and not unexpected and all(group["passed"] for group in groups),
            "categories": groups, "cases": rows, "unexpected_selectors": unexpected,
            "limitation": "离线固定档案、假搜索与MockTransport；验证业务行为，不是当前手机事实、自由表达理解或生成回答准确率。"}


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="源码目录六类选机助手验收；不访问真实外部服务")
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    project = Path(__file__).resolve().parents[1]
    if not (project / "tests").is_dir():
        parser.error("请在包含tests的源码检出中运行")
    categories = json.loads(files("evaluation").joinpath("chat_acceptance_cases.json").read_text(encoding="utf-8"))
    selectors = [selector for group in categories for selector in group["tests"]]
    if not categories or not selectors or len(set(selectors)) != len(selectors):
        raise ValueError("验收分类为空或存在重复测试归属")
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    junit = output / "pytest.xml"
    # conftest.py blocks real HTTP, search and DNS; every model path uses a fake.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    command = [sys.executable, "-m", "pytest", "-q", *selectors, f"--junitxml={junit}"]
    try:
        run = subprocess.run(command, cwd=project, env=env, capture_output=True,
                             text=True, encoding="utf-8", errors="replace", timeout=180)
    except subprocess.TimeoutExpired:
        (output / "summary.json").write_text(json.dumps({"passed": False, "error": "pytest_timeout"}), encoding="utf-8")
        print(f"验收超时，记录：{output}", file=sys.stderr)
        return 1
    (output / "pytest.log").write_text(run.stdout + run.stderr, encoding="utf-8")
    if junit.exists():
        result = summarize(junit, categories, run.returncode)
    else:
        result = {"passed": False, "pytest_exit_code": run.returncode, "error": "missing_pytest_report"}
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("passed", "category_count", "test_count", "passed_tests") if key in result}, ensure_ascii=False))
    for group in result.get("categories", []):
        print(f'{group["name"]}: {group["passed_tests"]}/{group["tests"]}，通过={group["passed"]}')
    print(f"完整结果：{output / 'summary.json'}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
