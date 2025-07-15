import shutil
import os.path as osp
import subprocess
from subprocess import TimeoutExpired
import sys
import json
import re
import os
from dolphin_utils.prompts import *
import filecmp

MAX_ITERS = 4
MAX_RUNS = 5
MAX_STDERR_OUTPUT = 3000


# return (file, line, function, content), message
def info_traceback(stderr):
    pattern = r'File "(.*)", line (\d+), in (.+)\n (.*)'
    matches = re.findall(pattern, stderr)
    match = re.search(rf'\w*Error\w*(.*)', stderr, re.DOTALL)
    message = match.group(1).strip()
    externel = []
    for match in matches:
        if match[0].split('/')[-1] == 'experiment.py':
            continue
        else:
            externel.append(match)
    for e in externel:
        matches.remove(e)

    return matches, message


# RUN EXPERIMENT
def run_experiment(folder_name, run_num, timeout=18000):
    cwd = osp.abspath(folder_name)
    # COPY CODE SO WE CAN SEE IT.
    if osp.exists(osp.join(cwd, f"run_{run_num}")):
        shutil.copy(osp.join(cwd, "experiment.py"), osp.join(cwd, f"run_{run_num}", "experiment.py"))
    else:
        os.mkdir(osp.join(cwd, f"run_{run_num}"))
        shutil.copy(osp.join(cwd, "experiment.py"), osp.join(cwd, f"run_{run_num}", "experiment.py"))

    # LAUNCH COMMAND
    command = ["bash", "launcher.sh", f"run_{run_num}"]
    try:
        result = subprocess.run(
            command, cwd=cwd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True, timeout=timeout
        )

        if os.path.exists(osp.join(cwd, f"run_{run_num}", "final_info.json")):
            results = {}

            baseline_path = osp.join(cwd, "run_0", "final_info.json")
            if os.path.exists(baseline_path):
                with open(baseline_path, "r") as f:
                    baseline_data = json.load(f)
                baseline_results = {k: v["means"] for k, v in baseline_data.items()}
                results["baseline"] = baseline_results

            for run_idx in range(1, run_num + 1):
                run_path = osp.join(cwd, f"run_{run_idx}", "final_info.json")
                if os.path.exists(run_path):
                    with open(run_path, "r") as f:
                        run_data = json.load(f)
                    run_results = {k: v["means"] for k, v in run_data.items()}
                    results[f"improve_{run_idx}"] = run_results

            next_prompt = next_experiment_prompt.format(RUN_NUM=run_num, RESULTS=results, NEXT_RUN_NUM=run_num+1)
            traceback, message, tb = None, None, None
            return result.returncode, next_prompt, traceback, message

        if result.stderr:
            print(result.stderr, file=sys.stderr)
            with open(osp.join(cwd, f"run_{run_num}", "traceback.log"), "r") as file:
                tb = file.read()
            traceback, message = info_traceback(tb)
        else:
            traceback, message, tb = None, None, None

        if result.returncode != 0:
            print(f"Run {run_num} failed with return code {result.returncode}")
            if osp.exists(osp.join(cwd, f"run_{run_num}")):
                shutil.rmtree(osp.join(cwd, f"run_{run_num}"))
            print(f"Run failed with the following error {result.stderr}")
            if tb:
                stderr_output = tb
            else:
                stderr_output = result.stderr
            if len(stderr_output) > MAX_STDERR_OUTPUT:
                stderr_output = "..." + stderr_output[-MAX_STDERR_OUTPUT:]
            next_prompt = f"Run failed with the following error {stderr_output}"
        else:
            with open(osp.join(cwd, f"run_{run_num}", "final_info.json"), "r") as f:
                results = json.load(f)
            results = {k: v["means"] for k, v in results.items()}

            next_prompt = next_experiment_prompt.format(RUN_NUM=run_num, RESULTS=results, NEXT_RUN_NUM=run_num+1)

        return result.returncode, next_prompt, traceback, message
    except TimeoutExpired:
        print(f"Run {run_num} timed out after {timeout} seconds")
        if osp.exists(osp.join(cwd, f"run_{run_num}")):
            shutil.rmtree(osp.join(cwd, f"run_{run_num}"))
        next_prompt = f"Run timed out after {timeout} seconds"
        return 1, next_prompt, None, None


# PERFORM EXPERIMENTS
def perform_experiments(idea, folder_name, coder, baseline_results) -> bool:
    ## RUN EXPERIMENT
    current_iter = 0
    run = 1
    next_prompt = coder_prompt.format(
        title=idea["Title"],
        method=idea["Method"],
        idea=idea["Experiment"],
        max_runs=MAX_RUNS,
        baseline_results=baseline_results,
    )
    while run < MAX_RUNS + 1:
        if current_iter >= MAX_ITERS:
            print("Max iterations reached")
            break
        coder_out = coder.run(next_prompt)
        print(coder_out)
        if "litellm.BadRequestError" in coder_out:
            return False
        if "ALL_COMPLETED" in coder_out:
            break
        if filecmp.cmp(os.path.join(folder_name, 'experiment.py'), os.path.join(folder_name, 'run_0', 'experiment.py')):
            print("do not modify code")
            continue
        return_code, next_prompt, traceback, message = run_experiment(folder_name, run)
        # add traceback and code_structure
        if traceback:
            functions_codes = ""
            for t in traceback:
                functions_codes = functions_codes + f"line: {t[1]}, function: {t[2]}, codes: {t[3]} \n"
            code_structure = coder.run(code_structure_prompt_v2.format(error_messages=next_prompt, function_code=functions_codes))
            next_prompt = debug_prompt_with_structure_v2.format(error_messages=next_prompt, code_structure=code_structure)

        if return_code == 0:
            run += 1
            current_iter = 0
        current_iter += 1
    if current_iter >= MAX_ITERS:
        print("Not all experiments completed.")
        return False
    return True
