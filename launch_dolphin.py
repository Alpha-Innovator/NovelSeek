import openai
import os.path as osp
import shutil
import json
import argparse
import multiprocessing
import torch
import os
import time
import sys
from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput
from datetime import datetime
from dolphin_utils.generate_ideas import generate_ideas, check_idea_novelty
from dolphin_utils.experiments_utils import perform_experiments

NUM_REFLECTIONS = 3


def print_time():
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def parse_arguments():
    parser = argparse.ArgumentParser(description="Automatic Algorithm Design.")
    parser.add_argument("--skip-idea-generation", action="store_true", help="Skip idea generation and load existing ideas")
    parser.add_argument("--skip-novelty-check", action="store_true", help="Skip novelty check and use existing ideas")
    parser.add_argument("--experiment", type=str, default="point_classification_modelnet", help="Experiment to run AutoAD on.")
    parser.add_argument("--model", type=str, default="claude-3-5-sonnet-20240620", help="Model to use for AutoAD.")
    parser.add_argument("--code_model", type=str, default="deepseek", help="Model to use for experimental implementation.")
    parser.add_argument("--parallel", type=int, default=0, help="Number of parallel processes to run. 0 for sequential execution.")
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated list of GPU IDs to use (e.g., '0,1,2'). If not specified, all available GPUs will be used.")
    parser.add_argument("--rag", action="store_true", help="Use RAG to generate ideas.")
    parser.add_argument("--topic", type=str, default=None, help="Topic for RAG.")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed.")
    parser.add_argument("--max_papers", type=int, default=20, help="Number of rag papers.")
    parser.add_argument("--memory_papers", type=int, default=10, help="Use memory papers to generate the next query.")
    parser.add_argument("--num-ideas", type=int, default=20, help="Number of ideas to generate")
    parser.add_argument("--check_similarity", action="store_true", help="check similarity when generate ideas")
    parser.add_argument("--embedding_model", type=str, default="sentence-transformers/all-roberta-large-v1", help="embedding model to check similarity")
    parser.add_argument("--round", type=int, default=0, help="Round of experiments")
    parser.add_argument("--save_name", type=str, default=None, help="Result dir (default: results/exp_name)")
    return parser.parse_args()


def get_available_gpus(gpu_ids=None):
    if gpu_ids is not None:
        return [int(gpu_id) for gpu_id in gpu_ids.split(',')]
    return list(range(torch.cuda.device_count()))


def worker(queue, base_dir, results_dir, model, client, client_model, writeup, improvement, gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Worker {gpu_id} started.")
    while True:
        idea = queue.get()
        if idea is None:
            break
        success = do_idea(
            base_dir, results_dir, idea, model, client, client_model, writeup, improvement, log_file=True
        )
        print(f"Completed idea: {idea['Name']}, Success: {success}")
    print(f"Worker {gpu_id} finished.")


def do_idea(base_dir, results_dir, idea, model, log_file=False):
    ## CREATE PROJECT FOLDER
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    idea_name = f"{timestamp}_{idea['Name']}"
    folder_name = osp.join(results_dir, idea_name)
    assert not osp.exists(folder_name), f"Folder {folder_name} already exists."
    destination_dir = folder_name
    shutil.copytree(base_dir, destination_dir, dirs_exist_ok=True)
    with open(osp.join(base_dir, "run_0", "final_info.json"), "r") as f:
        baseline_results = json.load(f)
    baseline_results = {k: v["means"] for k, v in baseline_results.items()}
    exp_file = osp.join(folder_name, "experiment.py")
    notes = osp.join(folder_name, "notes.txt")
    with open(notes, "w") as f:
        f.write(f"# Title: {idea['Title']}\n")
        f.write(f"# Experiment description: {idea['Experiment']}\n")
        f.write(f"## Run 0: Baseline\n")
        f.write(f"Results: {baseline_results}\n")
        f.write(f"Description: Baseline results.\n")
    if log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        log_path = osp.join(folder_name, "log.txt")
        log = open(log_path, "a")
        sys.stdout = log
        sys.stderr = log
    try:
        print_time()
        print(f"*Starting idea: {idea_name}*")
        ## PERFORM EXPERIMENTS
        # fnames = [model_file, train_file, notes]
        fnames = [exp_file, notes]
        io = InputOutput(yes=True, chat_history_file=f"{folder_name}/{idea_name}_aider.txt")
    
        if model.startswith("deepseek"):
            main_model = Model("deepseek/deepseek-coder")
        elif model.startswith("localhost"):
            ollama_model = "ollama/" + "-".join(model.split("-")[1:])
            main_model = Model(ollama_model)
        else:
            main_model = Model(model)
        coder = Coder.create(
            main_model=main_model, fnames=fnames, io=io, stream=False, use_git=False, edit_format="diff"
        )

        print_time()
        print(f"*Starting Experiments*")
        try:
            success = perform_experiments(idea, folder_name, coder, baseline_results)
        except Exception as e:
            print(f"Error during experiments: {e}")
            print(f"Experiments failed for idea {idea_name}")
            return False

        if not success:
            print(f"Experiments failed for idea {idea_name}")
            return False

        print_time()
    except Exception as e:
        print(f"Failed to evaluate idea {idea_name}: {str(e)}")
        return False
    finally:
        print("FINISHED IDEA")
        if log_file:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log.close()


def check_results(exp_base_dir):
    all_exp = os.listdir(exp_base_dir)

if __name__ == "__main__":
    args = parse_arguments()

    # Check available GPUs and adjust parallel processes if necessary
    available_gpus = get_available_gpus(args.gpus)
    if args.parallel > len(available_gpus):
        print(
            f"Warning: Requested {args.parallel} parallel processes, but only {len(available_gpus)} GPUs available. Adjusting to {len(available_gpus)}."
        )
        args.parallel = len(available_gpus)

    print(f"Using GPUs: {available_gpus}")

    # Create client
    if "claude" in args.model:
        import anthropic
        print(f"Using Anthropic API with model {args.model}.")
        client_model = args.model
        client = anthropic.Anthropic()
    elif "gpt" in args.model:
        import openai
        print(f"Using OpenAI API with model {args.model}.")
        client_model = args.model
        client = openai.OpenAI()
    elif "deepseek" in args.model:
        import openai
        print(f"Using DeepSeek API with {args.model}.")
        client_model = args.model
        client = openai.OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com"
        )
    elif args.model.startswith("localhost"):
        import openai
        print(f"Using OpenAI API with locally deployed model: {args.model}.")
        client_model = args.model
        client = openai.OpenAI(base_url="http://localhost:11434/v1", api_key="na")
    else:
        raise ValueError(f"Model {args.model} is not supported. You need to add the model to autoad_utils/llm_utils.py and launch.py manually")

    base_dir = osp.join("examples", args.experiment)
    if args.save_name:
        results_dir = osp.join("results", args.save_name)
    else:
        results_dir = osp.join("results", args.experiment)

    exp_base_file_list = None

    if args.rag:
        from dolphin_utils.rag_tools.lit_review import collect_papers
        assert args.topic is not None
        paper_bank, total_cost, all_queries = collect_papers(args.topic, client, client_model, args.seed, args.memory_papers, args.max_papers)
        paper_dict = {"topic_description": args.topic, "all_queries": all_queries, "paper_bank": paper_bank}
        with open(osp.join(base_dir, f"{args.experiment}_rag_papers.json"), "w") as f:
            json.dump(paper_dict, f, indent=4)

    ideas = generate_ideas(
        base_dir,
        client=client,
        model=client_model,
        skip_generation=args.skip_idea_generation,
        max_num_generations=args.num_ideas,
        num_reflections=NUM_REFLECTIONS,
        rag=args.rag,
        rag_path=osp.join(base_dir, f"{args.experiment}_rag_papers.json"),
        check_independence=args.check_similarity,
        embedding_model=args.embedding_model,
        round=args.round,
        exp_base_file_list=exp_base_file_list,
    )
    ideas = check_idea_novelty(
        ideas,
        base_dir=base_dir,
        client=client,
        model=client_model,
        round=args.round
    )

    filter_ideas = [idea for idea in ideas if idea['independence']]
    novel_ideas = [idea for idea in filter_ideas if idea["novel"]]
    print(f"Run experiments on {len(novel_ideas)} ideas.")

    if args.parallel > 0:
        print(f"Running {args.parallel} parallel processes")
        queue = multiprocessing.Queue()
        for idea in novel_ideas:
            queue.put(idea)

        processes = []
        for i in range(args.parallel):
            gpu_id = available_gpus[i % len(available_gpus)]
            p = multiprocessing.Process(
                target=worker,
                args=(
                    queue,
                    base_dir,
                    results_dir,
                    args.code_model,
                    gpu_id,
                )
            )
            p.start()
            time.sleep(150)
            processes.append(p)

        # Signal workers to exit
        for _ in range(args.parallel):
            queue.put(None)

        for p in processes:
            p.join()

        print("All parallel processes completed.")
    else:
        for idea in novel_ideas:
            print(f"Processing idea: {idea['Name']}")
            try:
                success = do_idea(
                    base_dir,
                    results_dir,
                    idea,
                    args.code_model
                )
                print(f"Completed idea: {idea['Name']}, Success: {success}")
            except Exception as e:
                print(f"Failed to evaluate idea {idea['Name']}: {str(e)}")

    print("All ideas evaluated.")
