import json
import os
import os.path as osp
import time
import torch
from typing import List, Dict, Union
from dolphin_utils.llm_utils import get_response_from_llm, extract_json_between_markers
from dolphin_utils.rag_utils import format_papers_for_printing, format_papers_for_printing_ai_researcher
from dolphin_utils.prompts import *

import requests
import backoff
import torch.nn.functional as F

S2_API_KEY = os.getenv("S2_API_KEY")

history_ideas_bank = []
history_ideas_id = []

negative_ideas_bank = []

# GENERATE IDEAS
def generate_ideas(
        base_dir,
        client,
        model,
        skip_generation=False,
        max_num_generations=20,
        num_reflections=5,
        rag=False,
        rag_path=None,
        check_independence=False,
        embedding_model=None,
        round=0,
        exp_base_file_list=None
):
    global history_ideas_bank, history_ideas_id
    if len(history_ideas_bank) != 0:
        history_ideas_bank, history_ideas_id = [], []
    total_price = 0
    if skip_generation:
        # Load existing ideas from file
        try:
            with open(osp.join(base_dir, "ideas.json"), "r") as f:
                ideas = json.load(f)
            print("Loaded existing ideas:")
            for idea in ideas:
                print(idea)
            return ideas
        except FileNotFoundError:
            print("No existing ideas found. Generating new ideas.")
        except json.JSONDecodeError:
            print("Error decoding existing ideas. Generating new ideas.")

    if check_independence:
        from transformers import AutoTokenizer, AutoModel
        sentence_tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        sentence_model = AutoModel.from_pretrained(embedding_model)
        independence_list = []
        history_ideas_list = []

    idea_str_archive = []
    with open(osp.join(base_dir, "seed_ideas.json"), "r") as f:
        seed_ideas = json.load(f)

    # only one seed idea
    if check_independence:
        for seed_idea in seed_ideas:
            independence = check_idea_independence(seed_idea, history_ideas_list, sentence_tokenizer, sentence_model)
            idea_str_archive.append(json.dumps(seed_idea))
            independence_list.append(independence)
            history_ideas_list.append(seed_idea)
    else:
        for seed_idea in seed_ideas:
            idea_str_archive.append(json.dumps(seed_idea))

    if round > 0:
        idea_str_archive_pos = []
        assert check_independence == True
        p_list, n_list, m_list, e_list = [], [], [], []
        for dir in exp_base_file_list[0]:
            p_temp, n_temp, m_temp, e_temp = check_results(dir)
            p_list += p_temp
            n_list += n_temp
            m_list += m_temp
            e_list += e_temp
        p_list = ['_'.join(name.split('_')[2:]) for name in p_list]
        n_list = ['_'.join(name.split('_')[2:]) for name in n_list]
        m_list = ['_'.join(name.split('_')[2:]) for name in m_list]

        previous_ideas = []
        for file in exp_base_file_list[1]:
            previous_ideas += json.load(open(file))

        previous_summary = []
        for p_idea in previous_ideas:
            if p_idea['Name'] in (n_list + m_list):
                previous_summary.append(encode_sentence(p_idea['Summary'], sentence_model, sentence_tokenizer))
                history_ideas_list.append(p_idea)
                idea_str_archive.append(json.dumps(p_idea))
            elif p_idea['Name'] in p_list:
                idea_str_archive_pos.append(json.dumps(p_idea))
                idea_str_archive.append(json.dumps(p_idea))

        history_ideas_bank += previous_summary
        history_ideas_id += [len(history_ideas_id) + i for i in list(range(len(previous_summary)))]

    with open(osp.join(base_dir, "experiment.py"), "r") as f:
        code = f.read()

    with open(osp.join(base_dir, "prompt.json"), "r") as f:
        prompt = json.load(f)

    if rag:
        with open(osp.join(rag_path), "r") as f:
            rag_papers = json.load(f)
            rag_reference = format_papers_for_printing_ai_researcher(rag_papers)

    idea_system_prompt = prompt["system"]

    for _ in range(max_num_generations):
        print()
        print(f"Generating idea {_ + 1}/{max_num_generations}")
        try:
            prev_ideas_string = "\n\n".join(idea_str_archive)

            msg_history = []
            print(f"Iteration 1/{num_reflections}")
            if round == 0:
                if rag:
                    text, msg_history, price = get_response_from_llm(
                        idea_first_prompt_with_rag.format(
                            task_description=prompt["task_description"],
                            rag_reference=rag_reference,
                            code=code,
                            prev_ideas_string=prev_ideas_string,
                            num_reflections=num_reflections,
                        ),
                        client=client,
                        model=model,
                        system_message=idea_system_prompt,
                        msg_history=msg_history,
                    )
                    total_price += price
                else:
                    text, msg_history, price = get_response_from_llm(
                        idea_first_prompt.format(
                            task_description=prompt["task_description"],
                            code=code,
                            prev_ideas_string=prev_ideas_string,
                            num_reflections=num_reflections,
                        ),
                        client=client,
                        model=model,
                        system_message=idea_system_prompt,
                        msg_history=msg_history,
                    )
                    total_price += price

            elif round > 0:
                prev_ideas_string_pos = "\n\n".join(idea_str_archive_pos)

                if rag:
                    text, msg_history, price = get_response_from_llm(
                        idea_first_prompt_with_rag_loop.format(
                            task_description=prompt["task_description"],
                            rag_reference=rag_reference,
                            code=code,
                            prev_ideas_string=prev_ideas_string,
                            prev_ideas_string_pos=prev_ideas_string_pos,
                            num_reflections=num_reflections,
                        ),
                        client=client,
                        model=model,
                        system_message=idea_system_prompt,
                        msg_history=msg_history,
                    )
                    total_price += price
                else:
                    text, msg_history, price = get_response_from_llm(
                        idea_first_prompt.format(
                            task_description=prompt["task_description"],
                            code=code,
                            prev_ideas_string=prev_ideas_string,
                            num_reflections=num_reflections,
                        ),
                        client=client,
                        model=model,
                        system_message=idea_system_prompt,
                        msg_history=msg_history,
                    )
                    total_price += price
            ## PARSE OUTPUT
            json_output = extract_json_between_markers(text)
            assert json_output is not None, "Failed to extract JSON from LLM output"
            print(json_output)

            # Iteratively improve task.
            if num_reflections > 1:
                for j in range(num_reflections - 1):
                    print(f"Iteration {j + 2}/{num_reflections}")
                    text, msg_history, price = get_response_from_llm(
                        idea_reflection_prompt.format(
                            current_round=j + 2, num_reflections=num_reflections
                        ),
                        client=client,
                        model=model,
                        system_message=idea_system_prompt,
                        msg_history=msg_history,
                    )
                    total_price += price
                    ## PARSE OUTPUT
                    json_output = extract_json_between_markers(text)
                    assert (
                            json_output is not None
                    ), "Failed to extract JSON from LLM output"
                    print(json_output)

                    if "I am done" in text:
                        print(f"Idea generation converged after {j + 2} iterations.")
                        break

            if check_independence:
                independence = check_idea_independence(json_output, history_ideas_list, sentence_tokenizer,
                                                       sentence_model)
                idea_str_archive.append(json.dumps(json_output))
                independence_list.append(independence)
                history_ideas_list.append(json_output)
            else:
                idea_str_archive.append(json.dumps(json_output))
        except Exception as e:
            print(f"Failed to generate idea: {e}")
            continue

    ## SAVE IDEAS
    ideas = []
    for idea_str in idea_str_archive:
        ideas.append(json.loads(idea_str))

    if round > 0:
        ideas = [ideas[0]] + ideas[-len(independence_list) + 1:]

    if check_independence:
        for idea, independence in zip(ideas, independence_list):
            idea.update({
                'independence': independence
            })

    if round > 0:
        ideas = ideas[len(seed_ideas):]

    with open(osp.join(base_dir, f"ideas_round_{round}_with_pos.json"), "w") as f:
        json.dump(ideas, f, indent=4)

    total_cost = f"Generate {max_num_generations} ideas, the total cost is ${total_price}, ${total_price / max_num_generations} per idea."
    with open(osp.join(base_dir, "cost.txt"), "w") as f:
        f.write(total_cost)

    return ideas


# GENERATE IDEAS OPEN-ENDED
def generate_next_idea(
        base_dir,
        client,
        model,
        prev_idea_archive=[],
        num_reflections=5,
        max_attempts=10,
):
    idea_archive = prev_idea_archive
    original_archive_size = len(idea_archive)

    print(f"Generating idea {original_archive_size + 1}")

    if len(prev_idea_archive) == 0:
        print(f"First iteration, taking seed ideas")
        # seed the archive on the first run with pre-existing ideas
        with open(osp.join(base_dir, "seed_ideas.json"), "r") as f:
            seed_ideas = json.load(f)
        for seed_idea in seed_ideas[:1]:
            idea_archive.append(seed_idea)
    else:
        with open(osp.join(base_dir, "experiment.py"), "r") as f:
            code = f.read()
        with open(osp.join(base_dir, "prompt.json"), "r") as f:
            prompt = json.load(f)
        idea_system_prompt = prompt["system"]

        for _ in range(max_attempts):
            try:
                idea_strings = []
                for idea in idea_archive:
                    idea_strings.append(json.dumps(idea))
                prev_ideas_string = "\n\n".join(idea_strings)

                msg_history = []
                print(f"Iteration 1/{num_reflections}")
                text, msg_history, price = get_response_from_llm(
                    idea_first_prompt.format(
                        task_description=prompt["task_description"],
                        code=code,
                        prev_ideas_string=prev_ideas_string,
                        num_reflections=num_reflections,
                    )
                    + """
Completed ideas have an additional "Score" field which indicates the assessment by an expert ML reviewer.
This is on a standard 1-10 ML conference scale.
Scores of 0 indicate the idea failed either during experimentation, writeup or reviewing.
""",
                    client=client,
                    model=model,
                    system_message=idea_system_prompt,
                    msg_history=msg_history,
                )
                ## PARSE OUTPUT
                json_output = extract_json_between_markers(text)
                assert json_output is not None, "Failed to extract JSON from LLM output"
                print(json_output)

                # Iteratively improve task.
                if num_reflections > 1:
                    for j in range(num_reflections - 1):
                        print(f"Iteration {j + 2}/{num_reflections}")
                        text, msg_history, price = get_response_from_llm(
                            idea_reflection_prompt.format(
                                current_round=j + 2, num_reflections=num_reflections
                            ),
                            client=client,
                            model=model,
                            system_message=idea_system_prompt,
                            msg_history=msg_history,
                        )
                        ## PARSE OUTPUT
                        json_output = extract_json_between_markers(text)
                        assert (
                                json_output is not None
                        ), "Failed to extract JSON from LLM output"
                        print(json_output)

                        if "I am done" in text:
                            print(
                                f"Idea generation converged after {j + 2} iterations."
                            )
                            break

                idea_archive.append(json_output)
                break
            except Exception as e:
                print(f"Failed to generate idea: {e}")
                continue

    ## SAVE IDEAS
    with open(osp.join(base_dir, "ideas.json"), "w") as f:
        json.dump(idea_archive, f, indent=4)

    return idea_archive


def on_backoff(details):
    print(
        f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries "
        f"calling function {details['target'].__name__} at {time.strftime('%X')}"
    )


@backoff.on_exception(
    backoff.expo, requests.exceptions.HTTPError, on_backoff=on_backoff
)
def search_for_papers(query, result_limit=10) -> Union[None, List[Dict]]:
    if not query:
        return None
    rsp = requests.get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        headers={"X-API-KEY": S2_API_KEY},
        params={
            "query": query,
            "limit": result_limit,
            "fields": "title,authors,venue,year,abstract,citationStyles,citationCount",
        },
    )
    print(f"Response Status Code: {rsp.status_code}")
    print(
        f"Response Content: {rsp.text[:500]}"
    )  # Print the first 500 characters of the response content
    rsp.raise_for_status()
    results = rsp.json()
    total = results["total"]
    time.sleep(1.0)
    if not total:
        return None

    papers = results["data"]
    return papers


novelty_system_msg = """You are an ambitious AI PhD student who is looking to publish a paper that will contribute significantly to the field.
You have an idea and you want to check if it is novel or not. I.e., not overlapping significantly with existing literature or already well explored.
Be a harsh critic for novelty, ensure there is a sufficient contribution in the idea for a new conference or workshop paper.
You will be given access to the Semantic Scholar API, which you may use to survey the literature and find relevant papers to help you make your decision.
The top 10 results for any search query will be presented to you with the abstracts.

You will be given {num_rounds} to decide on the paper, but you do not need to use them all.
At any round, you may exit early and decide on the novelty of the idea.
Decide a paper idea is novel if after sufficient searching, you have not found a paper that significantly overlaps with your idea.
Decide a paper idea is not novel, if you have found a paper that significantly overlaps with your idea.

{task_description}
<experiment.py>
{code}
</experiment.py>
"""

novelty_prompt = '''Round {current_round}/{num_rounds}.
You have this idea:

"""
{idea}
"""

The results of the last query are (empty on first round):
"""
{last_query_results}
"""

Respond in the following format:

THOUGHT:
<THOUGHT>

RESPONSE:
```json
<JSON>
```

In <THOUGHT>, first briefly reason over the idea and identify any query that could help you make your decision.
If you have made your decision, add "Decision made: novel." or "Decision made: not novel." to your thoughts.

In <JSON>, respond in JSON format with ONLY the following field:
- "Query": An optional search query to search the literature (e.g. attention is all you need). You must make a query if you have not decided this round.

A query will work best if you are able to recall the exact name of the paper you are looking for, or the authors.
This JSON will be automatically parsed, so ensure the format is precise.'''


def check_idea_novelty(
        ideas,
        base_dir,
        client,
        model,
        max_num_iterations=10,
        round=0
):
    total_price = 0
    with open(osp.join(base_dir, "experiment.py"), "r") as f:
        code = f.read()
    with open(osp.join(base_dir, "prompt.json"), "r") as f:
        prompt = json.load(f)
        task_description = prompt["task_description"]

    for idx, idea in enumerate(ideas):
        if "novel" in idea:
            print(f"Skipping idea {idx}, already checked.")
            continue

        print(f"\nChecking novelty of idea {idx}: {idea['Name']}")

        novel = False
        msg_history = []
        papers_str = ""

        for j in range(max_num_iterations):
            try:
                text, msg_history, price = get_response_from_llm(
                    novelty_prompt.format(
                        current_round=j + 1,
                        num_rounds=max_num_iterations,
                        idea=idea,
                        last_query_results=papers_str,
                    ),
                    client=client,
                    model=model,
                    system_message=novelty_system_msg.format(
                        num_rounds=max_num_iterations,
                        task_description=task_description,
                        code=code,
                    ),
                    msg_history=msg_history,
                )
                total_price += price
                if "decision made: novel" in text.lower():
                    print("Decision made: novel after round", j)
                    novel = True
                    break
                if "decision made: not novel" in text.lower():
                    print("Decision made: not novel after round", j)
                    break

                ## PARSE OUTPUT
                json_output = extract_json_between_markers(text)
                assert json_output is not None, "Failed to extract JSON from LLM output"

                ## SEARCH FOR PAPERS
                query = json_output["Query"]
                papers = search_for_papers(query, result_limit=10)
                if papers is None:
                    papers_str = "No papers found."

                paper_strings = []
                for i, paper in enumerate(papers):
                    paper_strings.append(
                        """{i}: {title}. {authors}. {venue}, {year}.\nNumber of citations: {cites}\nAbstract: {abstract}""".format(
                            i=i,
                            title=paper["title"],
                            authors=paper["authors"],
                            venue=paper["venue"],
                            year=paper["year"],
                            cites=paper["citationCount"],
                            abstract=paper["abstract"],
                        )
                    )
                papers_str = "\n\n".join(paper_strings)

            except Exception as e:
                print(f"Error: {e}")
                continue

        idea["novel"] = novel

    # Save results to JSON file
    results_file = osp.join(base_dir, f"ideas_round_{round}_with_pos.json")
    with open(results_file, "w") as f:
        json.dump(ideas, f, indent=4)

    total_cost = f"The total cost of ideas novelty check is ${total_price}."
    with open(osp.join(base_dir, "cost.txt"), "a") as f:
        f.write(total_cost)

    return ideas


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def encode_sentence(sentence, model, tokenizer):
    encoded_sentence = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
    attention_mask = encoded_sentence['attention_mask']
    with torch.no_grad():
        encoded_sentence = model(**encoded_sentence)
    encoded_sentence = mean_pooling(encoded_sentence, attention_mask)
    return encoded_sentence


def check_idea_independence(new_idea, ideas_history, tokenizer, model, threshold=0.8):
    independence = True
    if len(history_ideas_bank) == 0:
        encoded_ideas = encode_sentence(new_idea['Summary'], model, tokenizer)
        history_ideas_bank.append(encoded_ideas)
        history_ideas_id.append(0)
    else:
        encoded_history_ideas = torch.cat(history_ideas_bank, dim=0)
        encoded_new_idea = encode_sentence(new_idea['Summary'], model, tokenizer)
        similarity = F.normalize(encoded_new_idea, dim=-1) @ F.normalize(encoded_history_ideas, dim=-1).T
        if similarity.max() > threshold:
            print(
                f"The idea: {new_idea['Title']} is similar to {ideas_history[history_ideas_id[similarity.argmax()]]['Title']}.")
            independence = False
        else:
            history_ideas_bank.append(encoded_new_idea)
            history_ideas_id.append(len(ideas_history))
    return independence


def check_results(exp_base_dir):
    pos_idea_list, neg_idea_list, med_idea_list = [], [], []
    error_implement_list = []
    all_exp = os.listdir(exp_base_dir)

    def get_res_from_dict(results_dict):
        if isinstance(results_dict, dict):
            for k, v in results_dict.items():
                if k == 'means':
                    return v
                elif isinstance(v, dict):
                    res = get_res_from_dict(v)
                    if res is not None:
                        return res
        return None

    for exp in all_exp:
        performance = None
        run_dir = []
        for d in os.listdir(os.path.join(exp_base_dir, exp)):
            if d.split('_')[0] == 'run' and os.path.isdir(os.path.join(exp_base_dir, exp, d)): run_dir.append(d)
        if len(run_dir) == 1:
            error_implement_list.append(exp)
            continue  # only run_0 dir
        else:
            base_results = json.load(open(os.path.join(exp_base_dir, exp, 'run_0/final_info.json'), 'r'))
            base_res = get_res_from_dict(base_results)
        for d in run_dir:
            if d == 'run_0':
                continue
            else:
                try:
                    cur_results = json.load(open(os.path.join(exp_base_dir, exp, d, 'final_info.json'), 'r'))
                    cur_res = get_res_from_dict(cur_results)
                except:
                    cur_res = [0] * len(base_res)
            if all(b_res < c_res for b_res, c_res in zip(base_res.values(), cur_res.values())):
                performance = 'higher'
                continue
            elif all(b_res > c_res for b_res, c_res in zip(base_res.values(), cur_res.values())):
                if performance == None:
                    performance = 'lower'
            else:
                performance = 'medium'
        if performance == 'higher':
            pos_idea_list.append(exp)
        elif performance == 'lower':
            neg_idea_list.append(exp)
        elif performance == 'medium':
            med_idea_list.append(exp)

    assert len(pos_idea_list) + len(neg_idea_list) + len(med_idea_list) + len(error_implement_list) == len(all_exp)
    return pos_idea_list, neg_idea_list, med_idea_list, error_implement_list

