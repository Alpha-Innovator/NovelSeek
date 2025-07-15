from openai import OpenAI
import anthropic
import argparse
import json
import os
import retry
from dolphin_utils.rag_tools.prompts import *
from dolphin_utils.rag_tools.lit_review_tools import parse_and_execute, format_papers_for_printing, print_top_papers_from_paper_bank, \
    dedup_paper_bank, parse_io_description
from dolphin_utils.rag_tools.utils import cache_output, call_api


def initial_search(topic_description, openai_client, model, seed):
    prompt = INITIAL_SEARCH_PROMPT.format(TOPIC=topic_description.strip())
    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0., max_tokens=100, seed=seed,
                              json_output=False)
    return prompt, response, cost


def define_task_attribute(topic_description, openai_client, model, seed):
    prompt = TASK_ATTRIBUTE_PROMPT.format(TOPIC=topic_description.strip())
    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0., max_tokens=100, seed=seed,
                              json_output=False)
    return prompt, response, cost


def next_query(topic_description, openai_client, model, seed, grounding_papers, past_queries, task_attibute=False, io_description=None):
    grounding_papers_str = format_papers_for_printing(grounding_papers)
    if task_attibute and io_description is not None:
        prompt = NEXT_QUERY_PROMPT_WITH_ATTRIBUTE.format(TOPIC=topic_description.strip(), INPUT=io_description[0], OUTPUT=io_description[1], PAPER_LIST=grounding_papers_str, PAST_QUERIES=past_queries)
    else:
        prompt = NEXT_QUERY_PROMPT.format(TOPIC=topic_description.strip(),PAPER_LIST=grounding_papers_str, PAST_QUERIES=past_queries)

    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0., max_tokens=100, seed=seed,
                              json_output=False)
    return prompt, response, cost


def paper_score(paper_lst, topic_description, openai_client, model, seed, task_attribute=False, io_description=None):
    if task_attribute and io_description is not None:
        prompt = PAPER_SCORE_PROMPT_WITH_ATTRIBUTE.format(TOPIC=topic_description.strip(), INPUT=io_description[0], OUTPUT=io_description[1], PAPER_LIST=format_papers_for_printing(paper_lst))
    else:
        prompt = PAPER_SCORE_PROMPT.format(TOPIC=topic_description.strip(), PAPER_LIST=format_papers_for_printing(paper_lst))
    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0., max_tokens=4000, seed=seed,
                              json_output=True)
    return prompt, response, cost


@retry.retry(tries=3, delay=2)
def collect_papers(topic_description, openai_client, model, seed, grounding_k=10, max_papers=60, print_all=True):
    paper_bank = {}
    total_cost = 0
    all_queries = []

    ## get input and output description
    _, io_description, cost = define_task_attribute(topic_description, openai_client, model, seed)
    total_cost += cost
    all_queries.append(io_description)
    io_description = parse_io_description(io_description)

    ## get initial set of seed papers by KeywordSearch
    _, query, cost = initial_search(topic_description, openai_client, model, seed)
    total_cost += cost
    all_queries.append(query)
    paper_lst = parse_and_execute(query)
    if paper_lst:
        ## filter out those with incomplete abstracts
        paper_lst = [paper for paper in paper_lst if paper["abstract"] and len(paper["abstract"].split()) > 50]
        paper_bank = {paper["paperId"]: paper for paper in paper_lst}

        ## score each paper
        _, response, cost = paper_score(paper_lst, topic_description, openai_client, model, seed, task_attribute=True, io_description=io_description)
        total_cost += cost
        response = json.loads(response.strip())

        ## initialize all scores to 0 then fill in gpt4 scores
        for k, v in paper_bank.items():
            v["score"] = 0
        for k, v in response.items():
            try:
                paper_bank[k]["score"] = v
            except:
                continue
    else:
        paper_lst = []

    ## print stats
    if print_all:
        print("initial query: ", query)
        print("current total cost: ", total_cost)
        print("current size of paper bank: ", len(paper_bank))
        print_top_papers_from_paper_bank(paper_bank, top_k=10)
        print("\n")

    iter = 0
    ## keep expanding the paper bank until limit is reached
    while len(paper_bank) < max_papers and iter < 10:
        ## select the top k papers with highest scores for grounding
        data_list = [{'id': id, **info} for id, info in paper_bank.items()]
        grounding_papers = sorted(data_list, key=lambda x: x['score'], reverse=True)[: grounding_k]

        ## generate the next query
        _, new_query, cost = next_query(topic_description, openai_client, model, seed, grounding_papers, all_queries, task_attibute=True, io_description=io_description)
        all_queries.append(new_query)
        total_cost += cost
        if print_all:
            print("new query: ", new_query)
        try:
            paper_lst = parse_and_execute(new_query)
        except:
            paper_lst = None

        if paper_lst:
            ## filter out papers already in paper bank
            paper_lst = [paper for paper in paper_lst if paper["abstract"] and len(paper["abstract"].split()) > 50]
            paper_lst = [paper for paper in paper_lst if paper["paperId"] not in paper_bank]

            ## initialize all scores to 0 and add to paper bank
            for paper in paper_lst:
                paper["score"] = 0
            paper_bank.update({paper["paperId"]: paper for paper in paper_lst})

            ## gpt4 score new papers
            _, response, cost = paper_score(paper_lst, topic_description, openai_client, model, seed, task_attribute=True, io_description=io_description)
            total_cost += cost
            response = json.loads(response.strip())
            for k, v in response.items():
                try:
                    paper_bank[k]["score"] = v
                except:
                    continue

        elif print_all:
            print("No new papers found in this round.")

        ## print stats
        if print_all:
            print("current total cost: ", total_cost)
            print("current size of paper bank: ", len(paper_bank))
            print_top_papers_from_paper_bank(paper_bank, top_k=10)
            print("\n")

        iter += 1

    ## rank all papers by score
    data_list = [{'id': id, **info} for id, info in paper_bank.items()]
    sorted_data = sorted(data_list, key=lambda x: x['score'], reverse=True)
    sorted_data = dedup_paper_bank(sorted_data)

    return sorted_data, total_cost, all_queries




