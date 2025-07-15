import requests
import re
import os
import json

# Define the paper search endpoint URL
search_url = 'https://api.semanticscholar.org/graph/v1/paper/search/'
graph_url = 'https://api.semanticscholar.org/graph/v1/paper/'
rec_url = "https://api.semanticscholar.org/recommendations/v1/papers/forpaper/"

S2_KEY = os.environ["S2_API_KEY"]


def KeywordQuery(keyword):
    query_params = {
        'query': keyword,
        'limit': 20,
        'fields': 'title,year,citationCount,abstract,tldr'
    }
    headers = {'x-api-key': S2_KEY}
    response = requests.get(search_url, params=query_params, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        return None


def PaperQuery(paper_id):
    ## retrieve similar papers based on paper id
    query_params = {
        'paperId': paper_id,
        'limit': 20,
        'fields': 'title,year,citationCount,abstract'
    }
    headers = {'x-api-key': S2_KEY}
    response = requests.get(url=rec_url + paper_id, params=query_params, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        return None


def PaperDetails(paper_id, fields='title,year,abstract,authors,citationCount,venue,citations,references,tldr'):
    ## get paper details based on paper id
    paper_data_query_params = {'fields': fields}
    headers = {'x-api-key': S2_KEY}
    response = requests.get(url=graph_url + paper_id, params=paper_data_query_params, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        return None


def GetAbstract(paper_id):
    ## get the abstract of a paper based on paper id
    paper_details = PaperDetails(paper_id)

    if paper_details is not None:
        return paper_details["abstract"]
    else:
        return None


def GetCitationCount(paper_id):
    ## get the citation count of a paper based on paper id
    paper_details = PaperDetails(paper_id)

    if paper_details is not None:
        return int(paper_details["citationCount"])
    else:
        return None


def GetCitations(paper_id):
    ## get the citation list of a paper based on paper id
    paper_details = PaperDetails(paper_id)

    if paper_details is not None:
        return paper_details["citations"]
    else:
        return None


def GetReferences(paper_id):
    ## get the reference list of a paper based on paper id
    paper_details = PaperDetails(paper_id)
    references = paper_details["references"][: 100]

    ## get details of each reference, keep first 20 to save costs
    detailed_references = [PaperDetails(ref["paperId"], fields='title,year,abstract,citationCount') for ref in
                           references if ref["paperId"]]
    detailed_references = paper_filter(detailed_references)[: 20]

    if paper_details is not None:
        return detailed_references
    else:
        return None


def paper_filter(paper_lst):
    ## filter out papers based on some basic heuristics
    filtered_lst = []
    for paper in paper_lst:
        abstract = paper["abstract"] if paper["abstract"] else paper["title"]
        # if ("year" in paper.keys()) and (paper["year"] is not None) and (int(paper["year"]) < 2022):
        #     continue
        # if ("citationCount" in paper.keys()) and (paper["citationCount"] is not None) and int(paper["citationCount"]) <= 10:
        #     continue
        if "survey" in abstract.lower() or "review" in abstract.lower() or "position paper" in abstract.lower():
            continue
        filtered_lst.append(paper)
    return filtered_lst


def parse_and_execute(output):
    ## parse gpt4 output and execute corresponding functions
    if output.startswith("KeywordQuery"):
        match = re.match(r'KeywordQuery\("([^"]+)"\)', output)
        keyword = match.group(1) if match else None
        if keyword:
            response = KeywordQuery(keyword)
            if 'total' in response and response['total'] == 0:
                return None
            if response is not None:
                if "data" in response:
                    paper_lst = response["data"]
                else:
                    paper_lst = response[:]
                return paper_filter(paper_lst)
    elif output.startswith("PaperQuery"):
        match = re.match(r'PaperQuery\("([^"]+)"\)', output)
        paper_id = match.group(1) if match else None
        if paper_id:
            response = PaperQuery(paper_id)
            if response is not None and response["recommendedPapers"]:
                paper_lst = response["recommendedPapers"]
                return paper_filter(paper_lst)
    elif output.startswith("GetAbstract"):
        match = re.match(r'GetAbstract\("([^"]+)"\)', output)
        paper_id = match.group(1) if match else None
        if paper_id:
            return GetAbstract(paper_id)
    elif output.startswith("GetCitationCount"):
        match = re.match(r'GetCitationCount\("([^"]+)"\)', output)
        paper_id = match.group(1) if match else None
        if paper_id:
            return GetCitationCount(paper_id)
    elif output.startswith("GetCitations"):
        match = re.match(r'GetCitations\("([^"]+)"\)', output)
        paper_id = match.group(1) if match else None
        if paper_id:
            return GetCitations(paper_id)
    elif output.startswith("GetReferences"):
        match = re.match(r'GetReferences\("([^"]+)"\)', output)
        paper_id = match.group(1) if match else None
        if paper_id:
            return GetReferences(paper_id)
    return None


def parse_io_description(output):
    match_input = re.match(r'Input\("([^"]+)"\)', output)
    input_description = match_input.group(1) if match_input else None
    match_output = re.match(r'.*Output\("([^"]+)"\)', output)
    output_description = match_output.group(1) if match_output else None
    return input_description, output_description


def format_papers_for_printing(paper_lst, include_abstract=True, include_score=True, include_id=True):
    ## convert a list of papers to a string for printing or as part of a prompt
    output_str = ""
    for paper in paper_lst:
        if include_id:
            output_str += "paperId: " + paper["paperId"].strip() + "\n"
        output_str += "title: " + paper["title"].strip() + "\n"
        if include_abstract and "abstract" in paper and paper["abstract"]:
            output_str += "abstract: " + paper["abstract"].strip() + "\n"
        elif include_abstract and "tldr" in paper and paper["tldr"] and paper["tldr"]["text"]:
            output_str += "tldr: " + paper["tldr"]["text"].strip() + "\n"
        if "score" in paper and include_score:
            output_str += "relevance score: " + str(paper["score"]) + "\n"
        output_str += "\n"

    return output_str


def print_top_papers_from_paper_bank(paper_bank, top_k=10):
    data_list = [{'id': id, **info} for id, info in paper_bank.items()]
    top_papers = sorted(data_list, key=lambda x: x['score'], reverse=True)[: top_k]
    print(format_papers_for_printing(top_papers, include_abstract=False))


def dedup_paper_bank(sorted_paper_bank):
    idx_to_remove = []

    for i in reversed(range(len(sorted_paper_bank))):
        for j in range(i):
            if sorted_paper_bank[i]["paperId"].strip() == sorted_paper_bank[j]["paperId"].strip():
                idx_to_remove.append(i)
                break
            if ''.join(sorted_paper_bank[i]["title"].lower().split()) == ''.join(
                    sorted_paper_bank[j]["title"].lower().split()):
                idx_to_remove.append(i)
                break
            if sorted_paper_bank[i]["abstract"] == sorted_paper_bank[j]["abstract"]:
                idx_to_remove.append(i)
                break

    deduped_paper_bank = [paper for i, paper in enumerate(sorted_paper_bank) if i not in idx_to_remove]
    return deduped_paper_bank
