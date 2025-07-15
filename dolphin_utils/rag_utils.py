import json


def format_papers_for_printing(paper_lst, include_abstract=True, include_score=True, include_id=True):
    ## convert a list of papers to a string for printing or as part of a prompt
    output_str = ""
    for paper in paper_lst:
        if include_id:
            output_str += "paperId: " + paper["id"].strip() + "\n"
        output_str += "title: " + paper["title"].strip() + "\n"
        if include_abstract and "abs" in paper and paper["abs"]:
            output_str += "abstract: " + paper["abs"].strip() + "\n"
        elif include_abstract and "tldr" in paper and paper["tldr"] and paper["tldr"]["text"]:
            output_str += "tldr: " + paper["tldr"]["text"].strip() + "\n"
        if "score" in paper and include_score:
            output_str += "relevance score: " + str(paper["score"]) + "\n"
        output_str += "\n"

    return output_str


def format_papers_for_printing_ai_researcher(paper_lst, include_abstract=True, include_score=True, include_id=True):
    ## convert a list of papers to a string for printing or as part of a prompt
    output_str = ""
    paper_lst = paper_lst['paper_bank']
    for paper in paper_lst:
        if paper['score'] < 8:
            continue
        if include_id:
            output_str += "paperId: " + paper["id"].strip() + "\n"
        output_str += "title: " + paper["title"].strip() + "\n"
        if include_abstract and "abstract" in paper and paper["abstract"]:
            output_str += "abstract: " + paper["abstract"].strip() + "\n"
        elif include_abstract and "tldr" in paper and paper["tldr"] and paper["tldr"]["text"]:
            output_str += "tldr: " + paper["tldr"]["text"].strip() + "\n"
        if "score" in paper and include_score:
            output_str += "relevance score: " + str(paper["score"]) + "\n"
        output_str += "\n"

    return output_str
