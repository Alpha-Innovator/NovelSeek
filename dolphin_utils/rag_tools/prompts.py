INITIAL_SEARCH_PROMPT = """You are a researcher doing literature review on the topic of {TOPIC}.
You should propose some keywords for using the Semantic Scholar API to find the most relevant papers to this topic.
Formulate your query as: KeywordQuery(\"keyword\"). Just give me one query, with the most important keyword, the keyword can be a concatenation of multiple keywords (just put a space between every word) but please be concise and try to cover all the main aspects
Your query (just return the query itself with no additional text)
"""

TASK_ATTRIBUTE_PROMPT = """You are a researcher doing research on the topic of {TOPIC}.
You should define the task attribute such as the model input and output of the topic for better searching relevant papers.
Formulate the input and output as: Attribute(\"attribute\"). For example, Input(\"input\"), Output(\"output\").
The attribute: (just return the task attribute itself with no additional text):
"""

NEXT_QUERY_PROMPT_WITH_ATTRIBUTE = """You are a researcher doing literature review on the topic of {TOPIC}.
You should propose some queries for using the Semantic Scholar API to find the most relevant papers to this topic.
The input and output of the queries should be same with: input: {INPUT}, output: {OUTPUT}.
(1) KeywordQuery(\"keyword\"): find most relevant papers to the given keyword (the keyword shouldn't be too long and specific, otherwise the search engine will fail; it is ok to combine a few shor keywords with spaces, such as \"lanaguage model reasoning\").\n
(2) PaperQuery(\"paperId\"): find the most similar papers to the given paper (as specified by the paperId).
(3) GetReferences(\"paperId\"): get the list of papers referenced in the given paper (as specified by the paperId).
Right now you have already collected the following relevant papers: 
{PAPER_LIST}
You can formulate new search queries based on these papers. And you have already asked the following queries:
{PAST_QUERIES}
Please formulate a new query to expand our paper collection with more diverse and relevant papers (you can do so by diversifying the types of queries to generate and minimize the overlap with previous queries). Directly give me your new query without any explanation or additional text, just the query itself:
"""

NEXT_QUERY_PROMPT = """You are a researcher doing literature review on the topic of {TOPIC}.
You should propose some queries for using the Semantic Scholar API to find the most relevant papers to this topic.
(1) KeywordQuery(\"keyword\"): find most relevant papers to the given keyword (the keyword shouldn't be too long and specific, otherwise the search engine will fail; it is ok to combine a few shor keywords with spaces, such as \"lanaguage model reasoning\").\n
(2) PaperQuery(\"paperId\"): find the most similar papers to the given paper (as specified by the paperId).
(3) GetReferences(\"paperId\"): get the list of papers referenced in the given paper (as specified by the paperId).
Right now you have already collected the following relevant papers: 
{PAPER_LIST}
You can formulate new search queries based on these papers. And you have already asked the following queries:
{PAST_QUERIES}
Please formulate a new query to expand our paper collection with more diverse and relevant papers (you can do so by diversifying the types of queries to generate and minimize the overlap with previous queries). Directly give me your new query without any explanation or additional text, just the query itself:
"""

PAPER_SCORE_PROMPT_WITH_ATTRIBUTE = """You are a helpful literature review assistant whose job is to read the below set of papers and score each paper. The criteria for scoring are:
(1) The paper is directly relevant to the topic of: {TOPIC}. Note that it should be specific to solve the problem of focus, rather than just generic methods. 
(2) The input and output of the proposed method in this paper is same with input: {INPUT}, output: {OUTPUT}. Note that if the input and output are not match, the paper should get a low score.
(3) The paper is an empirical paper that proposes a novel method and conducts computational experiments to show improvement over baselines (position or opinion papers, review or survey papers, and analysis papers should get low scores for this purpose).
(4) The paper is interesting, exciting, and meaningful, with potential to inspire many new projects.
The papers are:
{PAPER_LIST}
Please score each paper from 1 to 10. Write the response in JSON format with \"paperID: score\" as the key and value for each paper.
"""

PAPER_SCORE_PROMPT = """You are a helpful literature review assistant whose job is to read the below set of papers and score each paper. The criteria for scoring are:
(1) The paper is directly relevant to the topic of: {TOPIC}. Note that it should be specific to solve the problem of focus, rather than just generic methods. 
(2) The paper is an empirical paper that proposes a novel method and conducts computational experiments to show improvement over baselines (position or opinion papers, review or survey papers, and analysis papers should get low scores for this purpose).
(3) The paper is interesting, exciting, and meaningful, with potential to inspire many new projects.
The papers are:
{PAPER_LIST}
Please score each paper from 1 to 10. Write the response in JSON format with \"paperID: score\" as the key and value for each paper.
"""




