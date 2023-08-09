prompt_summarization_from_mis   =   """You will be given a series of most important sentences from a paper. Your goal is to give a summary of the paper.
The sentences will be enclosed in triple backtrips (```).

sentences :
```{most_important_sents}```

SUMMARY :"""

global_summary_promp  = """
You will be given a series of summaries from a text.
Your goal is to write a general summary from the given summaries.

```{text}```
SUMMARY:
"""


summary_prompt  =   """
You will be given a text. Give a concise and understanding summary.

```{text}```
CONCISE SUMMARY :
"""

global_summary_prompt   = """
You will be given a series of summaries from a text.
Your goal is to write a general summary from the given summaries.

```{text}```
SUMMARY:
"""