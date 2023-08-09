from utils.prompts import prompt_summarization_from_mis, summary_prompt, global_summary_prompt
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from summarizer import Summarizer
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
import numpy as np
from sklearn.cluster import KMeans
import os

os.environ["OPENAI_API_KEY"]    =   "API_KEY"
embeddings  =   OpenAIEmbeddings()
llm         =   OpenAI(temperature=0.0) # Set temperature to 0.0 as we don't want creative answer
model       =   Summarizer()

Document()

def bert_extractive_summarization(full_text:str, n_sentences=40) -> str:
    """Performs an extractive summarization using BERT's embbedings
    then uses the most important sentences for summarization using an API Call

    Inputs :
        - full_text : The whole text of any size
        - n_sentences : The number of sentences that the extractive summarization process should output

    Outputs :
        - res : The result of the summarization from the LLM
    """
    prompt_summarize_most_important_sents   =   PromptTemplate(template=prompt_summarization_from_mis, input_variables=["most_important_sents"])
    result = model(full_text, num_sentences=n_sentences) # We specify a number of sentences
    
    res =   llm(prompt=prompt_summarize_most_important_sents.format(most_important_sents=result))

    return res

def kNN_long_text_summarization(docs:list, n_clusters:int) -> str:
    """Performs a map reduce style operation. By :
        - Getting the Embeddings of each chunk
        - Clustering the Embeddings (Considering each chunks in each clusters likely talking about the same thing)
        - Finding the closest chunks to the centroids (Ultimately considering each closest to centroids as the most representative chunk of each cluster)
        - Getting a summary for each most representative chunk
        - Getting the global summary from each summary

    Inputs :
        - docs : The documents to summarize
        - n_clusters : The number of clusters

    Outputs :
        - summary : The final summary
    """
    summary_prompt_template         =   PromptTemplate(template=summary_prompt, input_variables=["text"])
    global_summary_prompt_template  =   PromptTemplate(template=global_summary_prompt, input_variables=["text"])

    embeds      =   embeddings.embed_documents([doc.page_content for doc in docs])

    kmeans      =   KMeans(n_clusters=n_clusters).fit(embeds)

    most_representative =   [np.argmin(np.linalg.norm(embeds - kmeans.cluster_centers_[i], axis=1)) for i in range(n_clusters)]
    most_representative =   sorted(most_representative)

    map_summaries   =   []
    for idx in most_representative:
        res =   llm(prompt=summary_prompt_template.format(text=docs[idx]))
        map_summaries.append(res)

    summary =   llm(prompt=global_summary_prompt_template.format(text="\n".join(map_summaries)))

    return summary