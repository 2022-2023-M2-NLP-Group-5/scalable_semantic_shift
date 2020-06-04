

## Instructions ##


Install dependencies if needed: pip install -r requirements.txt<br/>
Add coha corpus (folders COHA_1960 and COHA_19900) and Gulordava_word_meaning_change_evaluation_dataset.csv into data/coha/<br/>
Run 'python build_coha_corpus'. This generates two .txt files, one for each time period, which are used  as input to embedding extraction script<br/>
If you are using fine-tuned BERT, change the path to the model in get_embeddings_scalable.py accordingly<br/>
Run 'python get_embeddings_scalable.py' to extract embeddings. Change kmeans_clustering argument of function get_time_embeddings to True,
to use the kmeans clustering approach instead of cosine similarity. Also change kmeans_clustering argument to True in the coha_get_all_results.py <br/>
Run python coha_get_all_results.py to cluster embeddings<br/>
Run python evaluate_coha.py to evaluate on gold standard<br/>



