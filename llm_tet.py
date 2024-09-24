import os
import argparse
import pickle
from pathlib import Path
from getpass import getpass
from tqdm.auto import tqdm
import numpy as np
import torch
import transformers
import datasets
from sklearn.metrics import precision_recall_fscore_support
from sentence_transformers import SentenceTransformer, util
from langchain_openai import OpenAIEmbeddings


_OPENAI_MODELS = ['text-embedding-ada-002', 'text-embedding-3-small', 'text-embedding-3-large']
_ST_MODELS = ['all-mpnet-base-v2', 'all-MiniLM-L6-v2']
_MODELS = _OPENAI_MODELS + _ST_MODELS


_DATASETS = ['wiki727k', 'en_city', 'en_disease']


def embed_sentences(sentences, embedder_fn, cache_path):
    if Path(cache_path).exists():
        print(f'Loading embeddings from cache: {cache_path}')
        with open(cache_path, 'rb') as file:
            embedded_sents = pickle.load(file)
    else:
        print(f'Embedding sentences and saving to cache: {cache_path}')
        embedded_sents = []
        for doc_sentences in tqdm(sentences, desc='Embedding sentences'):
            embedded_doc = embedder_fn(doc_sentences)
            assert len(embedded_doc) == len(doc_sentences)
            embedded_sents.append(embedded_doc)
        
        with open(cache_path, 'wb') as file:
            pickle.dump(embedded_sents, file)
            
    return embedded_sents


def calculate_cosine_similarities(embedded_sents, k=1, pool='mean'):
    def cosine_similarity(a, b):
        sim = util.cos_sim(a, b)
        if pool == 'mean':
            return sim.mean().item()
        elif pool == 'max':
            return sim.max().item()
        elif pool == 'min':
            return sim.min().item()
        else:
            raise ValueError(f'Invalid pooling method: {pool}')

    all_sims = []
    for doc in tqdm(embedded_sents, desc='Calculating cosine similarities', disable=True):
        doc_sims = []
        for i in range(len(doc) - 1):
            lctx = doc[max(0, i-k+1) : i+1]
            rctx = doc[i+1 : i+k+1]
            sim = cosine_similarity(lctx, rctx)
            doc_sims.append(sim)
        all_sims.append(doc_sims)
    return all_sims


def predict_boundaries(cosine_sims, threshold):
    predictions = []
    for doc_sims in cosine_sims:
        doc_boundaries = [1 if sim < threshold else 0 for sim in doc_sims]
        predictions.append(doc_boundaries)
    return predictions


def evaluate_predictions(predictions, labels):
    flattened_predictions = [pred for doc in predictions for pred in doc]
    flattened_labels = [label for doc in labels for label in doc[:-1]]
    assert len(flattened_labels) == len(flattened_predictions)
    
    precision, recall, f1, _ = precision_recall_fscore_support(flattened_labels, flattened_predictions, average='binary', zero_division=0)
    return f1, precision, recall


def repr_score(score):
    if not score:
        return 'N/A'
    f1, prec, recall = score
    return f'F1={100 * f1:.2f}, Prec={100 * prec:.2f}, Recall={100 * recall:.2f}'


def find_optimum_threshold(dataset, embedder_fn, cache_path):
    embedded_sents = embed_sentences(dataset["sentences"], embedder_fn, cache_path=cache_path)
    
    best_hp = None
    best_score = None
    
    ks = [1, 3, 5, 8, 10]
    pools = ['mean', 'max', 'min']
    ts = np.linspace(0, 1, 200, endpoint=False)
    pbar = tqdm(total=len(ks) * len(pools) * len(ts), desc='Hyperparameter tuning')
    for k in ks:
        for pool in pools:
            pbar.set_description(f'k={k}, pool={pool} (best: {repr_score(best_score)})')
            
            cosine_sims = calculate_cosine_similarities(embedded_sents, k=k, pool=pool)

            for threshold in ts:
                predictions = predict_boundaries(cosine_sims, threshold)
                score = evaluate_predictions(predictions, dataset["labels"])
                if not best_score or score[0] > best_score[0]:
                    best_score = score
                    best_hp = {'k': k, 'pool': pool, 'threshold': threshold}
                pbar.update(1)
    pbar.close()
    return best_hp, best_score


def tune_and_predict(validation_dset, test_dset, embedder_fn, cache_dir):
    tuned_hp, tuned_score = find_optimum_threshold(validation_dset, embedder_fn, cache_path=Path(cache_dir) / f'validation.pkl')
    print(f'Validation score: {repr_score(tuned_score)} | Tuned hp: {tuned_hp}')
    
    embedded_sents = embed_sentences(test_dset['sentences'], embedder_fn, cache_path=Path(cache_dir) / 'test.pkl')
    cosine_sims = calculate_cosine_similarities(embedded_sents, k=tuned_hp['k'], pool=tuned_hp['pool'])
    predictions = predict_boundaries(cosine_sims, threshold=tuned_hp['threshold'])
    score = evaluate_predictions(predictions, test_dset['labels'])
    
    print(f'Test score: {repr_score(score)}')


def llm_tet(model_name: str, 
            data_name: str, keep_titles: bool = False, max_samples: int = None, seed: int = None, 
            cache_dir: str = None):
    
    if seed is not None:
        transformers.set_seed(seed)
        
    # Load model
    if model_name in _OPENAI_MODELS:
        api_key = os.environ.get('OPENAI_API_KEY') or getpass('Enter OpenAI API key: ')
        model = OpenAIEmbeddings(model=model_name, api_key=api_key)
        embedder_fn = model.embed_documents
    elif model_name in _ST_MODELS:
        model = SentenceTransformer(model_name, device=('cuda' if torch.cuda.is_available() else 'cpu'))
        embedder_fn = model.encode   
    else:
        raise ValueError(f'Invalid model name: {model_name}')

    # Load and process dataset
    if data_name == 'wiki727k':
        dsets = datasets.load_dataset('saeedabc/wiki727k', drop_titles=not keep_titles, 
                                      trust_remote_code=True, num_proc=8)
    elif data_name in ['en_city', 'en_disease']:
        dsets = datasets.load_dataset('saeedabc/wikisection', data_name, drop_titles=not keep_titles, 
                                      trust_remote_code=True, num_proc=8)
    else:
        raise ValueError(f'Invalid dataset name: {data_name}')

    validation_dset = dsets['validation']
    test_dset = dsets['test']
    if max_samples:
        validation_dset = validation_dset.shuffle(seed=seed).select(range(max_samples))
        test_dset = test_dset.shuffle(seed=seed).select(range(max_samples))
    
    # Tune hyper-parameters on validation set and predict on test set
    data_id = data_name + ('_titled' if keep_titles else '') + \
              (f'_{max_samples}' if max_samples else '') + (f'_seed{seed}' if seed is not None else '')
    cache_dir = Path(cache_dir) / f'{data_id}_{model_name}'
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    tune_and_predict(validation_dset, test_dset, embedder_fn, cache_dir=cache_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_name', type=str, required=True, choices=_MODELS, help="Name of the embedding model")
    
    parser.add_argument('--data_name', type=str, required=True, choices=_DATASETS, help="Name of the dataset")
    parser.add_argument('--keep_titles', action='store_true', help="Whether to keep titles among the sentences in the dataset")
    parser.add_argument('--max_samples', type=int, default=None, help="Maximum number of validation and test samples to use")
    parser.add_argument('--seed', type=int, default=None, help="Random seed for reproducibility")
        
    parser.add_argument('--cache_dir', type=str, default=None, help="Directory to cache the embeddings (for reuse in later same-config runs)")

    args = parser.parse_args()
    llm_tet(model_name=args.model_name, 
            data_name=args.data_name, keep_titles=args.keep_titles, max_samples=args.max_samples, seed=args.seed, 
            cache_dir=args.cache_dir)