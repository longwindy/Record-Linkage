import pandas as pd
import recordlinkage
import networkx as nx

# Preprogressing
def preprocess_df(df, name_col='NAME'):
    df = df.copy()

    cleaned = df[name_col].astype(str).str.strip().str.lower()
    cleaned = cleaned.str.replace(r'[^\w\s]', '', regex=True)
    cleaned = cleaned.str.replace(r'\s+', ' ', regex=True)
    cleaned = cleaned.str.replace(' ', '') 

    cleaned = cleaned.str.normalize('NFKD')

    company_types = {
        r'\bllc\b': 'llc',
        r'\binc\b': 'inc',
        r'\bltd\b': 'ltd',
        r'\bcorp\b': 'corp'
    }
    for pattern, replacement in company_types.items():
        cleaned = cleaned.str.replace(pattern, replacement, regex=True)
    
    spelling_dict = {
        r'\bnd\b': 'and',
        r'\bsops\b': 'shops'
    }
    for pattern, replacement in spelling_dict.items():
        cleaned = cleaned.str.replace(pattern, replacement, regex=True)
    
    stop_words = {'and', 'the', 'inc'}
    cleaned = cleaned.str.split().apply(lambda x: ' '.join([w for w in x if w not in stop_words]))
    

    df[name_col] = cleaned
    return df

# Deduplication
def deduplication(df, name_col='NAME', type_col=None, sim_threshold=0.85, window_size=9):
    df = preprocess_df(df, name_col)
    df = df.reset_index(drop=True)
    
    if type_col and type_col in df.columns:
        groups = df.groupby(type_col)
    else:
        groups = [(None, df)]
    
    all_components = []
    for type_value, group in groups:
        sorted_group = group.sort_values(by=name_col, key=lambda x: x.str[:10])
        
        indexer = recordlinkage.Index()
        indexer.sortedneighbourhood(left_on=name_col, window=window_size)
        candidate_links = indexer.index(sorted_group)
        
        compare = recordlinkage.Compare(n_jobs=-1)
        compare.string(name_col, name_col, method='jarowinkler', label='name_sim')
        features = compare.compute(candidate_links, sorted_group)
        
        candidate_matches = features[(features['name_sim'] >= sim_threshold*0.95)]
        candidate_matches = candidate_matches[candidate_matches['name_sim'] >= sim_threshold]
        
        G = nx.Graph()
        G.add_nodes_from(sorted_group.index)
        for _, row in candidate_matches.reset_index().iterrows():
            idx1, idx2 = row['level_0'], row['level_1']
            G.add_edge(idx1, idx2)
        
        components = list(nx.connected_components(G))
        all_components.extend(components)
    
    id_map = {}
    for component in all_components:
        indices = list(component)
        ids_in_component = set(df.loc[indices, 'ID'])
        for idx in component:
            id_map[idx] = ids_in_component
    
    df['all_ids'] = df.index.map(id_map)
    return df

# Matching
def match_test_records(test_df, unified_df, name_col='NAME', sim_threshold=0.85):
    test_df = preprocess_df(test_df.copy(), name_col)
    unified_df = preprocess_df(unified_df.copy(), name_col)
    
    indexer = recordlinkage.Index()
    indexer.full()
    candidate_links = indexer.index(test_df, unified_df)
    
    compare = recordlinkage.Compare()
    
    compare.string(name_col, name_col, 
                   method='qgram',
                   label='qgram_sim')
    
    compare.string(name_col, name_col,
                   method='damerau_levenshtein',
                   label='dl_sim')
    
    features = compare.compute(candidate_links, test_df, unified_df)
    features['combined_sim'] = 0.2*features['dl_sim'] + 0.8*features['qgram_sim']
    
    matches = []
    for test_idx, group in features.groupby(level=0):
        best_idx = group['combined_sim'].idxmax()
        best_score = group.loc[best_idx, 'combined_sim']
        if best_score >= sim_threshold:
            uni_idx = best_idx[1]
            matched_ids = unified_df.loc[uni_idx, 'all_ids']
            matches.append({
                'test_index': test_idx,
                'test_name': test_df.loc[test_idx, name_col],
                'matched_ids': matched_ids,
                'score': best_score
            })
        else:
            matches.append({
                'test_index': test_idx,
                'test_name': test_df.loc[test_idx, name_col],
                'matched_ids': set(),
                'score': best_score
            })
    return pd.DataFrame(matches)

# Evaluation
def matching_evaluation(test_df, matching_results):
    test_df = test_df.reset_index(drop=True)
    matching_results['true_id'] = test_df.loc[matching_results['test_index'], 'ID'].values
    
    matching_results['correct'] = matching_results.apply(
        lambda row: row['true_id'] in row['matched_ids'], axis=1
    )
    accuracy = matching_results['correct'].mean()
    
    matching_results['true_group'] = matching_results['true_id'].apply(
        lambda x: next(ids for ids in test_df['all_ids'] if x in ids)
    )
    matching_results['recall'] = matching_results.apply(
        lambda row: len(row['matched_ids'] & row['true_group']) / len(row['true_group']),
        axis=1
    )
    recall = matching_results['recall'].mean()
    print(f"Recall: {recall:.2%}")
    
    matching_results.to_csv('evaluation.csv', index=False)
    return accuracy

# Main function
def main():
    primary = pd.read_csv('origin/primary.csv')
    alternate = pd.read_csv('origin/alternate.csv')


    test_df = pd.read_csv('test/test_01.csv')
    test_df = deduplication(test_df, name_col='NAME', sim_threshold=0.85)
    
    dedup_primary = deduplication(primary, name_col='NAME', type_col='TYPE', sim_threshold=0.85)
    dedup_primary.to_csv('dedup_primary.csv', index=False)
    print("Complete deduplicate for primary.csv")
    
    dedup_alternate = deduplication(alternate, name_col='NAME', sim_threshold=0.85)
    dedup_alternate.to_csv('dedup_alternate.csv', index=False)
    print("Complete deduplicate for alternate.csv")
    
    unified = pd.concat([dedup_primary, dedup_alternate], ignore_index=True)
    unified = deduplication(unified, name_col='NAME', sim_threshold=0.85)
    unified.to_csv('unified.csv', index=False)
    print("Complete converge")

    matching_results = match_test_records(test_df, unified, name_col='NAME', sim_threshold=0.65)
    matching_results.to_csv('matching_results.csv', index=False)
    print("Complete matching")
    
    accuracy = matching_evaluation(test_df, matching_results)
    print("Matching accuracy: {:.2%}".format(accuracy))

if __name__ == '__main__':
    main()
