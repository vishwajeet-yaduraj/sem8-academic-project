import pandas as pd
import networkx as nx
import json
from datetime import datetime

INPUT_FILE = "data/ai_tech_posts.csv"
OUTPUT_GRAPH = "data/interaction_graph.json"

def load_data():
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} posts")
    print(f"Unique authors: {df['author'].nunique()}")
    return df

def build_graph(df):
    G = nx.DiGraph()

    # Add all authors as nodes
    for _, row in df.iterrows():
        G.add_node(row['author'], 
                   display_name=row['author_display'],
                   post_count=0,
                   total_likes=0,
                   total_reposts=0)

    # Update node attributes
    for author, group in df.groupby('author'):
        G.nodes[author]['post_count'] = len(group)
        G.nodes[author]['total_likes'] = int(group['like_count'].sum())
        G.nodes[author]['total_reposts'] = int(group['repost_count'].sum())

    # Add edges based on repost weight
    # Two users are connected if they post about same topic
    # Edge weight = combined interaction score
    for _, row in df.iterrows():
        if row['repost_count'] > 0 or row['reply_count'] > 0:
            # This user's post got interaction — 
            # create edges to top posters in same search term
            same_term = df[df['search_term'] == row['search_term']]
            top_posters = same_term.nlargest(5, 'like_count')['author'].tolist()
            
            for target in top_posters:
                if target != row['author']:
                    if G.has_edge(row['author'], target):
                        G[row['author']][target]['weight'] += 1
                    else:
                        G.add_edge(row['author'], target, weight=1)

    return G

def compute_metrics(G):
    print(f"\nGraph Statistics:")
    print(f"  Nodes (users): {G.number_of_nodes()}")
    print(f"  Edges (interactions): {G.number_of_edges()}")
    print(f"  Density: {nx.density(G):.6f}")

    # Largest connected component
    undirected = G.to_undirected()
    components = list(nx.connected_components(undirected))
    largest = max(components, key=len)
    print(f"  Connected components: {len(components)}")
    print(f"  Largest component size: {len(largest)}")

    # PageRank
    print(f"\nComputing PageRank...")
    pagerank = nx.pagerank(G, alpha=0.85)
    top_10 = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print(f"\nTop 10 Influencers by PageRank:")
    for i, (user, score) in enumerate(top_10, 1):
        node = G.nodes[user]
        print(f"  {i}. {user}")
        print(f"     PageRank: {score:.6f}")
        print(f"     Posts: {node['post_count']}, "
              f"Likes: {node['total_likes']}, "
              f"Reposts: {node['total_reposts']}")

    return pagerank

def save_graph(G):
    data = nx.node_link_data(G)
    with open(OUTPUT_GRAPH, 'w') as f:
        json.dump(data, f)
    print(f"\nGraph saved to {OUTPUT_GRAPH}")

if __name__ == "__main__":
    print(f"Building graph at {datetime.now().strftime('%H:%M:%S')}")
    print("="*50)
    
    df = load_data()
    G = build_graph(df)
    pagerank = compute_metrics(G)
    save_graph(G)
    
    print("\nDone.")