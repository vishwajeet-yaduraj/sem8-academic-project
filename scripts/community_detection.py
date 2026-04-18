import pandas as pd
import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
from collections import Counter, defaultdict

INPUT_CSV = "data/ai_tech_posts.csv"
INPUT_GRAPH = "data/interaction_graph.json"

def load_data():
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} posts from {df['author'].nunique()} unique authors")
    return df

def load_graph():
    with open(INPUT_GRAPH, 'r') as f:
        data = json.load(f)
    G = nx.node_link_graph(data)
    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

def detect_communities(G):
    # Louvain works on undirected graphs
    G_undirected = G.to_undirected()

    # Run Louvain
    partition = community_louvain.best_partition(G_undirected, random_state=42)

    # Count community sizes
    community_sizes = Counter(partition.values())
    print(f"\nCommunities detected: {len(community_sizes)}")
    print(f"Modularity score: {community_louvain.modularity(partition, G_undirected):.4f}")

    # Show top communities by size
    print("\nTop 10 communities by size:")
    for comm_id, size in community_sizes.most_common(10):
        print(f"  Community {comm_id}: {size} users")

    return partition, community_sizes

def label_communities(partition, df, top_n=8):
    # For each community, find the most active users
    # and their most common search terms to auto-label
    community_users = defaultdict(list)
    for user, comm_id in partition.items():
        community_users[comm_id].append(user)

    # Get search terms per user from the dataframe
    user_terms = df.groupby('author')['search_term'].apply(list).to_dict()

    community_labels = {}
    community_sizes = Counter(partition.values())

    print("\nCommunity profiles (top communities):")
    for comm_id, size in community_sizes.most_common(top_n):
        users = community_users[comm_id]

        # Collect all search terms for users in this community
        all_terms = []
        for user in users:
            if user in user_terms:
                all_terms.extend(user_terms[user])

        term_counts = Counter(all_terms)
        top_terms = [t for t, _ in term_counts.most_common(3)]

        # Get top user by post count
        user_posts = df[df['author'].isin(users)].groupby('author').size()
        top_user = user_posts.idxmax() if len(user_posts) > 0 else "unknown"

        # Auto-generate label from top terms
        if top_terms:
            label = " / ".join(top_terms[:2])
        else:
            label = f"Community {comm_id}"

        community_labels[comm_id] = label
        print(f"\n  Community {comm_id} ({size} users)")
        print(f"    Top terms: {top_terms}")
        print(f"    Top user: {top_user}")
        print(f"    Label: {label}")

    return community_labels, community_users

def plot_community_network(G, partition, community_labels):
    print("\nPlotting community network...")

    # Keep only top 8 communities for clarity
    community_sizes = Counter(partition.values())
    top_communities = [c for c, _ in community_sizes.most_common(8)]

    # Filter nodes to top communities only
    nodes_to_keep = [n for n, c in partition.items() if c in top_communities]
    G_sub = G.subgraph(nodes_to_keep).copy()

    # Color palette for communities
    colors = [
        '#FF6B35', '#4ECDC4', '#45B7D1', '#96CEB4',
        '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F'
    ]

    color_map = {comm: colors[i % len(colors)]
                 for i, comm in enumerate(top_communities)}

    node_colors = [color_map.get(partition.get(n, -1), '#888888')
                   for n in G_sub.nodes()]

    # Size by degree
    degrees = dict(G_sub.degree())
    node_sizes = [max(30, degrees[n] * 40) for n in G_sub.nodes()]

    fig, ax = plt.subplots(figsize=(18, 13))
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#0f0f1a')

    pos = nx.spring_layout(G_sub, k=1.0, seed=42)

    nx.draw_networkx_edges(G_sub, pos,
                           alpha=0.08,
                           arrows=False,
                           edge_color='#ffffff',
                           ax=ax)

    nx.draw_networkx_nodes(G_sub, pos,
                           node_size=node_sizes,
                           node_color=node_colors,
                           alpha=0.85,
                           ax=ax)

    # Label top 3 nodes per community
    pagerank = nx.pagerank(G_sub, alpha=0.85)
    labeled = set()
    for comm in top_communities:
        comm_nodes = [n for n in G_sub.nodes()
                      if partition.get(n) == comm]
        top_nodes = sorted(comm_nodes,
                           key=lambda n: pagerank.get(n, 0),
                           reverse=True)[:2]
        for node in top_nodes:
            if node not in labeled:
                x, y = pos[node]
                label = node.split('.')[0][:12]
                ax.annotate(label,
                            xy=(x, y),
                            xytext=(0, 14),
                            textcoords='offset points',
                            fontsize=7,
                            fontweight='bold',
                            color='white',
                            ha='center',
                            bbox=dict(boxstyle='round,pad=0.2',
                                      facecolor=color_map[comm],
                                      edgecolor='none',
                                      alpha=0.85))
                labeled.add(node)

    # Legend
    legend_elements = [
        mpatches.Patch(
            color=color_map[comm],
            label=f"{community_labels.get(comm, f'C{comm}')} ({community_sizes[comm]})"
        )
        for comm in top_communities
    ]
    ax.legend(handles=legend_elements,
              loc='lower left',
              facecolor='#1a1a2e',
              edgecolor='#4ECDC4',
              labelcolor='white',
              fontsize=9,
              framealpha=0.9,
              title="Communities",
              title_fontsize=10)

    ax.set_title("Community Structure — AI/Tech Bluesky Network\n"
                 "Each color = one detected community  |  "
                 "Size = degree centrality",
                 fontsize=13, color='white', pad=15)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig("outputs/community_network.png",
                dpi=150, bbox_inches='tight',
                facecolor='#0f0f1a')
    print("Saved: outputs/community_network.png")
    plt.show()

def plot_community_heatmap(partition, df, community_labels):
    print("\nPlotting community retweet heatmap...")

    # Get top 6 communities
    community_sizes = Counter(partition.values())
    top_communities = [c for c, _ in community_sizes.most_common(6)]

    # Map users to community labels
    user_to_comm = {}
    for user, comm in partition.items():
        if comm in top_communities:
            label = community_labels.get(comm, f"C{comm}")
            # Shorten label for display
            short = label.split('/')[0].strip()[:15]
            user_to_comm[user] = short

    # Get unique community labels in order
    comm_labels_ordered = []
    for comm in top_communities:
        label = community_labels.get(comm, f"C{comm}")
        short = label.split('/')[0].strip()[:15]
        if short not in comm_labels_ordered:
            comm_labels_ordered.append(short)

    n = len(comm_labels_ordered)
    matrix = np.zeros((n, n))

    # Fill matrix — for each post, if author is in a community
    # add their repost count to that community's row
    for _, row in df.iterrows():
        author = row['author']
        if author in user_to_comm:
            from_label = user_to_comm[author]
            if from_label in comm_labels_ordered:
                i = comm_labels_ordered.index(from_label)
                # Simulate cross-community interaction
                # using search term overlap
                same_term = df[df['search_term'] == row['search_term']]
                for _, other_row in same_term.iterrows():
                    other = other_row['author']
                    if other in user_to_comm and other != author:
                        to_label = user_to_comm[other]
                        if to_label in comm_labels_ordered:
                            j = comm_labels_ordered.index(to_label)
                            matrix[i][j] += 1

    # Normalize rows
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    matrix_norm = matrix / row_sums * 100

    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#0f0f1a')

    im = ax.imshow(matrix_norm, cmap='YlOrRd', aspect='auto')

    # Add value labels
    for i in range(n):
        for j in range(n):
            val = matrix_norm[i][j]
            color = 'white' if val > 50 else 'black'
            ax.text(j, i, f'{val:.0f}%',
                    ha='center', va='center',
                    fontsize=9, color=color,
                    fontweight='bold')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(comm_labels_ordered,
                       rotation=35, ha='right',
                       fontsize=9, color='white')
    ax.set_yticklabels(comm_labels_ordered,
                       fontsize=9, color='white')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(labelcolor='white')
    cbar.set_label('Interaction %', color='white', fontsize=10)

    ax.set_title("Community-to-Community Interaction Heatmap\n"
                 "AI/Tech Bluesky Network  |  "
                 "Darker = stronger interaction",
                 fontsize=13, color='white', pad=15)

    ax.set_xlabel("Interacted-with Community", color='white', fontsize=10)
    ax.set_ylabel("Source Community", color='white', fontsize=10)

    plt.tight_layout()
    plt.savefig("outputs/community_heatmap.png",
                dpi=150, bbox_inches='tight',
                facecolor='#0f0f1a')
    print("Saved: outputs/community_heatmap.png")
    plt.show()

if __name__ == "__main__":
    print("="*55)
    print("Community Detection — AI/Tech Bluesky Network")
    print("="*55)

    df = load_data()
    G = load_graph()

    partition, community_sizes = detect_communities(G)
    community_labels, community_users = label_communities(partition, df)

    plot_community_network(G, partition, community_labels)
    plot_community_heatmap(partition, df, community_labels)

    print("\nCommunity detection complete.")