import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import numpy as np

INPUT_GRAPH = "data/interaction_graph.json"

def load_graph():
    with open(INPUT_GRAPH, 'r') as f:
        data = json.load(f)
    G = nx.node_link_graph(data)
    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

def get_tiers(pagerank, top_n=10, mid_n=40):
    pr_sorted = sorted(pagerank.values(), reverse=True)
    top_threshold = pr_sorted[min(top_n, len(pr_sorted)-1)]
    mid_threshold = pr_sorted[min(mid_n, len(pr_sorted)-1)]
    return top_threshold, mid_threshold

def plot_full_network(G):
    print("Plotting full network...")

    pagerank = nx.pagerank(G, alpha=0.85)
    top_threshold, mid_threshold = get_tiers(pagerank)

    node_colors = []
    node_sizes = []
    for n in G.nodes():
        pr = pagerank[n]
        if pr >= top_threshold:
            node_colors.append('#FF6B35')
            node_sizes.append(500)
        elif pr >= mid_threshold:
            node_colors.append('#4ECDC4')
            node_sizes.append(80)
        else:
            node_colors.append('#2d6a7f')
            node_sizes.append(20)

    fig, ax = plt.subplots(figsize=(18, 13))
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#0f0f1a')

    pos = nx.spring_layout(G, k=0.5, seed=42)

    nx.draw_networkx_edges(G, pos,
                           alpha=0.06,
                           arrows=False,
                           edge_color='#ffffff',
                           ax=ax)

    nx.draw_networkx_nodes(G, pos,
                           node_size=node_sizes,
                           node_color=node_colors,
                           alpha=0.9,
                           ax=ax)

    # Top 10 labels — spread them out with adjustable offsets
    top_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Manual offset directions to avoid overlap
    offsets = [
        (0, 18), (0, -22), (18, 10), (-18, 10),
        (18, -10), (-18, -10), (0, 26), (26, 0),
        (-26, 0), (14, 18)
    ]
    
    for i, (node, score) in enumerate(top_nodes):
        x, y = pos[node]
        ox, oy = offsets[i]
        label = node.split('.')[0][:12]
        ax.annotate(label,
                    xy=(x, y),
                    xytext=(ox, oy),
                    textcoords='offset points',
                    fontsize=8,
                    fontweight='bold',
                    color='white',
                    ha='center',
                    va='center',
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='#FF6B35',
                              edgecolor='white',
                              linewidth=0.5,
                              alpha=0.9),
                    arrowprops=dict(arrowstyle='-',
                                   color='white',
                                   alpha=0.4,
                                   lw=0.8))

    legend_elements = [
        mpatches.Patch(color='#FF6B35', label='Top influencers (PageRank top 10)'),
        mpatches.Patch(color='#4ECDC4', label='Mid-tier users (top 11-40)'),
        mpatches.Patch(color='#2d6a7f', label='General users'),
    ]
    ax.legend(handles=legend_elements,
              loc='lower left',
              facecolor='#1a1a2e',
              edgecolor='#4ECDC4',
              labelcolor='white',
              fontsize=10,
              framealpha=0.9)

    ax.set_title("AI/Tech Bluesky Network — Full Graph\n"
                 "Node color = influence tier  |  Size = PageRank  |  "
                 f"{G.number_of_nodes()} users  |  {G.number_of_edges()} edges",
                 fontsize=13,
                 color='white',
                 pad=15)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig("outputs/network_graph.png",
                dpi=150,
                bbox_inches='tight',
                facecolor='#0f0f1a')
    print("Saved: outputs/network_graph.png")
    plt.show()

def plot_largest_component(G):
    print("Plotting largest connected component...")

    undirected = G.to_undirected()
    components = list(nx.connected_components(undirected))
    largest = max(components, key=len)
    G_sub = G.subgraph(largest).copy()

    pagerank = nx.pagerank(G_sub, alpha=0.85)
    top_threshold, mid_threshold = get_tiers(pagerank, top_n=8, mid_n=35)

    node_colors = []
    node_sizes = []
    for n in G_sub.nodes():
        pr = pagerank[n]
        if pr >= top_threshold:
            node_colors.append('#FF6B35')
            node_sizes.append(900)
        elif pr >= mid_threshold:
            node_colors.append('#4ECDC4')
            node_sizes.append(180)
        else:
            node_colors.append('#2d6a7f')
            node_sizes.append(40)

    fig, ax = plt.subplots(figsize=(18, 13))
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#0f0f1a')

    pos = nx.spring_layout(G_sub, k=1.2, seed=42)

    nx.draw_networkx_edges(G_sub, pos,
                           alpha=0.12,
                           arrows=True,
                           arrowsize=6,
                           edge_color='#3a3a5a',
                           ax=ax)

    nx.draw_networkx_nodes(G_sub, pos,
                           node_size=node_sizes,
                           node_color=node_colors,
                           alpha=0.95,
                           ax=ax)

    # Top 12 labels with spread offsets
    top_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:12]
    offsets = [
        (0, 22), (22, 12), (-22, 12), (0, -24),
        (24, -10), (-24, -10), (18, 20), (-18, 20),
        (26, 0), (-26, 0), (14, -20), (-14, -20)
    ]

    for i, (node, score) in enumerate(top_nodes):
        x, y = pos[node]
        ox, oy = offsets[i]
        label = node.split('.')[0][:14]
        pr = pagerank[node]
        color = '#FF6B35' if pr >= top_threshold else '#4ECDC4'
        fontsize = 9 if pr >= top_threshold else 8

        ax.annotate(label,
                    xy=(x, y),
                    xytext=(ox, oy),
                    textcoords='offset points',
                    fontsize=fontsize,
                    fontweight='bold',
                    color='white',
                    ha='center',
                    va='center',
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor=color,
                              edgecolor='white',
                              linewidth=0.5,
                              alpha=0.9),
                    arrowprops=dict(arrowstyle='-',
                                   color='white',
                                   alpha=0.4,
                                   lw=0.8))

    legend_elements = [
        mpatches.Patch(color='#FF6B35', label='Top influencers (PageRank top 8)'),
        mpatches.Patch(color='#4ECDC4', label='Mid-tier users'),
        mpatches.Patch(color='#2d6a7f', label='General users'),
    ]
    ax.legend(handles=legend_elements,
              loc='lower left',
              facecolor='#1a1a2e',
              edgecolor='#4ECDC4',
              labelcolor='white',
              fontsize=10,
              framealpha=0.9)

    ax.set_title(f"Largest Connected Component — {len(largest)} users\n"
                 f"AI/Tech Bluesky Network  |  "
                 f"{G_sub.number_of_edges()} edges",
                 fontsize=13,
                 color='white',
                 pad=15)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig("outputs/largest_component.png",
                dpi=150,
                bbox_inches='tight',
                facecolor='#0f0f1a')
    print("Saved: outputs/largest_component.png")
    plt.show()

if __name__ == "__main__":
    G = load_graph()
    plot_full_network(G)
    plot_largest_component(G)
    print("\nVisualization complete.")