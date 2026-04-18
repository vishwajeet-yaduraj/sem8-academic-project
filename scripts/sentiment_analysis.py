import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
from datetime import datetime

INPUT_CSV = "data/ai_tech_posts.csv"

def load_data():
    df = pd.read_csv(INPUT_CSV)
    df['created_at'] = pd.to_datetime(df['created_at'], format='mixed', utc=True)
    print(f"Loaded {len(df)} posts")
    print(f"Date range: {df['created_at'].min()} to {df['created_at'].max()}")
    return df

def classify_stance(compound, text):
    text_lower = text.lower()

    # Meme indicators
    meme_words = ['lol', 'lmao', 'bruh', '💀', '😂', '🤣',
                  'imagine', 'bro', 'skill issue', 'ratio',
                  'touch grass', 'based', 'cope', 'slay']
    if any(w in text_lower for w in meme_words):
        return 'Meme'

    # Technical indicators
    tech_words = ['parameter', 'benchmark', 'token', 'inference',
                  'training', 'dataset', 'accuracy', 'model weights',
                  'fine-tun', 'gradient', 'architecture', 'latency',
                  'transformer', 'embedding', 'attention mechanism']
    if any(w in text_lower for w in tech_words):
        return 'Technical'

    # Skeptic indicators
    skeptic_words = ['overhyped', 'bubble', 'scam', 'waste',
                     'useless', 'garbage', 'hype', 'replace',
                     'job loss', 'dangerous', 'scary', 'worried',
                     'concern', 'problem', 'issue', 'fail']
    if any(w in text_lower for w in skeptic_words) and compound < 0.1:
        return 'Skeptic'

    # VADER compound score for remaining
    if compound >= 0.3:
        return 'Hype'
    elif compound <= -0.2:
        return 'Skeptic'
    else:
        return 'Neutral'

def run_sentiment(df):
    print("\nRunning VADER sentiment analysis...")
    analyzer = SentimentIntensityAnalyzer()

    stances = []
    compounds = []

    for text in df['text']:
        scores = analyzer.polarity_scores(str(text))
        compound = scores['compound']
        stance = classify_stance(compound, str(text))
        stances.append(stance)
        compounds.append(compound)

    df['compound'] = compounds
    df['stance'] = stances

    stance_counts = Counter(stances)
    print("\nStance distribution:")
    total = len(stances)
    for stance, count in stance_counts.most_common():
        pct = count / total * 100
        print(f"  {stance}: {count} ({pct:.1f}%)")

    return df

def plot_stance_distribution(df):
    print("\nPlotting stance distribution...")

    stance_colors = {
        'Hype':      '#4ECDC4',
        'Neutral':   '#95A5A6',
        'Skeptic':   '#FF6B35',
        'Meme':      '#FFEAA7',
        'Technical': '#A29BFE'
    }

    stance_counts = df['stance'].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('#0f0f1a')

    # Bar chart
    ax1 = axes[0]
    ax1.set_facecolor('#0f0f1a')
    bars = ax1.bar(
        stance_counts.index,
        stance_counts.values,
        color=[stance_colors.get(s, '#888888') for s in stance_counts.index],
        edgecolor='none',
        alpha=0.9
    )

    for bar, val in zip(bars, stance_counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 20,
                 str(val),
                 ha='center', va='bottom',
                 color='white', fontsize=11,
                 fontweight='bold')

    ax1.set_facecolor('#0f0f1a')
    ax1.tick_params(colors='white')
    ax1.spines['bottom'].set_color('#444444')
    ax1.spines['left'].set_color('#444444')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_title("Stance Distribution\nAll Posts",
                  color='white', fontsize=12, pad=10)
    ax1.set_ylabel("Post Count", color='white', fontsize=10)
    ax1.set_xlabel("Stance", color='white', fontsize=10)

    # Pie chart
    ax2 = axes[1]
    ax2.set_facecolor('#0f0f1a')
    wedge_colors = [stance_colors.get(s, '#888888')
                    for s in stance_counts.index]
    wedges, texts, autotexts = ax2.pie(
        stance_counts.values,
        labels=stance_counts.index,
        colors=wedge_colors,
        autopct='%1.1f%%',
        startangle=140,
        pctdistance=0.75
    )
    for text in texts:
        text.set_color('white')
        text.set_fontsize(11)
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')

    ax2.set_title("Stance Distribution\nProportional",
                  color='white', fontsize=12, pad=10)

    plt.suptitle("Sentiment and Stance Analysis — AI/Tech Bluesky Network",
                 color='white', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("outputs/stance_distribution.png",
                dpi=150, bbox_inches='tight',
                facecolor='#0f0f1a')
    print("Saved: outputs/stance_distribution.png")
    plt.show()

def plot_temporal_stance(df):
    print("\nPlotting temporal stance evolution...")

    stance_colors = {
        'Hype':      '#4ECDC4',
        'Neutral':   '#95A5A6',
        'Skeptic':   '#FF6B35',
        'Meme':      '#FFEAA7',
        'Technical': '#A29BFE'
    }

    # Group by 6-hour bins
    df['time_bin'] = df['created_at'].dt.floor('6h')
    temporal = df.groupby(['time_bin', 'stance']).size().unstack(fill_value=0)

    # Make sure all stances are present
    for stance in stance_colors:
        if stance not in temporal.columns:
            temporal[stance] = 0

    temporal = temporal[list(stance_colors.keys())]

    fig, ax = plt.subplots(figsize=(16, 8))
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#0f0f1a')

    # Stacked area chart
    ax.stackplot(
        temporal.index,
        [temporal[s] for s in stance_colors.keys()],
        labels=list(stance_colors.keys()),
        colors=list(stance_colors.values()),
        alpha=0.85
    )

    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('#444444')
    ax.spines['left'].set_color('#444444')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_title("Temporal Evolution of Stance\nAI/Tech Bluesky Network — 6-hour bins",
                 color='white', fontsize=13, pad=15)
    ax.set_xlabel("Time", color='white', fontsize=10)
    ax.set_ylabel("Number of Posts", color='white', fontsize=10)

    # Format x-axis dates
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
    plt.xticks(rotation=0)

    legend = ax.legend(loc='upper left',
                       facecolor='#1a1a2e',
                       edgecolor='#4ECDC4',
                       labelcolor='white',
                       fontsize=10,
                       title='Stance',
                       title_fontsize=11)

    plt.tight_layout()
    plt.savefig("outputs/temporal_stance.png",
                dpi=150, bbox_inches='tight',
                facecolor='#0f0f1a')
    print("Saved: outputs/temporal_stance.png")
    plt.show()

def plot_stance_by_community(df):
    print("\nPlotting stance by search term...")

    stance_colors = {
        'Hype':      '#4ECDC4',
        'Neutral':   '#95A5A6',
        'Skeptic':   '#FF6B35',
        'Meme':      '#FFEAA7',
        'Technical': '#A29BFE'
    }

    # Group by search term and stance
    term_stance = df.groupby(['search_term', 'stance']).size().unstack(fill_value=0)

    for stance in stance_colors:
        if stance not in term_stance.columns:
            term_stance[stance] = 0

    term_stance = term_stance[list(stance_colors.keys())]

    # Normalize to percentages
    term_stance_pct = term_stance.div(term_stance.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(16, 8))
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#0f0f1a')

    bottom = np.zeros(len(term_stance_pct))
    for stance, color in stance_colors.items():
        values = term_stance_pct[stance].values
        bars = ax.bar(term_stance_pct.index, values,
                      bottom=bottom, color=color,
                      label=stance, alpha=0.9,
                      edgecolor='none')
        bottom += values

    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('#444444')
    ax.spines['left'].set_color('#444444')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xticks(rotation=35, ha='right', color='white', fontsize=9)
    plt.yticks(color='white')

    ax.set_title("Stance Distribution by Search Term\nAI/Tech Bluesky Network",
                 color='white', fontsize=13, pad=15)
    ax.set_ylabel("Percentage of Posts (%)", color='white', fontsize=10)
    ax.set_xlabel("Search Term", color='white', fontsize=10)

    legend = ax.legend(loc='upper right',
                       facecolor='#1a1a2e',
                       edgecolor='#4ECDC4',
                       labelcolor='white',
                       fontsize=10)

    plt.tight_layout()
    plt.savefig("outputs/stance_by_term.png",
                dpi=150, bbox_inches='tight',
                facecolor='#0f0f1a')
    print("Saved: outputs/stance_by_term.png")
    plt.show()

if __name__ == "__main__":
    print("="*55)
    print("Sentiment Analysis — AI/Tech Bluesky Network")
    print("="*55)

    df = load_data()
    df = run_sentiment(df)

    plot_stance_distribution(df)
    plot_temporal_stance(df)
    plot_stance_by_community(df)

    # Save enriched dataset
    df.to_csv("data/posts_with_sentiment.csv", index=False)
    print("\nSaved enriched dataset: data/posts_with_sentiment.csv")
    print("\nSentiment analysis complete.")