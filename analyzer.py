"""
Stock Sentiment Analyzer - Sentiment Analysis Module
"""
from transformers import pipeline
import plotly.express as px
import pandas as pd
import streamlit as st

class StockSentimentAnalyzer:
    def __init__(self):
        # Load sentiment analysis model
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True,
            max_length=512
        )

    def analyze_articles(self, articles):
        """Analyze sentiment for a list of articles"""
        results = {
            'articles': [],
            'positive_count': 0,
            'negative_count': 0,
            'composite_score': 0
        }

        # Collect valid texts for batch inference
        valid = [
            (article, article.get('description') or article.get('title', ''))
            for article in articles
        ]
        valid = [(a, t) for a, t in valid if t]

        if not valid:
            return results

        try:
            batch_results = self.sentiment_analyzer(
                [t for _, t in valid], batch_size=16
            )
        except Exception as e:
            print(f"Error running batch inference: {e}")
            return results

        for (article, _), sentiment_result in zip(valid, batch_results):
            try:
                label = sentiment_result['label']
                score = sentiment_result['score']

                # score is confidence in [0.5, 1.0] for the predicted label.
                # Map to [-100, +100]: positive -> [0, 100], negative -> [-100, 0].
                sign = 1 if label == 'POSITIVE' else -1
                normalized_score = sign * int((score - 0.5) * 200)
                sentiment = 'Positive' if label == 'POSITIVE' else 'Negative'

                if label == 'POSITIVE':
                    results['positive_count'] += 1
                else:
                    results['negative_count'] += 1

                results['articles'].append({
                    'title': article.get('title', ''),
                    'source': article.get('source', 'Unknown'),
                    'description': article.get('description', ''),
                    'url': article.get('url', ''),
                    'publishedAt': article.get('publishedAt', ''),
                    'sentiment': sentiment,
                    'score': normalized_score,
                    'raw_score': score
                })

            except Exception as e:
                print(f"Error processing article result: {e}")
                continue

        # Calculate composite score
        if results['articles']:
            total = sum(a['score'] for a in results['articles'])
            results['composite_score'] = int(total / len(results['articles']))

            # Determine overall sentiment
            if results['composite_score'] > 20:
                results['overall_sentiment'] = 'Positive'
            elif results['composite_score'] < -20:
                results['overall_sentiment'] = 'Negative'
            else:
                results['overall_sentiment'] = 'Neutral'

        return results

    def plot_sentiment_distribution(self, results):
        """Create visualization of sentiment distribution"""
        if not results['articles']:
            return

        # Create dataframe
        df = pd.DataFrame(results['articles'])

        # Bar chart of scores
        fig = px.bar(
            df,
            y='score',
            x=list(range(len(df))),
            color='score',
            color_continuous_scale='RdYlGn',
            range_color=[-100, 100],
            title="Article Sentiment Scores",
            labels={'x': 'Article', 'score': 'Sentiment Score'},
        )
        fig.update_layout(
            yaxis_range=[-100, 100],
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

        # Pie chart
        col1, col2 = st.columns(2)
        with col1:
            sentiment_counts = df['sentiment'].value_counts()
            fig_pie = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Sentiment Distribution",
                color_discrete_map={
                    'Positive': '#22c55e',
                    'Negative': '#ef4444',
                    'Neutral': '#6b7280'
                }
            )
            st.plotly_chart(fig_pie, use_container_width=True)
