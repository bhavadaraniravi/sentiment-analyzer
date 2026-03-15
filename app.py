from textblob import TextBlob
import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import csv
import io

def analyze_text(text):
    if not text.strip():
        return None, None

    sentences = [s.strip() for s in text.replace('!','!|').replace('?','?|').replace('.','.|').split('|') if s.strip()]
    if len(sentences) == 0:
        return None, None

    scores = []
    colors = []
    sentence_results = []
    pos_count = neg_count = neu_count = 0
    total_score = 0

    for i, sentence in enumerate(sentences):
        blob = TextBlob(sentence)
        score = round(blob.sentiment.polarity, 2)
        subjectivity = round(blob.sentiment.subjectivity, 2)
        scores.append(score)
        total_score += score

        if score >= 0.5:
            label = "Very Positive"
            colors.append('#00e676')
            pos_count += 1
        elif score > 0.1:
            label = "Positive"
            colors.append('#00c853')
            pos_count += 1
        elif score <= -0.5:
            label = "Very Negative"
            colors.append('#ff1744')
            neg_count += 1
        elif score < -0.1:
            label = "Negative"
            colors.append('#d50000')
            neg_count += 1
        else:
            label = "Neutral"
            colors.append('#aa00ff')
            neu_count += 1

        sentence_results.append(
            f"S{i+1} {label} | Score: {score} | Subjectivity: {subjectivity}\n"
            f"    -> \"{sentence}\""
        )

    avg = round(total_score / len(sentences), 2)

    if avg >= 0.5:    overall = "Very Positive"
    elif avg > 0.1:   overall = "Positive"
    elif avg <= -0.5: overall = "Very Negative"
    elif avg < -0.1:  overall = "Negative"
    else:             overall = "Neutral"

    emotions = {
        "joy":      ["love","happy","wonderful","great","amazing","fantastic","excited","best"],
        "anger":    ["hate","angry","furious","terrible","awful","horrible","worst","rage"],
        "sadness":  ["sad","crying","depressed","unhappy","miserable","disappointed","upset"],
        "fear":     ["scared","afraid","worried","anxious","nervous","terrified","panic"],
        "surprise": ["wow","amazing","incredible","unbelievable","shocked","astonished"]
    }
    words = re.findall(r'\b\w+\b', text.lower())
    emotion_scores = {e: 0 for e in emotions}
    for word in words:
        for emotion, keywords in emotions.items():
            if word in keywords:
                emotion_scores[emotion] += 1
    top_emotion = max(emotion_scores, key=emotion_scores.get)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.patch.set_facecolor('#0d0d1a')
    fig.suptitle('Sentiment Analysis Report', color='white', fontsize=14, fontweight='bold')

    ax1 = axes[0, 0]
    ax1.set_facecolor('#1a1a2e')
    x = range(len(sentences))
    bars = ax1.bar(x, scores, color=colors, edgecolor='white', linewidth=0.5, width=0.5)
    ax1.axhline(y=0, color='white', linewidth=1, linestyle='--', alpha=0.5)
    ax1.set_title('Score Per Sentence', color='white', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'S{i+1}' for i in x], color='white')
    ax1.set_ylim(-1.2, 1.2)
    ax1.tick_params(colors='white')
    for bar, score in zip(bars, scores):
        ypos = bar.get_height()+0.05 if score >= 0 else bar.get_height()-0.15
        ax1.text(bar.get_x()+bar.get_width()/2, ypos,
                str(score), ha='center', color='white', fontsize=9, fontweight='bold')

    ax2 = axes[0, 1]
    ax2.set_facecolor('#1a1a2e')
    pie_data = [(pos_count,'Positive','#00c853'),
                (neg_count,'Negative','#ff1744'),
                (neu_count,'Neutral','#aa00ff')]
    pie_data = [(v,l,c) for v,l,c in pie_data if v > 0]
    ax2.pie(
        [d[0] for d in pie_data],
        labels=[f"{d[1]} ({d[0]})" for d in pie_data],
        colors=[d[2] for d in pie_data],
        autopct='%1.0f%%',
        textprops={'color':'white','fontsize':10},
        wedgeprops={'edgecolor':'white','linewidth':1.2},
        startangle=90
    )
    ax2.set_title('Sentiment Distribution', color='white', fontsize=11)

    ax3 = axes[1, 0]
    ax3.set_facecolor('#1a1a2e')
    emo_colors = ['#ffd600','#ff1744','#2979ff','#aa00ff','#ff6d00']
    ax3.bar(list(emotion_scores.keys()), list(emotion_scores.values()),
            color=emo_colors, edgecolor='white', linewidth=0.5)
    ax3.set_title('Emotion Detection', color='white', fontsize=11)
    ax3.tick_params(colors='white')

    ax4 = axes[1, 1]
    ax4.set_facecolor('#1a1a2e')
    clean_text = re.sub(r'[^\w\s]', '', text)
    if clean_text.strip():
        wc = WordCloud(width=400, height=200,
                      background_color='#1a1a2e',
                      colormap='cool', max_words=30).generate(clean_text)
        ax4.imshow(wc, interpolation='bilinear')
    ax4.axis('off')
    ax4.set_title('Word Cloud', color='white', fontsize=11)

    plt.tight_layout()

    report = "==============================\n"
    report += "SENTENCE BREAKDOWN\n"
    report += "==============================\n"
    report += "\n".join(sentence_results)
    report += "\n\n==============================\n"
    report += "FINAL SUMMARY\n"
    report += "==============================\n"
    report += f"Overall Sentiment  : {overall}\n"
    report += f"Average Score      : {avg}\n"
    report += f"Positive sentences : {pos_count}\n"
    report += f"Negative sentences : {neg_count}\n"
    report += f"Neutral sentences  : {neu_count}\n"
    report += f"Total sentences    : {len(sentences)}\n"
    report += f"Total words        : {len(words)}\n"
    report += f"Dominant Emotion   : {top_emotion.upper()}\n"
    report += "=============================="

    return report, fig


def analyze_csv(file):
    if file is None:
        return "Please upload a CSV file!", None

    try:
        content = open(file.name, 'r', encoding='utf-8').read()
        reader = csv.reader(io.StringIO(content))
        rows = list(reader)

        if len(rows) == 0:
            return "CSV file is empty!", None

        all_text = []
        pos = neg = neu = 0
        scores_list = []

        for i, row in enumerate(rows):
            if i == 0 and any(h.lower() in ['text','review','comment','feedback']
                             for h in row):
                continue
            if row:
                text = ' '.join(row)
                blob = TextBlob(text)
                score = round(blob.sentiment.polarity, 2)
                scores_list.append(score)
                all_text.append(text)
                if score > 0.1:    pos += 1
                elif score < -0.1: neg += 1
                else:              neu += 1

        if len(all_text) == 0:
            return "No text found in CSV!", None

        avg = round(sum(scores_list) / len(scores_list), 2)

        if avg >= 0.5:    overall = "Very Positive"
        elif avg > 0.1:   overall = "Positive"
        elif avg <= -0.5: overall = "Very Negative"
        elif avg < -0.1:  overall = "Negative"
        else:             overall = "Neutral"

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        fig.patch.set_facecolor('#0d0d1a')
        fig.suptitle(f'CSV Analysis - {len(all_text)} Reviews',
                    color='white', fontsize=13, fontweight='bold')

        ax1.set_facecolor('#1a1a2e')
        ax1.hist(scores_list, bins=10, color='#00c853', edgecolor='white', linewidth=0.5)
        ax1.axvline(x=0, color='white', linewidth=1, linestyle='--')
        ax1.set_title('Score Distribution', color='white', fontsize=11)
        ax1.tick_params(colors='white')
        ax1.set_xlabel('Polarity Score', color='white')
        ax1.set_ylabel('Number of Reviews', color='white')

        ax2.set_facecolor('#1a1a2e')
        pie_data = [(pos,'Positive','#00c853'),
                    (neg,'Negative','#ff1744'),
                    (neu,'Neutral','#aa00ff')]
        pie_data = [(v,l,c) for v,l,c in pie_data if v > 0]
        ax2.pie(
            [d[0] for d in pie_data],
            labels=[f"{d[1]} ({d[0]})" for d in pie_data],
            colors=[d[2] for d in pie_data],
            autopct='%1.0f%%',
            textprops={'color':'white','fontsize':10},
            wedgeprops={'edgecolor':'white','linewidth':1.2},
            startangle=90
        )
        ax2.set_title('Overall Distribution', color='white', fontsize=11)
        plt.tight_layout()

        report = "==============================\n"
        report += "CSV ANALYSIS REPORT\n"
        report += "==============================\n"
        report += f"Total reviews      : {len(all_text)}\n"
        report += f"Overall Sentiment  : {overall}\n"
        report += f"Average Score      : {avg}\n"
        report += f"Positive reviews   : {pos}\n"
        report += f"Negative reviews   : {neg}\n"
        report += f"Neutral reviews    : {neu}\n"
        report += "==============================\n"
        report += "FIRST 5 REVIEWS\n"
        report += "==============================\n"
        for i, (t, s) in enumerate(zip(all_text[:5], scores_list[:5])):
            label = "Positive" if s > 0.1 else "Negative" if s < -0.1 else "Neutral"
            report += f"[{label}] Score:{s} | {t[:60]}\n"
        report += "=============================="

        return report, fig

    except Exception as e:
        return f"Error reading file: {str(e)}", None


text_app = gr.Interface(
    fn=analyze_text,
    inputs=gr.Textbox(
        placeholder="Type any paragraph, review, tweet...",
        label="Enter Your Text",
        lines=6
    ),
    outputs=[
        gr.Textbox(label="Analysis Report", lines=20),
        gr.Plot(label="Visual Dashboard")
    ],
    examples=[
        ["I love this college! Canteen food is terrible. Teachers are amazing!"],
        ["This phone is fantastic! Battery life is poor. Camera is outstanding!"],
        ["I am so angry today! But my friends helped me. Overall okay day."],
    ],
    title="Text Analysis"
)

csv_app = gr.Interface(
    fn=analyze_csv,
    inputs=gr.File(
        label="Upload CSV File"
    ),
    outputs=[
        gr.Textbox(label="CSV Report", lines=20),
        gr.Plot(label="Charts")
    ],
    title="CSV File Analysis"
)

app = gr.TabbedInterface(
    [text_app, csv_app],
    ["Text Analysis", "CSV Analysis"],
    title="Sentiment Analyzer Pro",
)

app.launch()
