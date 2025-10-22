import pandas as pd
import numpy as np
from wordcloud import WordCloud
from collections import Counter
import matplotlib.pyplot as plt
from pathlib import Path
import re

class EmojiCloud:
    def __init__(self,
                 font_path=None,
                 color='yellow'):
        self.font_path = font_path
        self.color = color
        self.word_cloud = self.initialize_wordcloud()
        self.emoji_probability = None

    def initialize_wordcloud(self):
        # Try to find a suitable emoji font, fall back to default if none found
        word_cloud_params = {
            'width': 2000,
            'height': 1000,
            'background_color': 'white',
            'random_state': 42,
            'collocations': False,
            'prefer_horizontal': 1,
            'relative_scaling': 0.25
        }
        
        # Try with the specific font path first
        font_paths_to_try = [
            'NotoEmoji-VariableFont_wght.ttf',
            'NotoColorEmoji.ttf',
            'Apple Color Emoji.ttc',
            'Segoe UI Emoji.ttf',
            '/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf',
            '/System/Library/Fonts/Apple Color Emoji.ttc',
            None  # Use system default as last resort
        ]
        
        for font_path in font_paths_to_try:
            try:
                if font_path:
                    # Check if font file exists before trying to use it
                    if not Path(font_path).exists():
                        print(f"Font not found: {font_path}")
                        continue
                    word_cloud_params['font_path'] = font_path
                    word_cloud = WordCloud(**word_cloud_params)
                    print(f"Using font: {font_path}")
                    return word_cloud
                else:
                    # Try without font_path
                    if 'font_path' in word_cloud_params:
                        del word_cloud_params['font_path']
                    word_cloud = WordCloud(**word_cloud_params)
                    print("Using system default font")
                    return word_cloud
            except (OSError, IOError) as e:
                if 'font_path' in word_cloud_params:
                    del word_cloud_params['font_path']
                print(f"Failed to use font {font_path}: {e}")
                continue
        
        # If all fail, try minimal parameters without font
        try:
            minimal_params = {
                'width': 2000,
                'height': 1000,
                'background_color': 'white',
                'random_state': 42
            }
            word_cloud = WordCloud(**minimal_params)
            print("Using minimal WordCloud parameters (no font)")
            return word_cloud
        except Exception as final_e:
            print(f"All font attempts failed: {final_e}")
            # Last resort: create a simple wordcloud without any font requirements
            try:
                word_cloud = WordCloud(width=2000, height=1000, background_color='white')
                print("Using basic WordCloud (no font, no advanced features)")
                return word_cloud
            except Exception as ultimate_e:
                print(f"Even basic WordCloud failed: {ultimate_e}")
                raise ultimate_e

    def color_func(self, word, font_size, position, orientation, random_state=None,
                   **kwargs):
        hue_saturation = {
            'yellow': '42, 88%',
            'blue': '194, 60%',
            'green': '159, 55%',
            'grey': '45, 2%',
            'red': '0, 50%'
        }.get(self.color)

        current_emoji_probability = self.emoji_probability[word]
        # Use 50% opacity for emojis with 20% or more coverage
        if current_emoji_probability >= 0.20:
            opacity = 50
        else:
            # Use an opacity between 70 to 75 for other emojis
            opacity = 75 - current_emoji_probability/0.2 * 5
        return f"hsl({hue_saturation},{opacity}%)"

    def generate(self, emojis):
        emoji_frequencies = Counter(emojis)
        total_count = len(emojis)
        
        self.emoji_probability = {emoji: count/total_count for emoji, count in emoji_frequencies.items()}
        
        wc = self.word_cloud.generate_from_frequencies(emoji_frequencies)
        
        plt.imshow(wc.recolor(color_func=self.color_func, random_state=42),
                   interpolation="bilinear")
        plt.axis("off")

def create_emoji_wordclouds(input_csv_path, output_dir):
    print(f"Loading emoji data from {input_csv_path}...")
    df = pd.read_csv(input_csv_path)
    
    # Filter by sentiment
    bullish_df = df[df['label'] == 'bullish'].copy()
    bearish_df = df[df['label'] == 'bearish'].copy()
    
    print(f"Found {len(bullish_df)} bullish posts and {len(bearish_df)} bearish posts")
    
    # Extract emojis from text
    emoji_pattern = re.compile(
        "[\U0001F1E6-\U0001F1FF\U0001F300-\U0001F5FF\U0001F600-\U0001F64F"
        "\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FAFF"
        "\u2600-\u26FF\u2700-\u27BF]"
    )
    
    def extract_emojis(text):
        if pd.isna(text):
            return []
        return emoji_pattern.findall(str(text))
    
    bullish_emojis = bullish_df['text'].apply(extract_emojis).explode().dropna()
    bearish_emojis = bearish_df['text'].apply(extract_emojis).explode().dropna()
    
    print(f"Extracted {len(bullish_emojis)} bullish emojis and {len(bearish_emojis)} bearish emojis")
    
    # Create wordclouds
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Bullish emoji wordcloud (green)
    if len(bullish_emojis) > 0:
        plt.sca(axes[0])
        emoji_cloud_bullish = EmojiCloud(font_path=None, color='green')
        emoji_cloud_bullish.generate(bullish_emojis.tolist())
        plt.title('Bullish Emojis', fontsize=16, fontweight='bold')
    
    # Bearish emoji wordcloud (red)
    if len(bearish_emojis) > 0:
        plt.sca(axes[1])
        emoji_cloud_bearish = EmojiCloud(font_path=None, color='red')
        emoji_cloud_bearish.generate(bearish_emojis.tolist())
        plt.title('Bearish Emojis', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / "emoji_wordclouds.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Emoji wordclouds saved to {output_path}")
    plt.show()
    
    # Create individual wordclouds
    if len(bullish_emojis) > 0:
        plt.figure(figsize=(12, 8))
        emoji_cloud_bullish = EmojiCloud(font_path=None, color='green')
        emoji_cloud_bullish.generate(bullish_emojis.tolist())
        plt.title('Bullish Emojis Wordcloud', fontsize=16, fontweight='bold')
        
        bullish_path = Path(output_dir) / "bullish_emoji_wordcloud.png"
        plt.savefig(bullish_path, dpi=300, bbox_inches='tight')
        print(f"Bullish emoji wordcloud saved to {bullish_path}")
        plt.show()
    
    if len(bearish_emojis) > 0:
        plt.figure(figsize=(12, 8))
        emoji_cloud_bearish = EmojiCloud(font_path=None, color='red')
        emoji_cloud_bearish.generate(bearish_emojis.tolist())
        plt.title('Bearish Emojis Wordcloud', fontsize=16, fontweight='bold')
        
        bearish_path = Path(output_dir) / "bearish_emoji_wordcloud.png"
        plt.savefig(bearish_path, dpi=300, bbox_inches='tight')
        print(f"Bearish emoji wordcloud saved to {bearish_path}")
        plt.show()
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to emoji_only.csv")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    create_emoji_wordclouds(args.input, args.output_dir)
