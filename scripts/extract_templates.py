import pandas as pd
import json
import re
from collections import Counter
import os

def extract_templates(csv_path, output_path, top_n=50, min_freq=10):
    """
    Trích xuất các câu phổ biến nhất từ dataset báo cáo y tế.
    """
    if not os.path.exists(csv_path):
        print(f"Lỗi: Không tìm thấy file {csv_path}")
        return

    print(f"Reading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    all_sentences = []
    
    for report in df['report'].dropna():
        # Split report into sentences based on dot, semicolon, question mark
        sentences = re.split(r'[.;?!\n]', str(report))
        for s in sentences:
            s = s.strip()
            # Remove very short or non-meaningful sentences
            if len(s.split()) > 3:
                all_sentences.append(s)
                
    # Count frequency
    counts = Counter(all_sentences)
    
    # Select top N templates with frequency >= min_freq
    top_templates = [s for s, count in counts.most_common(top_n) if count >= min_freq]
    
    print(f"Found {len(top_templates)} most common sentence templates.")
    
    # Save to JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(top_templates, f, ensure_ascii=False, indent=4)
        
    print(f"Saved templates to {output_path}")
    
    # Print top 5 examples
    print("\nTop 5 template examples:")
    for i, t in enumerate(top_templates[:5]):
        print(f"{i+1}. {t} (Freq: {counts[t]})")

if __name__ == "__main__":
    csv_file = "data/processed/iu_xray_dataset_raw.csv"
    output_file = "data/processed/templates.json"
    
    # Adjust top_n depending on desired template diversity
    extract_templates(csv_file, output_file, top_n=100, min_freq=20)
