import numpy as np
import pandas as pd
import gradio as gr

import faiss
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer


# ===== Prepare model & data =====
passages_path = hf_hub_download(repo_id='YuITC/Vietnamese-Legal-Doc-Retrieval-Data',
                                filename='corpus_data.parquet', repo_type='dataset', 
                                local_dir='demo')

index_path = hf_hub_download(repo_id='YuITC/Vietnamese-Legal-Doc-Retrieval-Data', 
                             filename='legal_faiss.index', repo_type='dataset', 
                             local_dir='demo')

emb_model   = SentenceTransformer('YuITC/bert-base-multilingual-cased-finetuned-VNLegalDocs')
passages    = pd.read_parquet(passages_path)['text'].tolist()
legal_index = faiss.read_index(index_path)


# ===== Utility functions =====
def retrieval(emb_model, query, index, top_k=10):
    q_emb = emb_model.encode(
        query, 
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32).reshape(1, -1)
    
    scores, indices = index.search(q_emb, top_k)
    cand_idxs       = indices[0]
    cand_scores     = scores[0]
    cand_texts      = [passages[i] for i in cand_idxs]

    return [{'index': int(cand_idxs[i]),
             'score': float(cand_scores[i]),
             'text' : cand_texts[i]
            } for i in range(len(cand_idxs))]

def get_results(query, top_k):
    hits = retrieval(emb_model, query, legal_index, top_k=top_k)
    
    result = ""
    for rank, h in enumerate(hits, start=1):
        result += f"[Kết quả {rank} - Độ tin cậy={h['score']:.4f}]\n\n{h['text']}\n{'-'*100}\n"
    return result

    
# ===== Gradio UI =====
demo = gr.Interface(
    fn=get_results,
    inputs=[
        gr.Textbox(lines=2, placeholder='Nhập câu hỏi pháp lý của bạn...', label='Câu hỏi'),
        gr.Slider(minimum=5, maximum=20, value=10, step=1, label='Số lượng kết quả'),
    ],
    outputs=gr.Textbox(lines=20, label='Kết quả'),
    title='Vietnamese Legal Document Retrieval System',
    description='🔍 Nhập câu hỏi pháp lý của bạn bằng tiếng Việt để nhận các đoạn văn bản pháp luật liên quan.',
    examples=[
        ['Tội xúc phạm danh dự?'],
        ['Quyền lợi của người lao động?'],
        ['Thủ tục đăng ký kết hôn?'],
    ],
    flagging_mode='never'
)

if __name__ == '__main__':
    demo.launch()