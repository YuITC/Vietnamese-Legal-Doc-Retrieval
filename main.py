import os
import numpy as np
import pandas as pd
import gradio as gr

import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from settings import RERANKER_ID, OUTPUT_DIR, DEVICE, BATCH_SIZE
os.environ['WANDB_DISABLED'] = 'true'

    
fine_tuned_model = SentenceTransformer(OUTPUT_DIR, device=DEVICE)
reranker_model   = CrossEncoder(RERANKER_ID, device=DEVICE)
legal_index      = faiss.read_index('data/retrieval/legal_faiss.index')

def search_and_rerank(emb_model, rerank_model, query, index, faiss_k=50, final_k=10, batch_size=32):
    # FAISS
    q_emb = emb_model.encode(
        query, 
        convert_to_numpy=True, 
        normalize_embeddings=True,
    ).astype(np.float32).reshape(1, -1)
    
    scores, indices = index.search(q_emb, faiss_k) # shape: (1, faiss_k)
    
    cand_idxs   = indices[0]
    cand_scores = scores[0]
    cand_texts  = [passages[i] for i in cand_idxs]
    
    # Reranking
    pairs     = [(query, text) for text in cand_texts]
    re_scores = rerank_model.predict(pairs, batch_size=batch_size, convert_to_numpy=True)
    
    # Sort by rerank score
    merged = [{
        'index'         : int(cand_idxs[i]),
        'bm25_score'    : float(cand_scores[i]),
        'rerank_score'  : float(re_scores[i]),
        'text'          : cand_texts[i]
    } for i in range(len(cand_idxs))]

    merged.sort(key=lambda x: x['rerank_score'], reverse=True)
    return merged[:final_k]

def get_results(query, top_k):
    hits = search_and_rerank(
        emb_model=fine_tuned_model, 
        rerank_model=reranker_model, 
        query=query,
        index=legal_index,
        faiss_k=top_k * 5,
        final_k=top_k
    )
    
    result = ""
    for h in hits:
        result += f"[Rank {rank}] Relevance Score: {h['rerank_score']:.4f}\n\n{h['text']}\n\n{'=' * 80}\n\n"
    return result

    
demo = gr.Interface(
    fn=search_legal_documents,
    inputs=[
        gr.Textbox(lines=2, placeholder='Nhập câu hỏi pháp lý của bạn...', label='Câu hỏi'),
        gr.Slider(minimum=5, maximum=20, value=10, step=1, label='Số lượng kết quả'),
    ],
    outputs=gr.Textbox(lines=20, label='Kết quả'),
    title='Hệ thống Truy vấn Văn bản Pháp luật Tiếng Việt',
    description='Nhập câu hỏi pháp lý của bạn bằng tiếng Việt để nhận các đoạn văn bản pháp luật liên quan.',
    examples=[
        ['Hợp đồng lao động là gì?'],
        ['Quyền lợi của người lao động khi bị sa thải?'],
        ['Thủ tục đăng ký kết hôn như thế nào?'],
        ['Điều kiện thành lập doanh nghiệp tư nhân?'],
    ]
)

if __name__ == '__main__':
    demo.launch()