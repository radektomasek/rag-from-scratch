from sentence_transformers import CrossEncoder

def cross_encoder_rerank(pairs: list):
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    scores = cross_encoder.predict(pairs)
    return scores

