"""
model_def.py
============
HCTF (Hierarchical Cross-Transformer Fusion) model architecture.

Two complementary model variants are defined:
  - build_dual_mgaf_transformer : training-stage model, outputs class probabilities only.
                                   Save with save_weights_only=True for clean deployment.
  - build_explainable_hctf      : deployment-stage XAI engine; shares the exact same
                                   weight topology but additionally exposes cross-modal
                                   attention tensors and MGAF spectral scores for
                                   interpretability and heatmap visualization.

Architecture pipeline (manuscript Section 2.6):
  Input A : PATCH features      (N, 60, 10)      -- PCA-guided spatial-spectral patches
  Input B : MGAF matrices       (N, 60, 10, 10)  -- multi-dimensional Gramian Angular Field
  ├── MGAF diagonal → spectral importance gate (Eq. 23)
  ├── MGAF column-mean → MGAF token embeddings
  ├── PATCH linear projection → PATCH token embeddings
  ├── Cross-Modal Transformer (CMT): PATCH queries MGAF keys/values  (Eq. 20-22)
  ├── MGAF-Guided Self-Attention (MGT): gate weighted self-attention  (Eq. 24-27)
  └── Global average pooling → dense classifier → 3-class softmax

Reference
---------
Guo, Z. et al. (2025). "Bridging deep learning and biochemical mechanisms: An explainable
hierarchical cross-transformer expert system for postharvest fungal detection in nuts."
"""

import tensorflow as tf
from keras import layers, models, Input


# ---------------------------------------------------------------------------
# Transformer block definitions
# ---------------------------------------------------------------------------

class CrossModalAttentionBlock(layers.Layer):
    """
    Cross-Modal Transformer (CMT) block.

    PATCH spatial-spectral token embeddings act as queries; MGAF structural
    token embeddings act as keys and values.  This asymmetric cross-attention
    forces the spatial representation to actively retrieve spectral correlation
    information, ensuring structurally coherent integration (manuscript Eq. 20-22).

    Parameters
    ----------
    embed_dim    : model embedding dimension
    num_heads    : number of parallel attention heads
    ff_dim       : inner dimension of the position-wise feed-forward network
    dropout_rate : dropout probability applied after attention and FFN
    name         : layer name prefix (must be stable for weight alignment)
    """

    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1,
                 name="cmt_block", **kwargs):
        super().__init__(name=name, **kwargs)
        self.cross_att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, name=f"{name}_mha"
        )
        self.ffn = models.Sequential([
            layers.Dense(ff_dim, activation="relu", name=f"{name}_ffn_1"),
            layers.Dense(embed_dim, name=f"{name}_ffn_2"),
        ], name=f"{name}_ffn_seq")
        self.norm1 = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_norm1")
        self.norm2 = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_norm2")
        self.dropout1 = layers.Dropout(dropout_rate, name=f"{name}_drop1")
        self.dropout2 = layers.Dropout(dropout_rate, name=f"{name}_drop2")

    def call(self, query, key_value, training=False, return_attention_scores=False):
        if return_attention_scores:
            attn_output, attn_scores = self.cross_att(
                query=query, key=key_value, value=key_value,
                return_attention_scores=True, training=training,
            )
        else:
            attn_output = self.cross_att(
                query=query, key=key_value, value=key_value, training=training
            )
            attn_scores = None

        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(query + attn_output)         # residual + norm
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.norm2(out1 + ffn_output)           # residual + norm

        if return_attention_scores:
            return out2, attn_scores
        return out2


class MGAFGuidedSelfAttentionBlock(layers.Layer):
    """
    MGAF-Guided Self-Attention (MGT) block.

    MGAF diagonal spectral scores (mean of each patch's MGAF matrix diagonal)
    gate the embedded feature sequence before multi-head self-attention.
    Patches exhibiting strong inter-band spectral correlations (high MGAF
    diagonal values) receive amplified attention weights, making the learned
    focus explicitly traceable to spectral structure (manuscript Eq. 24-27).

    Parameters
    ----------
    embed_dim    : model embedding dimension
    num_heads    : number of parallel attention heads
    ff_dim       : inner dimension of the position-wise feed-forward network
    dropout_rate : dropout probability
    name         : layer name prefix (must be stable for weight alignment)
    """

    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1,
                 name="mgt_block", **kwargs):
        super().__init__(name=name, **kwargs)
        self.self_att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, name=f"{name}_mha"
        )
        self.ffn = models.Sequential([
            layers.Dense(ff_dim, activation="relu", name=f"{name}_ffn_1"),
            layers.Dense(embed_dim, name=f"{name}_ffn_2"),
        ], name=f"{name}_ffn_seq")
        self.norm1 = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_norm1")
        self.norm2 = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_norm2")
        self.dropout1 = layers.Dropout(dropout_rate, name=f"{name}_drop1")
        self.dropout2 = layers.Dropout(dropout_rate, name=f"{name}_drop2")

    def call(self, x, mgaf_score_vector, training=False,
             return_attention_scores=False):
        # Element-wise gating: amplify patches with stronger spectral correlation
        x_weighted = x * tf.expand_dims(mgaf_score_vector, -1)

        if return_attention_scores:
            attn_output, attn_scores = self.self_att(
                query=x_weighted, value=x_weighted, key=x_weighted,
                return_attention_scores=True, training=training,
            )
        else:
            attn_output = self.self_att(
                query=x_weighted, value=x_weighted, key=x_weighted,
                training=training,
            )
            attn_scores = None

        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(x + attn_output)             # residual uses un-gated x
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.norm2(out1 + ffn_output)

        if return_attention_scores:
            return out2, attn_scores
        return out2


# ---------------------------------------------------------------------------
# Shared computational graph
# ---------------------------------------------------------------------------

def _build_core_graph(patch_input, mgaf_input, embed_dim, num_heads, ff_dim,
                      dropout_rate, num_classes, return_attention=False):
    """
    Shared weight topology for both training and deployment models.

    The layer name strings (e.g. 'cross_modal_transformer', 'mgaf_projection')
    must remain identical between build_dual_mgaf_transformer and
    build_explainable_hctf to guarantee lossless weight transfer via
    load_weights().  Do not rename these layers.
    """
    # --- MGAF spectral importance scores (Eq. 23) ---
    mgaf_diag = tf.linalg.diag_part(mgaf_input)                           # (N, 60, 10)
    mgaf_scores = tf.reduce_mean(mgaf_diag, axis=-1,
                                 name="mgaf_scores_extraction")            # (N, 60)

    # --- Dual-stream linear projections ---
    mgaf_vector = tf.reduce_mean(mgaf_input, axis=-1,
                                 name="mgaf_vector_pooling")               # (N, 60, 10)
    mgaf_embed = layers.Dense(embed_dim, name="mgaf_projection")(mgaf_vector)
    patch_embed = layers.Dense(embed_dim, name="patch_projection")(patch_input)

    # --- Hierarchical cross-transformer fusion ---
    cross_block = CrossModalAttentionBlock(
        embed_dim, num_heads, ff_dim, dropout_rate,
        name="cross_modal_transformer",
    )
    self_block = MGAFGuidedSelfAttentionBlock(
        embed_dim, num_heads, ff_dim, dropout_rate,
        name="mgaf_guided_transformer",
    )

    if return_attention:
        x, cross_attn = cross_block(patch_embed, mgaf_embed,
                                    return_attention_scores=True)
        x, self_attn = self_block(x, mgaf_scores,
                                   return_attention_scores=True)
    else:
        x = cross_block(patch_embed, mgaf_embed)
        x = self_block(x, mgaf_scores)

    # --- Classifier head ---
    pooled = layers.GlobalAveragePooling1D(name="global_pooling")(x)
    h = layers.Dense(64, activation="relu", name="classifier_dense_1")(pooled)
    h = layers.Dropout(dropout_rate, name="classifier_dropout")(h)
    cls_output = layers.Dense(num_classes, activation="softmax",
                               name="classifier_output")(h)

    if return_attention:
        return cls_output, cross_attn, self_attn, mgaf_scores
    return cls_output


# ---------------------------------------------------------------------------
# Public model constructors
# ---------------------------------------------------------------------------

def build_dual_mgaf_transformer(
    seq_len=60, patch_dim=10, mgaf_dim=10,
    embed_dim=64, num_heads=4, ff_dim=128,
    num_classes=3, dropout_rate=0.3,
):
    """
    Training-stage HCTF model.

    Outputs a single softmax probability tensor of shape (N, num_classes).
    Compile with Adam + categorical cross-entropy and use
    ModelCheckpoint(save_weights_only=True) to persist the best checkpoint.
    The saved .weights.h5 file loads directly into build_explainable_hctf.

    Parameters
    ----------
    seq_len      : patches per kernel (= N_PATCHES = 60)
    patch_dim    : spectral bands per patch after PCA selection (= 10)
    mgaf_dim     : MGAF matrix side length (= patch_dim = 10)
    embed_dim    : transformer embedding width
    num_heads    : multi-head attention heads
    ff_dim       : feed-forward network inner width
    num_classes  : output classes (3: Control / AC / NC)
    dropout_rate : dropout probability
    """
    patch_input = Input(shape=(seq_len, patch_dim), name="patch_input")
    mgaf_input = Input(shape=(seq_len, mgaf_dim, mgaf_dim), name="mgaf_input")
    output = _build_core_graph(
        patch_input, mgaf_input,
        embed_dim, num_heads, ff_dim, dropout_rate, num_classes,
        return_attention=False,
    )
    return models.Model(
        inputs=[patch_input, mgaf_input], outputs=output, name="HCTF_Train"
    )


def build_explainable_hctf(
    seq_len=60, patch_dim=10, mgaf_dim=10,
    embed_dim=64, num_heads=4, ff_dim=128,
    num_classes=3, dropout_rate=0.3,
):
    """
    Deployment-stage explainable HCTF model (XAI engine).

    Shares the exact weight topology with build_dual_mgaf_transformer.
    Returns a dict of four tensors enabling full per-sample interpretability:

      cls_output        (N, num_classes)           -- softmax probabilities
      cross_attention   (N, num_heads, seq, seq)   -- CMT attention weights
      self_attention    (N, num_heads, seq, seq)   -- MGT attention weights
      mgaf_token_scores (N, seq_len)               -- per-patch spectral gate

    The mgaf_token_scores vector, reshaped to (10, 6) per sample, maps directly
    onto the peanut kernel spatial patch grid (10 rows x 6 cols), generating
    the biomarker activation heatmap displayed in the SpectraX-Inspect dashboard
    and reported as Fig. 10(c) in the manuscript.
    """
    patch_input = Input(shape=(seq_len, patch_dim), name="patch_input")
    mgaf_input = Input(shape=(seq_len, mgaf_dim, mgaf_dim), name="mgaf_input")
    cls_out, c_attn, s_attn, mgaf_scores = _build_core_graph(
        patch_input, mgaf_input,
        embed_dim, num_heads, ff_dim, dropout_rate, num_classes,
        return_attention=True,
    )
    return models.Model(
        inputs=[patch_input, mgaf_input],
        outputs={
            "cls_output": cls_out,
            "cross_attention": c_attn,
            "self_attention": s_attn,
            "mgaf_token_scores": mgaf_scores,
        },
        name="HCTF_Explainable",
    )