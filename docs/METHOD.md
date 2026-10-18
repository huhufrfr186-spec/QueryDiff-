# Method

## Object targets and matched queries

The frozen CLIP ViT-B/16 teacher produces a 40 × 40 feature map for a 640 × 640 input. AnyUp upsamples the features to 80 × 80. For each ground-truth box, `object_targets` pools a 3 × 3 bilinear quadrature and normalizes the resulting feature to obtain the target $t_k$.

Hungarian matching associates ground-truth objects with ordinary queries from the final decoder layer. A learned linear projection maps each matched query $h_{\sigma(k)}$ to $z_k = P(h_{\sigma(k)})$. Denoising queries are excluded. Teacher targets are detached from the training graph.

## Query anchor

$$
L_{\mathrm{anchor}} = \frac{1}{N}\sum_k \lVert z_k-t_k\rVert_2^2.
$$

## Local graph

`graph_edges` connects unordered object pairs whose centers occupy the same cell in either the 80 × 80 student grid or the 40 × 40 teacher grid. For aligned grids, the finer cells are nested within the coarse cells, so the resulting edge set equals the set of pairs sharing a teacher cell.

## Weighted query differences

Teacher views use brightness scales of 1.0, 0.95, and 1.05 while preserving geometry. For each local edge $(k,l)$:

$$
d_{kl}=t_k-t_l,\qquad e_{kl}=\lVert d_{kl}\rVert_2^2,
$$

$$
v_{kl}=\frac{1}{R}\sum_r\lVert(t_k^r-t_l^r)-d_{kl}\rVert_2^2,
$$

$$
w_{kl}=\frac{e_{kl}}{(e_{kl}+v_{kl}+\epsilon)^2},\qquad \epsilon=10^{-4},
$$

$$
L_{\mathrm{diff}}=\frac{1}{|E|}\sum_{(k,l)\in E}w_{kl}\lVert(z_k-z_l)-d_{kl}\rVert_2^2.
$$

The weights are detached. Mode `A2` uses uniform weights on the same edges. Empty object and edge sets return differentiable zero losses.

## Training objective

$$
L=L_{\mathrm{det}}+\lambda_{\mathrm{deep}}L_{\mathrm{DSI}}+\lambda_{\mathrm{anchor}}L_{\mathrm{anchor}}+\lambda_{\mathrm{diff}}L_{\mathrm{diff}}.
$$

Deep distillation uses the AIFI output before the feature pyramid, a 1 × 1 projection to 768 channels, and coarse CLIP features. GAM adjusts the deep-distillation coefficient using a gradient-share odds ratio with target 11%, exponent 1, and an update factor bounded between 0.1 and 10.

Query coefficients are calibrated using the first four training images containing local edges. The mean initial anchor and weighted-difference losses are each scaled to 0.1 times the mean initial detection loss, with command-line multipliers applied. Random generator states are restored after calibration.

## Implementation

| Component | Location |
|---|---|
| Object targets, local graph, anchor and difference losses, GAM | `projects/rtdetrv4_uav/query_difference.py` |
| Frozen teacher, matched-query extraction, calibration and training | `projects/rtdetrv4_uav/run_query_difference.py` |
| Detector construction | `projects/rtdetrv4_uav/train_visdrone.py` |
| Detector inference | `tools/infer_querydiff.py` |

The detector uses ten VisDrone classes and a shared random initialization for classification heads under a fixed seed. Teacher features are cached by input pixels, boxes, source hashes, and teacher weight hashes. Inference uses the trained detector weights and excludes teacher computation and auxiliary projections.
