# Method

## Object references and queries

Inputs preserve aspect ratio and are centered on a 640×640 canvas. CLIP ViT-B/16 applies bicubic interpolation to patch positional embeddings while retaining the CLS position. Final transformer patch outputs receive `ln_post`; CLS and `visual.proj` are excluded. AnyUp uses RGB guidance normalized with ImageNet statistics and outputs 160×160 features. Object references are the normalized means of bilinear samples at 4×4 bin centers.

The final ordinary queries are matched to all valid annotations using rectangular Hungarian assignment. Denoising queries are excluded. At most 300 objects enter the object loss; evaluation denominators retain every valid annotation. The independent projections are `Linear(256,768,bias=False)` and `Conv2d(256,768,1,bias=False)`, initialized with Xavier uniform weights.

## Graph and losses

A pair belongs to the graph exactly when its object centers occupy the same 16-pixel teacher patch. The anchor averages squared Euclidean residuals over matched objects, retaining the sum over 768 feature dimensions. Each image contributes equally, including differentiable zeros for empty objects or edges.

For an edge, define `d=t_k-t_l`, `s=||d||²`, `a=s+v+1e-6`, `gamma=s/a`, and `omega=gamma/a`. Variation `v` is the average squared difference between the clean edge target and two independently perturbed teacher-view targets. Both brightness and contrast factors are sampled independently from `[0.9,1.1]`, without changing geometry. Teacher features and all edge coefficients are detached.

The difference term is the edge mean of `omega * ||(z_k-z_l)-d||²`. Uniform difference supervision replaces omega by one. Weighted-degree anchoring replaces the difference term with `sum_k degree_k*||z_k-t_k||² / max(1,|E|)`. This preserves the diagonal quadratic coefficients on the same graph and removes the cross-object terms.

Deep supervision uses mean cosine error on the 20×20 AIFI output, after its independent projection. Teacher resizing uses bilinear interpolation with `align_corners=False`. Cosine calculations use FP32, norm floor `1e-6`, and clipping to `[-1,1]`.

## Optimization

The object coefficients remain fixed at 1 and 0.1. GAM starts at 20, uses the gradient L1 share of all AIFI attention, FFN, and normalization parameters, and includes the independent projections in its total trainable-parameter denominator. Unscaled gradients are averaged across workers before measurement and global norm clipping. Above share 0.12 the next coefficient is multiplied by `0.10/share`; below 0.10 it is multiplied by `0.12/max(share,1e-12)`. The coefficient is retained inside the interval and when there are no valid steps; its bounds are `[1e-4,1e4]`.

The fixed-strength control accumulates base and auxiliary gradients over the effective batch, averages them across workers, and maintains EMA L2 norms with decay 0.95 on deployed detector parameters. The coefficient is the requested gradient target times the ratio of EMA norms, clipped to `[1e-4,1e3]`. Empty or negligible auxiliary gradients receive coefficient zero. Nonfinite losses or gradients skip both the optimization update and monitoring statistics.

LMN computes the two log-loss gradients on AIFI parameters, averages them across workers, solves the two-vector minimum-norm mixture with alpha constrained to `[0.05,0.95]`, and forms detached loss coefficients whose sum is two. The object terms remain additive and are excluded from the mixture solution.

## Inference

Only the detector is evaluated. Each ordinary query contributes its highest-scoring class and one continuous-coordinate box. Predictions are clipped to the original image, stably sorted by score and query ID, and limited to 300 per image.
