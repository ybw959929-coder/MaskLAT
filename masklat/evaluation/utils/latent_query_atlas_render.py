"""Render real Query masks selected by every latent into a fixed-ID PNG atlas.

This module is deliberately independent of masklat/model imports.  Callers must
remove SAM padding and undo image transforms *before* passing ``mask_prob``:
its whole canvas must correspond to the whole, unmodified ``original`` image.
No mask is normalized by its own min/max, thresholded, or merged with others.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont


_GRAY = (239, 239, 239)
_INK = (26, 31, 38)
_MUTED = (78, 88, 100)
_ACCENT = (205, 40, 57)


def _font(size):
    for candidate in ("DejaVuSans.ttf", "Arial.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Older Pillow has only the fixed-size default font.
        return ImageFont.load_default()


def _wrap(draw, text, font, width):
    """Pixel-based wrapping, including paths/identifiers without spaces."""
    lines = []
    for paragraph in str(text).splitlines() or [""]:
        current = ""
        for word in paragraph.split():
            proposed = f"{current} {word}" if current else word
            if draw.textlength(proposed, font=font) <= width:
                current = proposed
                continue
            if current:
                lines.append(current)
                current = ""
            for char in word:
                if current and draw.textlength(current + char, font=font) > width:
                    lines.append(current)
                    current = ""
                current += char
        lines.append(current)
    return lines


def _probability(value, ndim, name):
    if not isinstance(value, torch.Tensor) or value.ndim != ndim or not value.numel():
        raise ValueError(f"{name} must be a nonempty {ndim}-dimensional torch tensor")
    result = value.detach().to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(result).all() or ((result < 0) | (result > 1)).any():
        raise ValueError(f"{name} must contain finite probabilities in [0, 1]")
    return result


def _fitted_size(original_size, bounds):
    scale = min(bounds[0] / original_size[0], bounds[1] / original_size[1])
    return tuple(max(1, int(round(side * scale))) for side in original_size)


def _query_tile(mask, original, size, mode):
    """Fixed [0,1] probability display; gray letterboxing is not image content."""
    fit = _fitted_size(original.size, (size, size))
    values = (mask * 255).round().to(torch.uint8).numpy()
    probability_image = Image.fromarray(values).resize(fit, Image.Resampling.BILINEAR)
    if mode == "grayscale":
        visible = probability_image.convert("RGB")
    else:
        rgb = original.resize(fit, Image.Resampling.LANCZOS)
        # Globally fixed alpha = 0.65*p. Never scale a mask to its own maximum.
        alpha = probability_image.point(lambda p: round(p * 0.65))
        visible = Image.composite(Image.new("RGB", fit, _ACCENT), rgb, alpha)
    tile = Image.new("RGB", (size, size), _GRAY)
    offset = ((size - fit[0]) // 2, (size - fit[1]) // 2)
    tile.paste(visible, offset)
    return tile, (*offset, offset[0] + fit[0], offset[1] + fit[1])


def render_stage_sheet(original, weights, mask_prob, output_path, metadata=None, *,
                       tile_size=72, columns=4, weight_kind="latent_reads_query",
                       display_mode="grayscale", top_k=5):
    """Write one atlas without overwriting, returning a JSON-serializable manifest.

    ``weights`` is [L,Q]; ``mask_prob`` is [Q,H,W] in the ORIGINAL image frame.
    L can vary for test/ablation use; normal runs should pass all 64 slots.
    Latent panels are always row-major slot order L00, L01, ...; only candidate
    Queries inside each panel are sorted.  Ties use ascending Query ID.

    ``latent_reads_query``: rows sum to one, percentages are p(Query|latent).
    ``query_reads_latent``: columns sum to one, percentages are p(latent|Query).
    The latter must NOT be relabeled/renormalized as a latent read distribution:
    it selects Queries sending the largest attention share to this latent.

    Metadata fields title/dataset/dataset_index/stage/expression/semantics/notes are shown
    when supplied; the caller should state which attention/mask timepoint was
    captured.  This renderer makes no claims about latent feature similarity.
    Stored BF16 softmax probabilities may sum only approximately to one; the
    default absolute sum tolerance is 0.005 (metadata can override via
    probability_sum_tolerance). The actual weights are never renormalized.
    """
    if not isinstance(original, Image.Image) or min(original.size) < 1:
        raise ValueError("original must be a nonempty PIL image")
    if original.mode != "RGB":
        raise ValueError("original must be RGB; convert explicitly before rendering")
    if not isinstance(tile_size, int) or tile_size < 48:
        raise ValueError("tile_size must be an integer >= 48")
    if not isinstance(columns, int) or columns < 1:
        raise ValueError("columns must be a positive integer")
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if display_mode not in ("grayscale", "overlay"):
        raise ValueError("display_mode must be grayscale or overlay")
    if weight_kind not in ("latent_reads_query", "query_reads_latent"):
        raise ValueError("unknown weight_kind")
    attention = _probability(weights, 2, "weights")
    masks = _probability(mask_prob, 3, "mask_prob")
    latent_count, query_count = attention.shape
    if masks.shape[0] != query_count:
        raise ValueError("weights and mask_prob must use the same Query ordering/count")
    if top_k > query_count:
        raise ValueError("top_k cannot exceed Query count")
    meta = dict(metadata or {})
    tolerance = float(meta.get("probability_sum_tolerance", 0.005))
    if not math.isfinite(tolerance) or not 0 <= tolerance <= 0.02:
        raise ValueError("probability_sum_tolerance must be finite and in [0, 0.02]")
    sums = attention.sum(1 if weight_kind == "latent_reads_query" else 0)
    sum_error = float((sums - 1).abs().max())
    if not torch.allclose(sums, torch.ones_like(sums), atol=tolerance, rtol=0):
        axis = "rows" if weight_kind == "latent_reads_query" else "columns"
        raise ValueError(f"weights {axis} must approximately sum to one for {weight_kind}; "
                         f"max error {sum_error:.6g} exceeds tolerance {tolerance:.6g}")
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    # Stable sorting matters when all slots/Queries have near-uniform weights.
    ids = torch.argsort(attention, dim=1, descending=True, stable=True)[:, :top_k]
    selected_weights = attention.gather(1, ids)

    gap, panel_padding, margin = 8, 10, 18
    panel_width = 2 * panel_padding + top_k * tile_size + (top_k - 1) * gap
    panel_height = tile_size + 89
    used_columns = min(columns, latent_count)
    width = 2 * margin + used_columns * panel_width + (used_columns - 1) * 12
    title_font, text_font, small_font, tiny_font = _font(22), _font(15), _font(12 if tile_size >= 64 else 10), _font(10)
    header_draw = ImageDraw.Draw(Image.new("RGB", (width, 1)))
    title = meta.get("title") or " / ".join(str(meta.get(k, "?")) for k in
                                             ("dataset", "dataset_index", "stage"))
    title_lines = _wrap(header_draw, title, title_font, width - 2 * margin)
    source_column_width = min(256, width // 3)
    source_fit = _fitted_size(original.size, (source_column_width, 192))
    source_x, source_y = margin, margin + len(title_lines) * 29 + 8
    text_x = margin + source_column_width + 20
    text_width = width - text_x - margin
    if text_width < 100:
        # Narrow, one-panel test/ablation sheets place notes below the original.
        text_x, text_width = margin, width - 2 * margin
        text_y = source_y + source_fit[1] + 29
    else:
        text_y = source_y
    direction = ("Weights: p(Query | latent). Each latent independently reads Queries."
                 if weight_kind == "latent_reads_query" else
                 "Weights: p(latent | Query). Top Queries by attention share sent to this latent; NOT p(Query | latent).")
    descriptions = [
        f"{latent_count} latent slots; fixed row-major IDs; top {top_k} Queries per slot.",
        direction,
        "Each tile is ONE candidate Query mask, not a merged latent mask.",
        ("Fixed grayscale: black=0, white=1. Gray margins are outside the image."
         if display_mode == "grayscale" else
         "Fixed overlay: red alpha=0.65 x mask probability. Gray margins are outside the image."),
        "All mask tiles use the same original-image frame and aspect ratio.",
        "empty@.5 means no stored mask pixel exceeds 0.5; not an empty Query feature.",
    ]
    if meta.get("expression"):
        descriptions.append(f"Expression: {meta['expression']}")
    if sum_error > 3e-4:
        descriptions.append("Stored low-precision probabilities are shown without renormalization; sums are approximate.")
    for key in ("semantics", "notes"):
        if meta.get(key):
            descriptions.append(str(meta[key]))
    note_lines = []
    for description in descriptions:
        note_lines.extend(_wrap(header_draw, description, text_font, text_width))
    header_height = max(source_y + source_fit[1] + 29, text_y + len(note_lines) * 21) + 20
    panel_rows = math.ceil(latent_count / used_columns)
    footer_lines = _wrap(header_draw,
                        "Attention similarity is not latent-feature similarity. Compare the SAME latent ID across stages; "
                        "Query IDs are model slots, not fixed object identities across stages.",
                        small_font, width - 2 * margin)
    height = header_height + panel_rows * panel_height + 24 + len(footer_lines) * 18
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, line in enumerate(title_lines):
        draw.text((margin, margin + index * 29), line, font=title_font, fill=_INK)
    canvas.paste(original.resize(source_fit, Image.Resampling.LANCZOS), (source_x, source_y))
    draw.text((source_x, source_y + source_fit[1] + 5), "Original RGB (thumbnail)", font=small_font, fill=_MUTED)
    for index, line in enumerate(note_lines):
        draw.text((text_x, text_y + index * 21), line, font=text_font, fill=_MUTED)

    tile_cache, rows = {}, []
    for latent_id in range(latent_count):
        column, row = latent_id % used_columns, latent_id // used_columns
        x = margin + column * (panel_width + 12)
        y = header_height + row * panel_height
        draw.rounded_rectangle((x, y, x + panel_width - 1, y + panel_height - 9),
                               radius=5, outline=(210, 216, 224), width=1)
        label = f"L{latent_id:02d}"
        if weight_kind == "latent_reads_query":
            label += f"  |  Top{top_k} mass {100 * float(selected_weights[latent_id].sum()):.2f}%"
        draw.text((x + panel_padding, y + 7), label, font=text_font, fill=_INK)
        record = {"latent_id": latent_id, "query_ids": ids[latent_id].tolist(),
                  "weights": selected_weights[latent_id].tolist(), "tiles": []}
        if weight_kind == "latent_reads_query":
            record["top_mass"] = float(selected_weights[latent_id].sum())
        for k, query in enumerate(record["query_ids"]):
            tile_x, tile_y = x + panel_padding + k * (tile_size + gap), y + 51
            if query not in tile_cache:
                tile_cache[query] = _query_tile(masks[query], original, tile_size, display_mode)
            tile, visible_box = tile_cache[query]
            draw.text((tile_x, y + 29), f"Q{query}", font=small_font, fill=_INK)
            canvas.paste(tile, (tile_x, tile_y))
            draw.text((tile_x, tile_y + tile_size + 3),
                      f"{100 * record['weights'][k]:.2f}%", font=small_font, fill=_INK)
            is_empty = not bool((masks[query] > 0.5).any())
            if is_empty:
                draw.text((tile_x, tile_y + tile_size + 18), "empty@.5", font=tiny_font, fill=_MUTED)
            record["tiles"].append({"box": [tile_x, tile_y, tile_x + tile_size, tile_y + tile_size],
                                    "image_box": [visible_box[0] + tile_x, visible_box[1] + tile_y,
                                                  visible_box[2] + tile_x, visible_box[3] + tile_y],
                                    "empty_at_threshold_0p5": is_empty})
        rows.append(record)
    footer_y = header_height + panel_rows * panel_height + 7
    for index, line in enumerate(footer_lines):
        draw.text((margin, footer_y + index * 18), line, font=small_font, fill=_MUTED)
    # Atomic refusal to overwrite, including a destination created during render.
    with destination.open("xb") as stream:
        canvas.save(stream, format="PNG")
    return {"path": str(destination), "width": width, "height": height,
            "latent_count": latent_count, "query_count": query_count,
            "columns": used_columns, "panel_rows": panel_rows, "top_k": top_k,
            "weight_kind": weight_kind, "display_mode": display_mode,
            "probability_sum_max_abs_error": sum_error, "probability_sum_tolerance": tolerance,
            "original_size": list(original.size),
            "source_thumbnail_box": [source_x, source_y, source_x + source_fit[0], source_y + source_fit[1]],
            "header_lines": title_lines + note_lines, "latents": rows}
