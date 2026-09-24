"""Render all 64 grouped latent slots with attention Top5 and Query/GT IoU."""
from __future__ import annotations

import math
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from masklat.evaluation.utils.latent_query_atlas_render import (
    _GRAY, _INK, _MUTED, _font, _fitted_size, _probability, _query_tile, _wrap,
)


_GREEN = (20, 138, 72)
_LIGHT_GREEN = (229, 247, 237)


def render_latent_gt_focus_sheet(original, gt_mask, weights, mask_prob, query_ious,
                                 analysis, output_path, metadata=None, *, tile_size=80,
                                 columns=4, display_mode="grayscale"):
    """Write a fixed-L-ID sheet; green tiles are Query/GT IoU >= 0.7."""
    if not isinstance(original, Image.Image) or original.mode != "RGB":
        raise ValueError("original must be a nonempty RGB PIL image")
    attention = _probability(weights, 2, "weights")
    masks = _probability(mask_prob, 3, "mask_prob")
    ious = torch.as_tensor(query_ious, dtype=torch.float64).detach().cpu()
    if attention.shape[0] != 64 or masks.shape[0] != attention.shape[1] or ious.shape != (attention.shape[1],):
        raise ValueError("weights, masks, and Query/GT IoUs must share the Query axis")
    gt = torch.as_tensor(gt_mask).detach().cpu()
    if gt.ndim != 2 or tuple(gt.shape) != (original.height, original.width):
        raise ValueError("GT mask must be in the original image frame")
    if not torch.all((gt == 0) | (gt == 1)):
        raise ValueError("GT mask must be binary")
    if not isinstance(analysis, dict) or analysis.get("num_latents") != 64:
        raise ValueError("analysis must describe all 64 latents")
    latent_rows = analysis.get("latent_rows")
    if not isinstance(latent_rows, list) or len(latent_rows) != 64:
        raise ValueError("analysis latent rows are missing")
    if not isinstance(tile_size, int) or tile_size < 64 or not isinstance(columns, int) or columns < 1:
        raise ValueError("tile_size must be >=64 and columns must be positive")
    if display_mode not in ("grayscale", "overlay"):
        raise ValueError("display_mode must be grayscale or overlay")
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")

    meta = dict(metadata or {})
    top_k = analysis.get("top_k")
    gap, padding, margin = 8, 10, 18
    panel_width = 2 * padding + top_k * tile_size + (top_k - 1) * gap
    panel_height = tile_size + 116
    used_columns = min(columns, 64)
    width = 2 * margin + used_columns * panel_width + (used_columns - 1) * 12
    title_font, text_font, small_font, tiny_font = _font(22), _font(15), _font(12), _font(10)
    scratch = ImageDraw.Draw(Image.new("RGB", (width, 1)))
    title = meta.get("title") or " / ".join(str(meta.get(k, "?")) for k in
                                             ("dataset", "dataset_index", "stage"))
    title += " / GT-focused Top5"
    title_lines = _wrap(scratch, title, title_font, width - 2 * margin)

    thumbnail_width, thumbnail_height = min(230, width // 5), 180
    source_fit = _fitted_size(original.size, (thumbnail_width, thumbnail_height))
    source_y = margin + len(title_lines) * 29 + 8
    source_x = margin
    gt_x = source_x + thumbnail_width + 16
    text_x = gt_x + thumbnail_width + 24
    text_width = width - text_x - margin
    notes = [
        f"GT-good Query: original-resolution mask IoU >= {analysis['primary_gt_threshold']:.1f}; "
        f"this stage has {analysis['primary_good_query_count']} / {analysis['num_queries']}.",
        f"Top5 hit latents: {analysis['latent_top5_hit_count']} / 64; Top5 contains >=2 GT-good Queries: "
        f"{analysis['latent_top5_multi_hit_count']} / 64.",
        "GT mass is the full p(Query|latent) mass over ALL GT-good Queries, not only these five tiles.",
        "Green tile/border means this Query is GT-good. Latent IDs remain fixed L00..L63 across stages.",
        "Association diagnostic only: it does not by itself prove causal contribution.",
    ]
    if meta.get("expression"):
        notes.append(f"Expression: {meta['expression']}")
    note_lines = [line for note in notes for line in _wrap(scratch, note, text_font, text_width)]
    header_height = max(source_y + source_fit[1] + 28, source_y + len(note_lines) * 21) + 20
    rows_count = math.ceil(64 / used_columns)
    footer = _wrap(
        scratch,
        "Read labels as: Query ID, attention weight, Query/GT IoU. Multiple green tiles in one L panel "
        "directly show that this latent's attention Top5 contains multiple Queries matching the same GT.",
        small_font, width - 2 * margin,
    )
    height = header_height + rows_count * panel_height + 24 + len(footer) * 18
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, line in enumerate(title_lines):
        draw.text((margin, margin + index * 29), line, font=title_font, fill=_INK)
    canvas.paste(original.resize(source_fit, Image.Resampling.LANCZOS), (source_x, source_y))
    gt_image = Image.fromarray((gt.to(torch.uint8).numpy() * 255)).convert("RGB").resize(
        source_fit, Image.Resampling.NEAREST
    )
    canvas.paste(gt_image, (gt_x, source_y))
    draw.text((source_x, source_y + source_fit[1] + 4), "Original RGB", font=small_font, fill=_MUTED)
    draw.text((gt_x, source_y + source_fit[1] + 4), "Referring GT", font=small_font, fill=_MUTED)
    for index, line in enumerate(note_lines):
        draw.text((text_x, source_y + index * 21), line, font=text_font, fill=_MUTED)

    tile_cache, manifest_rows = {}, []
    for latent_id, record in enumerate(latent_rows):
        if record.get("latent_id") != latent_id:
            raise ValueError("analysis latent rows must be fixed L00..L63 order")
        column, row = latent_id % used_columns, latent_id // used_columns
        x = margin + column * (panel_width + 12)
        y = header_height + row * panel_height
        highlighted = record["primary_top5_hit"]
        draw.rounded_rectangle(
            (x, y, x + panel_width - 1, y + panel_height - 9), radius=5,
            fill=_LIGHT_GREEN if highlighted else "white",
            outline=_GREEN if highlighted else (210, 216, 224), width=2 if highlighted else 1,
        )
        label = (f"L{latent_id:02d}  |  GT mass {100 * record['primary_gt_attention_mass']:.2f}%  |  "
                 f"Top5 good {record['primary_top5_good_count']}/{top_k}")
        draw.text((x + padding, y + 7), label, font=text_font, fill=_GREEN if highlighted else _INK)
        manifest_row = {"latent_id": latent_id, "gt_mass": record["primary_gt_attention_mass"],
                        "top5_good_count": record["primary_top5_good_count"], "tiles": []}
        for offset, (query, weight, iou) in enumerate(zip(
                record["top_query_ids"], record["top_weights"], record["top_query_gt_ious"])):
            tile_x, tile_y = x + padding + offset * (tile_size + gap), y + 62
            if query not in tile_cache:
                tile_cache[query] = _query_tile(masks[query], original, tile_size, display_mode)
            tile, visible_box = tile_cache[query]
            good = iou >= analysis["primary_gt_threshold"]
            draw.text((tile_x, y + 34), f"Q{query}", font=small_font, fill=_GREEN if good else _INK)
            canvas.paste(tile, (tile_x, tile_y))
            draw.rectangle((tile_x, tile_y, tile_x + tile_size - 1, tile_y + tile_size - 1),
                           outline=_GREEN if good else (180, 186, 194), width=3 if good else 1)
            draw.text((tile_x, tile_y + tile_size + 3), f"w {100 * weight:.2f}%", font=tiny_font, fill=_INK)
            draw.text((tile_x, tile_y + tile_size + 17), f"IoU {100 * iou:.1f}%", font=tiny_font,
                      fill=_GREEN if good else _MUTED)
            manifest_row["tiles"].append({
                "query_id": query, "weight": weight, "gt_iou": iou, "gt_good": good,
                "box": [tile_x, tile_y, tile_x + tile_size, tile_y + tile_size],
                "image_box": [visible_box[0] + tile_x, visible_box[1] + tile_y,
                              visible_box[2] + tile_x, visible_box[3] + tile_y],
            })
        manifest_rows.append(manifest_row)
    footer_y = header_height + rows_count * panel_height + 7
    for index, line in enumerate(footer):
        draw.text((margin, footer_y + index * 18), line, font=small_font, fill=_MUTED)
    with destination.open("xb") as stream:
        canvas.save(stream, format="PNG")
    return {
        "path": str(destination), "width": width, "height": height,
        "latent_count": 64, "query_count": int(attention.shape[1]), "top_k": top_k,
        "primary_gt_threshold": analysis["primary_gt_threshold"], "latents": manifest_rows,
    }
