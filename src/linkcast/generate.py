"""Deterministic fictional advertising data and schema-v1 link rows.

The simulator works in two layers, like a real exporter would:

1. Source tables (stores, media, campaigns, posts, observations, links,
   conversions) with explicit occurrence/measurement/receipt timestamps.
2. One schema-v1 row per link. Features use only source records available at
   link creation; labels use only matured, reviewed, completely tracked windows.

Hidden simulator variables (true quality, appeal, expected counts) are kept in
separate ``truth`` tables and never enter link rows. All times are handled as
integer seconds since ``GeneratorConfig.start`` so window boundaries are exact.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from linkcast.schema import HORIZON, SCHEMA_VERSION, LinkRow

GENERATOR_VERSION = "1"
HOUR = 3600
DAY = 24 * HOUR
WINDOW = int(HORIZON.total_seconds())

PLATFORMS = ("telegram", "instagram")
CATEGORIES = ("news", "lifestyle", "tech", "food", "fashion", "sports")
OFFER_TYPES = ("discount", "free_shipping", "bundle")
OFFER_EFFECTS = np.array([0.3, 0.0, -0.15])  # log multipliers per offer type
FORMATS = ("image", "video", "sequence")
FORMAT_EFFECTS = np.array([0.0, 0.1, 0.05])
# Relative link-creation intensity per UTC hour, busiest in the afternoon.
CREATION_HOUR_WEIGHTS = np.array(
    [1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 6, 6, 6, 7, 8, 8, 8, 7, 6, 5, 4, 3, 2, 1],
    dtype=float,
)
# Conversion delay after creation: mostly within days, plus a late uniform tail.
FAST_SHARE = 0.9
FAST_MEAN_HOURS = 36
TAIL_HOURS = 336
WINDOW_SHARE = FAST_SHARE * (1 - np.exp(-168 / FAST_MEAN_HOURS)) + (
    1 - FAST_SHARE
) * min(1.0, 168 / TAIL_HOURS)
OUTAGE_RECORDED_SHARE = 0.4


@dataclass(frozen=True)
class GeneratorConfig:
    seed: int = 7
    start: dt.datetime = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    days: int = 182
    n_stores: int = 20
    n_publishers: int = 80
    n_media: int = 150
    links_per_media_week: float = 5.0
    mean_conversions_7d: float = 1.5
    dispersion: float = 1.5
    approval_rate: float = 0.9
    reporting_delay_hours: int = 48
    outage_rate: float = 0.04
    unverified_rate: float = 0.02

    def __post_init__(self) -> None:
        if self.days < 70:
            raise ValueError("days must be at least 70")
        if min(self.n_stores, self.n_publishers, self.n_media) < 1:
            raise ValueError("entity counts must be positive")
        if self.start.utcoffset() is None:
            raise ValueError("start must be timezone-aware")
        if not 0 <= self.outage_rate + self.unverified_rate <= 1:
            raise ValueError("outage_rate + unverified_rate must be within [0, 1]")


@dataclass
class SyntheticData:
    config: GeneratorConfig
    rows: list[LinkRow]
    sources: dict[str, pd.DataFrame]
    truth: dict[str, pd.DataFrame]


def generate(config: GeneratorConfig | None = None) -> SyntheticData:
    """Generate the same fictional sources and link rows for the same config."""
    config = config or GeneratorConfig()
    rng = np.random.default_rng(config.seed)
    end = config.days * DAY  # snapshot time

    publishers = pd.DataFrame({"publisher_id": _ids("publisher", config.n_publishers)})
    stores = _stores(rng, config, end)
    media = _media(rng, config, publishers, end)
    campaigns = _campaigns(rng, stores, end)
    creatives = _creatives(rng, campaigns)
    landings = _landings(rng, stores)
    links = _links(rng, config, media, campaigns, creatives, landings, end)
    posts, post_links = _posts(rng, media, links, end)
    views = _view_observations(rng, media, posts, end)
    audience = _audience_observations(rng, config, media, end)
    conversions, reviews = _conversions(
        rng, config, media, stores, campaigns, creatives, landings, links, end
    )
    rows = _link_rows(
        config, media, campaigns, creatives, links, posts, views, audience,
        conversions, reviews, end,
    )  # fmt: skip

    tables = {
        "publishers": publishers,
        "stores": stores,
        "media": media,
        "campaigns": campaigns,
        "creatives": creatives,
        "landings": landings,
        "links": links,
        "posts": posts,
        "post_links": post_links,
        "view_observations": views,
        "audience_observations": audience,
        "conversions": conversions,
        "tracking_reviews": reviews,
    }
    sources = {name: _public(frame, config.start) for name, frame in tables.items()}
    truth = {
        name: _hidden(frame, config.start)
        for name, frame in tables.items()
        if any(column.startswith("hidden_") for column in frame.columns)
    }
    return SyntheticData(config, rows, sources, truth)


def write(data: SyntheticData, out: str | Path) -> Path:
    """Write link rows (JSONL), source and truth tables (CSV), and a manifest."""
    out = Path(out)
    for folder, tables in (("sources", data.sources), ("truth", data.truth)):
        (out / folder).mkdir(parents=True, exist_ok=True)
        for name, frame in tables.items():
            frame.to_csv(out / folder / f"{name}.csv", index=False)
    with (out / "link_rows.jsonl").open("w") as stream:
        for row in data.rows:
            stream.write(json.dumps(row.to_dict(), separators=(",", ":")) + "\n")
    config = asdict(data.config) | {"start": data.config.start.isoformat()}
    manifest = {
        "generator_version": GENERATOR_VERSION,
        "schema_version": SCHEMA_VERSION,
        "config": config,
        "links": len(data.rows),
        "label_status": {
            status: sum(row.label_status == status for row in data.rows)
            for status in ("ready", "pending", "unknown")
        },
        "source_rows": {name: len(frame) for name, frame in data.sources.items()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out


# --- Source entities -------------------------------------------------------


def _ids(prefix: str, n: int) -> list[str]:
    width = max(3, len(str(n)))
    return [f"{prefix}_{i:0{width}d}" for i in range(1, n + 1)]


def _stores(rng, config, end) -> pd.DataFrame:
    n = config.n_stores
    joins_late = rng.random(n) < 0.3  # new stores appear in later periods
    joined = np.where(joins_late, rng.integers(end // 3, end - 21 * DAY, n), 0)
    return pd.DataFrame(
        {
            "store_id": _ids("store", n),
            "joined_at": joined,
            "hidden_category": rng.choice(CATEGORIES, n),
            "hidden_appeal": rng.normal(0, 0.4, n),
        }
    )


def _media(rng, config, publishers, end) -> pd.DataFrame:
    n = config.n_media
    is_new = rng.random(n) < 0.2
    created = np.where(
        is_new, rng.integers(0, end - 30 * DAY, n), -rng.integers(30, 3 * 365, n) * DAY
    )
    late_start = np.where(rng.random(n) < 0.2, rng.integers(0, end // 2, n), 0)
    first_active = np.minimum(np.maximum(created, 0) + late_start, end - 14 * DAY)
    platform = rng.choice(PLATFORMS, n, p=[0.6, 0.4])
    coverage = np.where(platform == "telegram", rng.beta(4, 2, n), rng.beta(2, 3, n))
    return pd.DataFrame(
        {
            "media_id": _ids("media", n),
            "publisher_id": rng.choice(publishers.publisher_id.to_numpy(), n),
            "platform": platform,
            "category": rng.choice(CATEGORIES, n),
            "created_at": created,
            "hidden_first_active_at": first_active,
            "hidden_audience_base": rng.lognormal(np.log(20_000), 1.0, n),
            "hidden_audience_growth_per_day": rng.normal(0.0015, 0.001, n),
            "hidden_engagement": rng.lognormal(np.log(0.08), 0.4, n),
            "hidden_quality": rng.normal(0, 0.35, n),
            "hidden_posts_per_day": rng.lognormal(np.log(2), 0.5, n),
            "hidden_links_per_week": rng.lognormal(
                np.log(config.links_per_media_week), 0.6, n
            ),
            "hidden_view_coverage": coverage,
            "hidden_audience_tracked": rng.random(n) < 0.9,
        }
    )


def _true_audience(media: pd.DataFrame, media_idx, t) -> np.ndarray:
    base = media.hidden_audience_base.to_numpy()[media_idx]
    growth = media.hidden_audience_growth_per_day.to_numpy()[media_idx]
    return base * np.exp(growth * np.asarray(t) / DAY)


def _campaigns(rng, stores, end) -> pd.DataFrame:
    n_stores = len(stores)
    n = n_stores * 4
    store_idx = np.concatenate(
        [np.arange(n_stores), rng.integers(0, n_stores, n - n_stores)]
    )
    joined = stores.joined_at.to_numpy()[store_idx]
    first = np.arange(n) < n_stores  # every store starts a campaign when it joins
    later = joined + (rng.random(n) * (end - 14 * DAY - joined)).astype(np.int64)
    starts = np.where(first, joined, later)
    days = np.where(first, rng.integers(90, 181, n), rng.integers(30, 121, n))
    offer = rng.choice(len(OFFER_TYPES), n)
    pushed = rng.random(n) < 0.25
    return pd.DataFrame(
        {
            "campaign_id": _ids("campaign", n),
            "store_id": stores.store_id.to_numpy()[store_idx],
            "starts_at": starts,
            "ends_at": starts + days * DAY,
            "offer_type": np.array(OFFER_TYPES)[offer],
            "offer_source": rng.choice(["store", "network"], n, p=[0.7, 0.3]),
            "is_pushed": pushed,
            "lead_price_rials": rng.integers(5, 51, n) * 10_000,
            "publisher_share_percent": rng.integers(6, 15, n) * 5.0,
            "daily_budget_rials": rng.integers(5, 101, n) * 1_000_000,
            "_store_idx": store_idx,
            "hidden_effect": OFFER_EFFECTS[offer]
            + 0.2 * pushed
            + rng.normal(0, 0.2, n),
        }
    )


def _creatives(rng, campaigns) -> pd.DataFrame:
    counts = rng.integers(2, 7, len(campaigns))
    campaign_idx = np.repeat(np.arange(len(campaigns)), counts)
    n = len(campaign_idx)
    fmt = rng.choice(len(FORMATS), n, p=[0.55, 0.25, 0.2])
    portrait = rng.random(n) < 0.5
    has_cta = rng.random(n) < 0.7
    # Each campaign has a small asset pool, so several creatives can share one.
    pool = np.maximum(1, (counts + 1) // 2)
    offset = np.concatenate([[0], np.cumsum(pool)[:-1]])
    asset = offset[campaign_idx] + (rng.random(n) * pool[campaign_idx]).astype(int)
    return pd.DataFrame(
        {
            "creative_id": _ids("creative", n),
            "campaign_id": campaigns.campaign_id.to_numpy()[campaign_idx],
            "created_at": campaigns.starts_at.to_numpy()[campaign_idx],
            "format": np.array(FORMATS)[fmt],
            "width_px": np.full(n, 1080),
            "height_px": np.where(fmt == 1, 1920, np.where(portrait, 1350, 1080)),
            "caption_length": rng.integers(40, 600, n),
            "sequence_length": np.where(fmt == 2, rng.integers(2, 7, n), 1),
            "has_cta": has_cta,
            "asset_ref": [f"asset_{i + 1:04d}" for i in asset],
            "_campaign_idx": campaign_idx,
            "hidden_effect": rng.normal(0, 0.3, n)
            + FORMAT_EFFECTS[fmt]
            + 0.15 * has_cta,
        }
    )


def _landings(rng, stores) -> pd.DataFrame:
    counts = rng.integers(1, 4, len(stores))
    store_idx = np.repeat(np.arange(len(stores)), counts)
    return pd.DataFrame(
        {
            "landing_id": _ids("landing", len(store_idx)),
            "store_id": stores.store_id.to_numpy()[store_idx],
            "_store_idx": store_idx,
            "hidden_effect": rng.normal(0, 0.15, len(store_idx)),
        }
    )


def _links(rng, config, media, campaigns, creatives, landings, end) -> pd.DataFrame:
    hour_p = CREATION_HOUR_WEIGHTS / CREATION_HOUR_WEIGHTS.sum()
    first_active = media.hidden_first_active_at.to_numpy()
    rate = media.hidden_links_per_week.to_numpy()
    media_parts, time_parts = [], []
    for m in range(len(media)):
        start = first_active[m]
        k = rng.poisson(rate[m] * (end - start) / (7 * DAY))
        day = rng.integers(start // DAY, config.days, k)
        t = day * DAY + rng.choice(24, k, p=hour_p) * HOUR + rng.integers(0, HOUR, k)
        t = t[(t >= start) & (t < end)]
        media_parts.append(np.full(len(t), m))
        time_parts.append(t)
    media_idx = np.concatenate(media_parts)
    created = np.concatenate(time_parts)
    order = np.lexsort((media_idx, created))
    media_idx, created = media_idx[order], created[order]

    starts = campaigns.starts_at.to_numpy()
    ends = campaigns.ends_at.to_numpy()
    budget = campaigns.daily_budget_rials.to_numpy().astype(float)
    campaign_store = campaigns._store_idx.to_numpy()
    by_campaign = _group(creatives._campaign_idx.to_numpy(), len(campaigns))
    by_store = _group(landings._store_idx.to_numpy(), int(campaign_store.max()) + 1)

    kept, chosen_campaign, chosen_creative, chosen_landing = [], [], [], []
    for i, t in enumerate(created):
        active = np.flatnonzero((starts <= t) & (t < ends))
        if not len(active):
            continue  # no campaign running, so no link is created
        c = rng.choice(active, p=budget[active] / budget[active].sum())
        creative = rng.choice(by_campaign[c]) if rng.random() >= 0.08 else -1
        store_landings = by_store[campaign_store[c]]
        landing = rng.choice(store_landings) if rng.random() >= 0.05 else -1
        kept.append(i)
        chosen_campaign.append(c)
        chosen_creative.append(creative)
        chosen_landing.append(landing)

    kept = np.array(kept, dtype=int)
    campaign_idx = np.array(chosen_campaign, dtype=int)
    creative_idx = np.array(chosen_creative, dtype=int)
    landing_idx = np.array(chosen_landing, dtype=int)
    creative_ids = creatives.creative_id.to_numpy()
    landing_ids = landings.landing_id.to_numpy()
    return pd.DataFrame(
        {
            "link_id": _ids("link", len(kept)),
            "media_id": media.media_id.to_numpy()[media_idx[kept]],
            "campaign_id": campaigns.campaign_id.to_numpy()[campaign_idx],
            "creative_id": [creative_ids[j] if j >= 0 else None for j in creative_idx],
            "landing_id": [landing_ids[j] if j >= 0 else None for j in landing_idx],
            "created_at": created[kept],
            "_media_idx": media_idx[kept],
            "_campaign_idx": campaign_idx,
            "_creative_idx": creative_idx,
            "_landing_idx": landing_idx,
        }
    )


def _group(keys: np.ndarray, n: int) -> list[np.ndarray]:
    """Row indices for each key 0..n-1, preserving the original row order."""
    order = np.argsort(keys, kind="stable")
    bounds = np.searchsorted(keys[order], np.arange(n + 1))
    return [order[bounds[i] : bounds[i + 1]] for i in range(n)]


# --- Timestamped observations ----------------------------------------------


def _posts(rng, media, links, end) -> tuple[pd.DataFrame, pd.DataFrame]:
    created = media.created_at.to_numpy()
    per_day = media.hidden_posts_per_day.to_numpy()
    media_parts, published_parts = [], []
    for m in range(len(media)):
        start = max(created[m], 0)  # post history is collected from data start
        k = rng.poisson(per_day[m] * (end - start) / DAY)
        media_parts.append(np.full(k, m))
        published_parts.append(rng.integers(start, end, k))
    organic_media = np.concatenate(media_parts)
    organic_published = np.concatenate(published_parts)

    # About 90% of links are published in a known post shortly after creation.
    n_links = len(links)
    has_post = rng.random(n_links) < 0.9
    ad_published = links.created_at.to_numpy() + rng.integers(0, 6 * HOUR, n_links)

    media_idx = np.concatenate([organic_media, links._media_idx.to_numpy()[has_post]])
    published = np.concatenate([organic_published, ad_published[has_post]])
    link_idx = np.concatenate(
        [np.full(len(organic_media), -1), np.flatnonzero(has_post)]
    )
    known = published + rng.exponential(HOUR, len(published)).astype(np.int64)
    associated = known + rng.integers(0, 2 * HOUR, len(known))

    keep = known <= end
    order = np.lexsort((media_idx[keep], published[keep]))
    media_idx, published, known, link_idx, associated = (
        array[keep][order]
        for array in (media_idx, published, known, link_idx, associated)
    )
    post_ids = np.array(_ids("post", len(published)), dtype=object)
    posts = pd.DataFrame(
        {
            "post_id": post_ids,
            "media_id": media.media_id.to_numpy()[media_idx],
            "published_at": published,
            "known_at": known,
            "_media_idx": media_idx,
        }
    )
    linked = (link_idx >= 0) & (associated <= end)
    post_links = pd.DataFrame(
        {
            "post_id": post_ids[linked],
            "link_id": links.link_id.to_numpy()[link_idx[linked]],
            "associated_at": associated[linked],
        }
    )
    return posts, post_links


def _view_observations(rng, media, posts, end) -> pd.DataFrame:
    n = len(posts)
    media_idx = posts._media_idx.to_numpy()
    published = posts.published_at.to_numpy()
    measured = rng.random(n) < media.hidden_view_coverage.to_numpy()[media_idx]
    measured_at = published + rng.integers(HOUR, 72 * HOUR, n)
    received_at = measured_at + rng.integers(0, 2 * HOUR, n)
    reach = (
        _true_audience(media, media_idx, published)
        * media.hidden_engagement.to_numpy()[media_idx]
        * rng.lognormal(0, 0.5, n)
    )
    views = np.round(reach * (1 - np.exp(-(measured_at - published) / (10 * HOUR))))
    keep = measured & (received_at <= end)
    return pd.DataFrame(
        {
            "post_id": posts.post_id.to_numpy()[keep],
            "kind": "views",
            "measured_at": measured_at[keep],
            "received_at": received_at[keep],
            "cumulative_value": views[keep].astype(np.int64),
            "_post_idx": np.flatnonzero(keep),
        }
    )


def _audience_observations(rng, config, media, end) -> pd.DataFrame:
    n_weeks = config.days // 7 + 1
    media_idx = np.repeat(np.arange(len(media)), n_weeks)
    week = np.tile(np.arange(n_weeks), len(media))
    measured = week * 7 * DAY + rng.integers(0, DAY, len(week))
    received = measured + rng.integers(0, 12 * HOUR, len(week))
    noise = rng.lognormal(0, 0.02, len(week))
    keep = (
        media.hidden_audience_tracked.to_numpy()[media_idx]
        & (measured >= np.maximum(media.created_at.to_numpy()[media_idx], 0))
        & (rng.random(len(week)) < 0.85)
        & (received <= end)
    )
    values = np.round(_true_audience(media, media_idx, measured) * noise)
    return pd.DataFrame(
        {
            "media_id": media.media_id.to_numpy()[media_idx[keep]],
            "measured_at": measured[keep],
            "received_at": received[keep],
            "audience_size": values[keep].astype(np.int64),
            "_media_idx": media_idx[keep],
        }
    )


def _conversions(
    rng, config, media, stores, campaigns, creatives, landings, links, end
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_links = len(links)
    m = links._media_idx.to_numpy()
    c = links._campaign_idx.to_numpy()
    cr = links._creative_idx.to_numpy()
    ln = links._landing_idx.to_numpy()
    created = links.created_at.to_numpy()
    store = campaigns._store_idx.to_numpy()[c]
    hour = (created % DAY) // HOUR

    # Hidden data-generating process (log scale). A model can only recover these
    # effects through features such as media/store IDs, audience, and views.
    log_rate = (
        0.5 * np.log(_true_audience(media, m, created) / 20_000)
        + 0.8 * np.log(media.hidden_engagement.to_numpy()[m] / 0.08)
        + media.hidden_quality.to_numpy()[m]
        + stores.hidden_appeal.to_numpy()[store]
        + campaigns.hidden_effect.to_numpy()[c]
        + np.where(cr >= 0, creatives.hidden_effect.to_numpy()[cr], 0.0)
        + np.where(ln >= 0, landings.hidden_effect.to_numpy()[ln], 0.0)
        + np.where((hour >= 14) & (hour < 20), 0.2, np.where(hour < 6, -0.3, 0.0))
        + 0.35
        * (stores.hidden_category.to_numpy()[store] == media.category.to_numpy()[m])
        - 0.2 * (media.platform.to_numpy()[m] == "instagram")
    )
    raw = np.exp(log_rate)
    scale = config.mean_conversions_7d / (
        raw.mean() * WINDOW_SHARE * config.approval_rate
    )
    rate = raw * scale  # expected conversions over all delays, before review
    expected_7d = rate * WINDOW_SHARE * config.approval_rate

    # Gamma-Poisson (negative binomial) counts: many zeros, a few large links.
    counts = rng.poisson(rng.gamma(config.dispersion, rate / config.dispersion))
    link_idx = np.repeat(np.arange(n_links), counts)
    n = len(link_idx)
    fast = rng.random(n) < FAST_SHARE
    delay = np.where(
        fast,
        rng.exponential(FAST_MEAN_HOURS * HOUR, n),
        rng.uniform(0, TAIL_HOURS * HOUR, n),
    ).astype(np.int64)
    occurred = created[link_idx] + delay

    draw = rng.random(n_links)
    tracking = np.where(
        draw < config.outage_rate,
        "outage",
        np.where(
            draw < config.outage_rate + config.unverified_rate, "unverified", "complete"
        ),
    )
    recorded = np.where(
        tracking[link_idx] == "outage", rng.random(n) < OUTAGE_RECORDED_SHARE, True
    )
    received = occurred + rng.integers(0, 3 * HOUR, n)
    decided = occurred + rng.integers(12 * HOUR, 96 * HOUR, n)
    approved = rng.random(n) < config.approval_rate

    keep = recorded & (received <= end)
    order = np.lexsort((link_idx[keep], occurred[keep]))
    link_idx, occurred, received, decided, approved = (
        array[keep][order]
        for array in (link_idx, occurred, received, decided, approved)
    )
    is_decided = decided <= end
    conversions = pd.DataFrame(
        {
            "conversion_id": _ids("conversion", len(link_idx)),
            "link_id": links.link_id.to_numpy()[link_idx],
            "occurred_at": occurred,
            "received_at": received,
            "status": np.where(
                is_decided, np.where(approved, "approved", "rejected"), "under_review"
            ),
            "decided_at": np.where(is_decided, decided, np.nan),
            "_link_idx": link_idx,
        }
    )

    # Tracking coverage is reviewed once the window and reporting buffer pass.
    reviewed_at = created + WINDOW + config.reporting_delay_hours * HOUR
    reviewed = reviewed_at <= end
    coverage = np.select(
        [tracking == "complete", tracking == "outage"],
        ["complete", "incomplete"],
        "unknown",
    )
    reviews = pd.DataFrame(
        {
            "link_id": links.link_id.to_numpy()[reviewed],
            "coverage": coverage[reviewed],
            "reviewed_at": reviewed_at[reviewed],
        }
    )

    links["hidden_tracking"] = tracking
    links["hidden_expected_conversions_7d"] = expected_7d
    return conversions, reviews


# --- Point-in-time link rows -----------------------------------------------


def _link_rows(
    config, media, campaigns, creatives, links, posts, views, audience,
    conversions, reviews, end,
) -> list[LinkRow]:  # fmt: skip
    def ts(seconds) -> dt.datetime:
        return config.start + dt.timedelta(seconds=int(seconds))

    n_media = len(media)
    post_groups = _group(posts._media_idx.to_numpy(), n_media)
    published = posts.published_at.to_numpy()
    known = posts.known_at.to_numpy()
    view_received = np.full(len(posts), np.inf)
    view_value = np.zeros(len(posts))
    view_received[views._post_idx.to_numpy()] = views.received_at.to_numpy()
    view_value[views._post_idx.to_numpy()] = views.cumulative_value.to_numpy()

    audience = audience.sort_values(["_media_idx", "received_at"], kind="stable")
    audience_groups = _group(audience._media_idx.to_numpy(), n_media)
    audience_received = audience.received_at.to_numpy()
    audience_value = audience.audience_size.to_numpy()

    link_media = links._media_idx.to_numpy()
    link_created = links.created_at.to_numpy()
    link_groups = _group(link_media, n_media)

    conversion_groups = _group(conversions._link_idx.to_numpy(), len(links))
    occurred = conversions.occurred_at.to_numpy()
    status = conversions.status.to_numpy()
    decided = conversions.decided_at.to_numpy()
    review_by_link = dict(zip(reviews.link_id, reviews.coverage))

    def column(frame, name, idx):
        return frame[name].to_numpy()[idx].tolist()

    camp_idx = links._campaign_idx.to_numpy()
    cre_idx = links._creative_idx.to_numpy()
    cre_rows = np.maximum(cre_idx, 0)
    link_values = {
        "link_id": links.link_id.tolist(),
        "media_id": links.media_id.tolist(),
        "campaign_id": links.campaign_id.tolist(),
        "creative_id": links.creative_id.tolist(),
        "landing_id": links.landing_id.tolist(),
        "platform": column(media, "platform", link_media),
        "publisher_id": column(media, "publisher_id", link_media),
        "media_category": column(media, "category", link_media),
        "media_created": column(media, "created_at", link_media),
        "store_id": column(campaigns, "store_id", camp_idx),
        "lead_price_rials": column(campaigns, "lead_price_rials", camp_idx),
        "publisher_share_percent": column(
            campaigns, "publisher_share_percent", camp_idx
        ),
        "campaign_daily_budget_rials": column(
            campaigns, "daily_budget_rials", camp_idx
        ),
        "offer_type": column(campaigns, "offer_type", camp_idx),
        "offer_source": column(campaigns, "offer_source", camp_idx),
        "is_pushed": column(campaigns, "is_pushed", camp_idx),
    }
    creative_values = {
        "asset_ref": "asset_ref",
        "creative_format": "format",
        "creative_width_px": "width_px",
        "creative_height_px": "height_px",
        "caption_length": "caption_length",
        "creative_sequence_length": "sequence_length",
        "has_cta": "has_cta",
    }
    creative_values = {
        name: [
            value if j >= 0 else None
            for value, j in zip(column(creatives, source, cre_rows), cre_idx)
        ]
        for name, source in creative_values.items()
    }

    rows = []
    for i in range(len(links)):
        m, t = link_media[i], int(link_created[i])
        value = {name: values[i] for name, values in link_values.items()}
        value |= {name: values[i] for name, values in creative_values.items()}
        has_history = t - WINDOW >= 0  # sources are only collected from data start

        audience_size = audience_at = None
        group = audience_groups[m]
        k = np.searchsorted(audience_received[group], t, side="right") - 1
        if k >= 0:
            audience_size = int(audience_value[group[k]])
            audience_at = ts(audience_received[group[k]])

        posts_7d = measured_7d = median_views = None
        links_7d = None
        if has_history:
            group = post_groups[m]
            in_window = (
                (published[group] >= t - WINDOW)
                & (published[group] < t)
                & (known[group] <= t)
            )
            measured = in_window & (view_received[group] <= t)
            posts_7d, measured_7d = int(in_window.sum()), int(measured.sum())
            if measured_7d:
                median_views = float(np.median(view_value[group][measured]))
            earlier = link_created[link_groups[m]]
            links_7d = int(((earlier >= t - WINDOW) & (earlier < t)).sum())
        earlier = link_created[link_groups[m]]
        earlier = earlier[earlier < t]
        since_previous = float((t - earlier[-1]) / HOUR) if len(earlier) else None

        label = _label(
            t, conversion_groups[i], occurred, status, decided,
            review_by_link.get(value["link_id"]), config, end,
        )  # fmt: skip
        media_created = value.pop("media_created")
        rows.append(
            LinkRow(
                schema_version=SCHEMA_VERSION,
                created_at=ts(t),
                snapshot_at=ts(end),
                # The link's own configuration first becomes available at creation.
                features_available_at=ts(t),
                window_end=ts(t + WINDOW),
                reporting_delay_hours=config.reporting_delay_hours,
                label_status=label["label_status"],
                tracking_coverage=label["tracking_coverage"],
                label_available_at=label["label_available_at"],
                purchase_conversions_7d=label["target"],
                observed_through=label["observed_through"],
                observed_purchase_conversions=label["observed"],
                audience_size=audience_size,
                audience_observed_at=audience_at,
                media_age_days=round((t - media_created) / DAY, 3),
                previous_posts_7d=posts_7d,
                previous_posts_with_views_7d=measured_7d,
                previous_posts_median_observed_views_7d=median_views,
                previous_links_7d=links_7d,
                hours_since_previous_link=since_previous,
                **value,
            )
        )
    return rows


def _label(t, group, occurred, status, decided, coverage, config, end) -> dict:
    """Assemble a seven-day label exactly as an exporter would at snapshot time."""
    window_end = t + WINDOW
    in_window = group[occurred[group] < window_end]
    available = max(
        window_end + config.reporting_delay_hours * HOUR,
        np.nanmax(decided[in_window], initial=0),
    )
    decided_all = not (status[in_window] == "under_review").any()
    result = dict(
        label_status="pending",
        tracking_coverage="unknown" if coverage is None else coverage,
        label_available_at=None,
        target=None,
        observed_through=None,
        observed=None,
    )
    if coverage is not None and coverage != "complete":
        result["label_status"] = "unknown"  # mature window, untrustworthy label
    elif coverage == "complete" and decided_all and available <= end:
        result["label_status"] = "ready"
        result["label_available_at"] = config.start + dt.timedelta(
            seconds=int(available)
        )
        result["target"] = int((status[in_window] == "approved").sum())
    else:
        # Pending: keep a partial count of non-rejected conversions known so far.
        through = min(window_end, end)
        partial = group[occurred[group] < through]
        result["observed_through"] = config.start + dt.timedelta(seconds=int(through))
        result["observed"] = int((status[partial] != "rejected").sum())
    return result


# --- Export helpers --------------------------------------------------------


def _format_times(frame: pd.DataFrame, start: dt.datetime) -> pd.DataFrame:
    frame = frame.copy()
    for column in frame.columns:
        if column.endswith("_at"):
            times = pd.Timestamp(start) + pd.to_timedelta(frame[column], unit="s")
            frame[column] = times.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return frame


def _public(frame: pd.DataFrame, start: dt.datetime) -> pd.DataFrame:
    columns = [c for c in frame.columns if not c.startswith(("_", "hidden_"))]
    return _format_times(frame[columns], start)


def _hidden(frame: pd.DataFrame, start: dt.datetime) -> pd.DataFrame:
    key = frame.columns[0]
    columns = [key] + [c for c in frame.columns if c.startswith("hidden_")]
    return _format_times(frame[columns], start)
