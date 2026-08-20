# Media Providers & Unified Search

This document describes the open, key-less media providers added to the
`core/media/` pipeline and how `MediaManager` aggregates them.

## Providers

| Provider        | Class                  | Endpoint(s)                                                            | Media types        | Key |
|-----------------|------------------------|-----------------------------------------------------------------------|--------------------|-----|
| Openverse       | `OpenverseProvider`    | `api.openverse.org/v1/images`, `/audio`                               | image, audio       | No |
| Wikimedia Commons | `WikimediaProvider` | `commons.wikimedia.org/w/api.php` (search + imageinfo)                | image, video       | No |
| Internet Archive | `InternetArchiveProvider` | `archive.org/advancedsearch.php`, `archive.org/metadata/{id}` | image, video, audio | No |
| YouTube (yt-dlp) | `YouTubeAPI`        | `yt-dlp` subprocess (`ytsearch:`)                                      | video              | No |

The existing commercial providers (Pexels, Pixabay, Unsplash, Giphy) and the
SearXNG image search remain available. **YouTube via `yt-dlp` already existed**
and was extended with a richer `search()` API, `capabilities()`, and thumbnail /
uploader / duration metadata.

## Unified interface

Every provider subclasses `BaseMediaAPI` and exposes:

- `search_videos(query, orientation, per_page) -> List[Dict]` — the manager's
  bandit pipeline consumes these dicts (keys: `url`, `ext`, `width`, `height`,
  `thumbnail`, `title`).
- `search(query, media_type, limit) -> List[Media]` — the richer discovery API
  returning `Media` dataclasses (carries `license`, `license_version`,
  `attribution`, `creator`, `media_type`, etc.).
- `download_video(url, output_path)` — direct `requests` download, except
  YouTube which shells out to `yt-dlp`.
- `capabilities()` — declares supported `media_type`s and whether a key is
  required, so the manager can route requests.

The shared `Media` dataclass and `MediaType` enum live in `core/media/base.py`.

## Aggregation strategy (`MediaManager.search`)

`manager.search(query, media_type=MediaType.ANY, limit=50, min_results=50)`:

1. Walks `preferred_order` (open providers first):
   `Openverse → Wikimedia → SearXNG → InternetArchive → YouTube → Pexels →
   Pixabay → Unsplash → Giphy`.
2. Skips a provider when it doesn't support the requested `media_type` or
   requires a missing API key (per `capabilities()`).
3. Collects results from each provider until `min_results` are gathered or the
   list is exhausted.
4. **De-duplicates** by a canonical URL key (scheme+netloc+path, lower-cased);
   falls back to `title|source` when no URL is present.

The bandit pipeline (`get_random_media` / `_search_and_download`) is unchanged in
behaviour but now also searches the open providers first, because they declare
`requires_key: False` and are no longer filtered out by the missing-key check.

## Error handling

All providers wrap network/CLI failures, log a warning, and return an empty list
(`FileNotFoundError` for a missing `yt-dlp` binary is caught). The manager never
crashes if a provider fails — it simply continues with the next source.

## Graceful degradation

- Open providers need no API keys, so the system keeps working (with freely
  licensed media) even when every commercial key is absent.
- `yt-dlp` is optional; if not installed, YouTube contributes no results.

## Testing

`tests/test_media_providers.py` mocks the HTTP/CLI layers and runs offline. The
`MediaManager` tests `importorskip("spacy")` because `manager.py` pulls in the
neuron extractor.
