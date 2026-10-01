# Required scene backgrounds

Scene backgrounds use available video footage first. If footage is unavailable,
the existing local SD-Turbo implementation generates an image from the scene
context. The renderer never substitutes a branded gradient, including for invalid
assets, missing intro footage, CTA scenes and clips removed by deduplication.

`VISUAL_SOURCE=ai` and `mixed` remain accepted for existing saved jobs but now obey
the same video-first policy. Retired gradient overrides use automatic selection.
User-selected media remains supported. Accent gradients on text are unchanged.

Image generation uses the existing `IMAGE_GENERATION_*` settings. No remote
image provider or gateway credential has been added. The text-only Ollama API is
not an image-generation endpoint.

If SD-Turbo cannot load or generate a valid image, the render fails rather than
silently substituting a gradient or dropping the affected scene. Check model
availability, memory and image-generation logs, then retry the job. All scenes
must pass validation before final composition.

A shared image pipeline serializes inference across scene workers. Cached PNGs
are validated, keyed by output size, and written atomically. Existing old cache
files are not deleted, but their old size-independent names are no longer used.

## Verification

Run the normal project test suite with its full dependencies. Dependency-light
boundary coverage is available via:

```
PYTHONPATH=. pytest tests/test_visual_providers.py tests/test_required_background.py -q
```

The boundary tests run the actual renderer method and FFmpeg with a fake image
generator. Production acceptance additionally requires real SD-Turbo inference
and inspection of a resulting video frame on the deployment host.
