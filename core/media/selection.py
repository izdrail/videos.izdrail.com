"""Exact preview selection, with no random/source fallback."""
import hashlib
import json
from pathlib import Path

SELECTION_JS = """() => {
 if (window.__exactClipSelection) return;
 window.__exactClipSelection = true;
 function store() {
  const selected = {};
  document.querySelectorAll('.vid-card.selected').forEach(c => selected[c.dataset.slide] = c.dataset.candidate);
  const root = document.getElementById('js-selections-storage');
  const input = root && root.querySelector('textarea,input');
  if (input) { const setter = Object.getOwnPropertyDescriptor(input.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype, 'value').set; setter.call(input, JSON.stringify(selected)); input.dispatchEvent(new Event('input', {bubbles:true})); }
 }
 document.addEventListener('click', e => {
  const card = e.target.closest('.vid-card'); if (!card) return;
  const selected = card.classList.contains('selected');
  document.querySelectorAll('.vid-card').forEach(c => {if(c.dataset.slide === card.dataset.slide) {c.classList.remove('selected'); c.setAttribute('aria-pressed','false'); const cb=c.querySelector('input'); if(cb) cb.checked=false;}});
  if(!selected) {card.classList.add('selected');card.setAttribute('aria-pressed','true');const cb=card.querySelector('input');if(cb)cb.checked=true;}
  store();
 });
 document.addEventListener('keydown', e => {if(e.target.matches('.vid-card') && (e.key === 'Enter' || e.key === ' ')){e.preventDefault();e.target.click();}});
}"""

def candidate_key(source, result):
    return hashlib.sha256((source + '\n' + str(result.get('url', ''))).encode()).hexdigest()[:24]

def resolve_selections(media_manager, selection_json, slide_data, overrides='', current_text=None):
    """Resolve tokens only against server-side preview data. Fail closed."""
    try:
        selections = json.loads(selection_json or '{}')
    except (TypeError, ValueError) as exc:
        raise ValueError('Selection data is invalid. Clear and preview again.') from exc
    if not isinstance(selections, dict):
        raise ValueError('Selection data is invalid. Clear and preview again.')
    slides = {int(s['slide_num']): s for s in (slide_data or [])}
    if selections and current_text is not None and any(s.get("script") != current_text.strip() for s in slides.values()):
        raise ValueError("Script changed since preview. Preview backgrounds again before selecting clips.")
    chosen = {}
    for raw_num, token in selections.items():
        try:
            num = int(raw_num)
            candidate = slides[num]['candidates'][token]
        except (ValueError, KeyError, TypeError):
            raise ValueError('A selected clip is stale. Clear and preview again.')
        chosen[num] = candidate
    paths = {}
    # Overrides replace choices, before any downloads.
    for line in (overrides or '').splitlines():
        line = line.strip()
        if not line or line.startswith('#'): continue
        try:
            num_text, action = line.split(':', 1)
            num = int(num_text.strip()); action = action.strip()
        except ValueError:
            raise ValueError('Use slide_number:/path/to/clip or slide_number:auto.')
        if num not in slides:
            raise ValueError(f'Slide {num} is not in the current preview.')
        if action.lower() in ('skip', 'gradient'):
            raise ValueError(f'Slide {num}: "{action}" is not supported. Use "auto" for automatic selection; slides are not skipped.')
        chosen.pop(num, None)
        if action.lower() == 'auto': continue
        path = Path(action)
        if not path.is_file():
            raise ValueError(f'Slide {num}: selected file is missing. No replacement was chosen.')
        paths[num] = str(path.resolve())
    identities = set()
    for num, candidate in chosen.items():
        identity = candidate['source'], candidate['result']['url']
        if identity in identities:
            raise ValueError(f'Slide {num}: this clip is already selected on another slide. Choose a different clip.')
        identities.add(identity)
        api = media_manager.apis.get(candidate['source'])
        download = (getattr(api, 'download_video', None) or getattr(api, 'download_photo', None)) if api else None
        if not download:
            raise ValueError(f'Slide {num}: selected source is unavailable. No replacement was chosen.')
        directory = Path(media_manager.config.VIDEOS_DIR) / 'selected'
        directory.mkdir(parents=True, exist_ok=True)
        ext = str(candidate['result'].get('ext') or 'mp4').lower().lstrip('.')
        if ext not in ('mp4','mov','webm','mkv','jpg','jpeg','png','webp','gif'): ext='mp4'
        path = directory / (candidate_key(candidate['source'], candidate['result']) + '.' + ext)
        if not path.is_file() or not path.stat().st_size:
            try:
                ok = download(candidate['result']['url'], path)
            except Exception as exc:
                raise ValueError(f'Slide {num}: selected clip download failed. No replacement was chosen.') from exc
            if not ok or not path.is_file() or not path.stat().st_size:
                raise ValueError(f'Slide {num}: selected clip download failed. No replacement was chosen.')
        paths[num] = str(path)
    resolved = [str(Path(p).resolve()) for p in paths.values()]
    if len(resolved) != len(set(resolved)):
        raise ValueError('The same file is selected on multiple slides. Choose distinct clips.')
    return paths
