"""Conservative script-anchored preview queries without model or network work."""
import re

_STOP = set('a an the and or of in on at by for to from through with into over under is are was were be been this that these those it its their our we they he she as has have had can could will would should scene clip video footage'.split())
_ACTIONS = set('flows flow flowing walks walk walking moves move moving shows show showing looks look looking'.split())

def terms(text):
    return list(dict.fromkeys(w for w in re.findall(r'[^\W_]+', str(text).casefold(), re.UNICODE) if len(w)>2 and w not in _STOP and w not in _ACTIONS and not w.isdecimal()))

def preview_query(sentence, keyword, manual=''):
    """Explicit user query wins; generated phrase cannot erase script subject."""
    if manual and manual.strip(): return manual.strip()[:160]
    words=terms(sentence)
    words+= [w for w in terms(keyword) if w not in words]
    return ' '.join(words[:9]) or str(keyword or '').strip()[:160]

def youtube_relevant(title, query):
    """Cheap lexical guard, not a claim of semantic accuracy.

    Keep the leading script subject and one other query term when present.
    Missing titles are held for review rather than called a good match.
    """
    wanted=terms(query); available=set(terms(title))
    if not wanted or not available: return False
    return wanted[0] in available and len(available.intersection(wanted)) >= min(2,len(wanted))
