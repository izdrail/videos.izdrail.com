import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('exact_selection',ROOT/'core/media/selection.py')
selection=importlib.util.module_from_spec(spec);spec.loader.exec_module(selection)

class API:
    def __init__(self, ok=True): self.calls=[];self.ok=ok
    def download_video(self,url,path):
        self.calls.append(url)
        if self.ok: path.write_bytes(b'exact-clip')
        return self.ok

def fixture(tmp_path,ok=True):
    api=API(ok);manager=SimpleNamespace(apis={'Test':api},config=SimpleNamespace(VIDEOS_DIR=tmp_path))
    result={'url':'https://media.example/Exact.mp4','id':'chosen','thumbnail':'https://example/thumb.jpg'}
    key=selection.candidate_key('Test',result)
    slides=[{'slide_num':1,'candidates':{key:{'source':'Test','result':result}}}]
    return manager,api,slides,key

def test_exact_clip_and_apply_generate_cache(tmp_path):
    mgr,api,slides,key=fixture(tmp_path)
    for _ in range(3):
        got=selection.resolve_selections(mgr,json.dumps({'1':key}),slides)
        assert Path(got[1]).read_bytes()==b'exact-clip'
    assert api.calls==['https://media.example/Exact.mp4']

@pytest.mark.parametrize('payload',['bad','[]','{"1":"Test"}','{"2":"unknown"}','{"1":["Test"]}'])
def test_stale_or_invalid_selection_is_visible(tmp_path,payload):
    mgr,api,slides,key=fixture(tmp_path)
    with pytest.raises(ValueError):selection.resolve_selections(mgr,payload,slides)
    assert not api.calls

def test_download_failure_no_fallback(tmp_path):
    mgr,api,slides,key=fixture(tmp_path,False)
    with pytest.raises(ValueError,match='No replacement'):selection.resolve_selections(mgr,json.dumps({'1':key}),slides)
    assert len(api.calls)==1

def test_override_preserves_case(tmp_path):
    mgr,api,slides,key=fixture(tmp_path)
    path=tmp_path/'MyClip.MP4';path.write_bytes(b'clip')
    assert selection.resolve_selections(mgr,'{}',slides,f'1:{path}')=={1:str(path)}

def test_auto_deselects_before_download(tmp_path):
    mgr,api,slides,key=fixture(tmp_path)
    assert selection.resolve_selections(mgr,json.dumps({'1':key}),slides,'1:auto')=={}
    assert not api.calls

@pytest.mark.parametrize('action',['skip','gradient','/missing/clip.mp4'])
def test_no_false_skip_or_missing_path_fallback(tmp_path,action):
    mgr,api,slides,key=fixture(tmp_path)
    with pytest.raises(ValueError):selection.resolve_selections(mgr,'{}',slides,'1:'+action)

def test_duplicate_selection_rejected(tmp_path):
    mgr,api,slides,key=fixture(tmp_path)
    slides.append({**slides[0],'slide_num':2})
    with pytest.raises(ValueError,match='already selected'):selection.resolve_selections(mgr,json.dumps({'1':key,'2':key}),slides)

def test_all_three_ui_paths_use_exact_resolver():
    tree=ast.parse((ROOT/'main.py').read_text())
    for name in ['_submit_job_wrapper','generate_wrapper','apply_overrides_action']:
        fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
        assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='resolve_selections' for n in ast.walk(fn))
        assert not any(isinstance(n,ast.Attribute) and n.attr=='get_random_media' for n in ast.walk(fn))

def test_clear_and_preview_reset_storage():
    tree=ast.parse((ROOT/'main.py').read_text())
    for button in ['clear_preview_btn','preview_btn']:
        call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id==button)
        outputs=next(k.value for k in call.keywords if k.arg=='outputs')
        assert 'js_selections' in [e.id for e in outputs.elts]

def test_render_preserves_manual_choices():
    text=(ROOT/'core/video/ffmpeg_generator.py').read_text()
    assert 's_num not in pre_selected_videos and opt_clip' in text
    assert 'selected clip could not be preserved' in text
    assert 'source_videos_to_cleanup.add(p)' not in text

def test_single_choice_and_keyboard_js():
    assert "c.dataset.slide === card.dataset.slide" in selection.SELECTION_JS
    assert "classList.remove('selected')" in selection.SELECTION_JS
    assert "e.key === 'Enter'" in selection.SELECTION_JS

def test_preview_returns_exact_server_candidate_and_safe_text():
    tree=ast.parse((ROOT/'main.py').read_text())
    fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='preview_backgrounds_action')
    result={'url':'https://example/exact.mp4','thumbnail':'https://example/t.jpg','id':'123'}
    extractor=SimpleNamespace(clear_used=lambda:None,extract_keywords=lambda *a,**k:['test'],is_semantically_unique=lambda x:True,add_used_keyword=lambda x:None)
    media=SimpleNamespace(_get_ordered_sources=lambda x:['Test'],apis={'Test':SimpleNamespace(search_videos=lambda *a,**k:[result])})
    generator=SimpleNamespace(keyword_extractor=extractor,video_generator=SimpleNamespace(split_into_sentences=lambda x:[x],media_manager=media))
    ns={'generator':generator,'candidate_key':selection.candidate_key}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'preview','exec'),ns)
    html,slides,status,storage,applied=ns['preview_backgrounds_action']('<script>alert(1)</script>','en','Test','')
    key=selection.candidate_key('Test',result)
    assert slides[0]['candidates'][key]['result']==result
    assert f'data-candidate="{key}"' in html
    assert '<script>alert' not in html
    assert storage=='{}' and applied is None

def test_changed_script_rejects_old_selection(tmp_path):
    mgr,api,slides,key=fixture(tmp_path)
    slides[0]['script']='old text'
    with pytest.raises(ValueError,match='Script changed'):
        selection.resolve_selections(mgr,json.dumps({'1':key}),slides,current_text='new text')
    assert not api.calls

def test_selection_load_hook_is_inside_blocks_context():
    import gradio as gr
    source=(ROOT/'main.py').read_text()
    assert '\n        demo.load(fn=None, js=SELECTION_JS)\n    return demo' in source
    with gr.Blocks() as demo:
        gr.Textbox()
        demo.load(fn=None,js=selection.SELECTION_JS)
    assert demo.config['dependencies'][-1]['js']==selection.SELECTION_JS

def test_preview_text_contrast_overrides_theme():
    assert '.slide-section, .slide-section * {{ color:#e5e7eb !important; }}' in (ROOT/'main.py').read_text()
