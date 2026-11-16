"""Validate Jobs action wiring without loading heavy media/model dependencies."""
import ast
from pathlib import Path
SOURCE=(Path(__file__).parents[1]/'main.py').read_text()
TREE=ast.parse(SOURCE)
def calls(name):
    return [n for n in ast.walk(TREE) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id==name]
def test_retry_failed_uses_failed_selector():
    c=next(n for n in calls('retry_btn') if n.func.attr=='click')
    assert next(k.value.elts[0].id for k in c.keywords if k.arg=='inputs')=='restart_dropdown'
def test_cancel_uses_running_selector():
    c=next(n for n in calls('cancel_btn') if n.func.attr=='click')
    assert next(k.value.elts[0].id for k in c.keywords if k.arg=='inputs')=='cancel_dropdown'
def test_summary_has_narrow_screen_scroll_wrapper():
    assert "max-width:100%;overflow-x:auto" in SOURCE
    assert 'html+="</table></div>"' in SOURCE
def test_model_default_matches_user_choice():
    assert 'value="gemma3:4b",\n                                        info="Select LLM' in SOURCE
def test_refresh_updates_cancel_selector():
    for name in ('jobs_timer','refresh_jobs_btn'):
        c=next(n for n in calls(name) if n.func.attr in ('tick','click'))
        outputs=next(k.value for k in c.keywords if k.arg=='outputs')
        assert outputs.elts[-1].id=='cancel_dropdown'
    assert "j['status'] in ('queued', 'processing')" in SOURCE

def test_retry_does_not_duplicate_completed_job():
    fn=next(n for n in ast.walk(TREE) if isinstance(n,ast.FunctionDef) and n.name=='_retry')
    class Manager:
        def get_job(self,job_id): return {'status':'completed'}
        def retry_job(self,job_id): raise AssertionError('must not enqueue completed job')
    ns={'job_manager':Manager()}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'retry','exec'),ns)
    assert ns['_retry']('existing')=='Select a failed or interrupted job to retry.'
