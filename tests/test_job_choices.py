from core.job_choices import job_choices

def test_labels_distinguish_jobs_and_keep_exact_id():
    jobs=[{'job_id':'12345678exact','status':'completed','progress':100,'created_at':'2026-11-16T09:00:00','params':'{"text":"A river flows."}'}]
    label,value=job_choices(jobs,{'completed'})[0]
    assert value=='12345678exact'
    assert 'A river flows.' in label and '2026-11-16 09:00' in label and '100%' in label
    assert job_choices(jobs,{'processing'})==[]

def test_malformed_params_does_not_break_list():
    assert job_choices([{'job_id':'id','status':'failed','params':'bad'}],{'failed'})

def test_stale_cancel_cannot_cancel_completed_job():
    import ast
    from pathlib import Path
    tree=ast.parse((Path(__file__).parents[1]/'main.py').read_text())
    fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_cancel')
    class Manager:
        def get_job(self,id): return {'status':'completed'}
        def cancel_job(self,id): raise AssertionError('must not cancel completed')
    ns={'job_manager':Manager()}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'cancel','exec'),ns)
    assert 'no longer queued' in ns['_cancel']('id')
