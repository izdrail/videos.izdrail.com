"""Keep header foreground explicit: Gradio styles th, not its parent tr."""
import ast
from pathlib import Path
from html.parser import HTMLParser

class Headers(HTMLParser):
    def __init__(self):
        super().__init__(); self.styles=[]
    def handle_starttag(self,tag,attrs):
        if tag=='th':self.styles.append(dict(attrs).get('style',''))

def test_every_jobs_header_sets_its_own_contrast():
    tree=ast.parse((Path(__file__).parents[1]/'main.py').read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_jobs_html')
    ns={'job_manager':None}
    exec(compile(ast.Module(body=[function],type_ignores=[]),'jobs','exec'),ns)
    ns['job_manager']=type('Jobs',(),{'get_all_jobs':lambda self,**kwargs:[{'job_id':'testjob','status':'completed','progress':100,'created_at':'2026-11-16'}]})()
    parser=Headers();parser.feed(ns['_jobs_html']())
    assert len(parser.styles)==4
    for style in parser.styles:
        assert 'background:#263247' in style and 'color:#fff' in style
