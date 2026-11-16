"""Readable job labels with exact IDs retained as action values."""
import json

def job_choices(jobs, statuses):
    choices=[]
    for job in jobs:
        if job.get('status') not in statuses: continue
        params=job.get('params') or {}
        if isinstance(params,str):
            try: params=json.loads(params)
            except (ValueError,TypeError): params={}
        text=' '.join(str(params.get('text','')).split()) if isinstance(params,dict) else ''
        preview=(text[:45]+'...') if len(text)>45 else text
        created=str(job.get('created_at',''))[:16].replace('T',' ')
        label=f"{job['job_id'][:8]} | {job['status']} {job.get('progress',0)}% | {created}"
        if preview: label+=' | '+preview
        choices.append((label,job['job_id']))
    return choices
