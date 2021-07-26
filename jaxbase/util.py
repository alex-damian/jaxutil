def clean_dict(d):
    return {key:val.item() for key,val in d.items()}

def Identity(*args,**kwargs):
    return lambda x: x
