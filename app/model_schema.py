"""Remove descriptive schema repetition without changing validation constraints."""


def compact_schema(value):
    if isinstance(value,dict):
        return {key:compact_schema(item) for key,item in value.items() if key not in {'title','default'}}
    if isinstance(value,list):
        return [compact_schema(item) for item in value]
    return value
