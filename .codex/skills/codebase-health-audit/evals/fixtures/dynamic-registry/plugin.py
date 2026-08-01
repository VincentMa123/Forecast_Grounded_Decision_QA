from registry import register


@register("normalize")
def normalize(value):
    return value.strip().lower()
