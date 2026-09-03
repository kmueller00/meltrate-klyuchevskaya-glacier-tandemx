try:
    import _style  # applies grey bg + white dashed grid + hyphen sanitizer
except Exception as _e:
    import sys; print("style hook failed:",_e,file=sys.stderr)
