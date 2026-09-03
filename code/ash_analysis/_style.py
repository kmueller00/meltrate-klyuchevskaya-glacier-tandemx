"""Shared figure style for the Klyuchevskaya deliverables.
 - light-grey plot background, white dashed gridlines at every tick
 - NO ASCII hyphen-minus '-' anywhere: negatives -> U+2212 minus, ranges/compounds
   -> U+2013 en dash, '->' -> arrow. Applied by monkey-patching matplotlib text.
Import + apply via sitecustomize (auto), so no per-script edits are needed.
"""
import re
_MINUS="−"; _EN="–"; _ARROW="→"

def sanitize(s):
    if not isinstance(s,str) or ("-" not in s): return s
    s=s.replace("->",_ARROW)
    s=re.sub(r'(?<=[eE])-(?=\d)', _MINUS, s)              # exponent  8e-86
    s=re.sub(r'(^|[\s(\[=/,:>])-(?=[\d.])', lambda m:m.group(1)+_MINUS, s)  # negatives
    s=re.sub(r'(?<=\d)-(?=\d)', _EN, s)                   # numeric ranges 1300-1500
    s=re.sub(r'(?<=\w)-(?=\w)', _EN, s)                   # word compounds
    s=s.replace("-",_EN)                                  # any leftover
    return s

def apply_style():
    import matplotlib
    matplotlib.rcParams.update({
        "axes.facecolor":(0.912,0.912,0.917),   # light grey plot area
        "axes.edgecolor":(0.55,0.55,0.58),
        "axes.linewidth":0.9,
        "axes.axisbelow":True,
        "axes.unicode_minus":True,
        "figure.facecolor":"white",
        "savefig.facecolor":"white",
        "savefig.dpi":600,
        "figure.dpi":140,
        "grid.color":"white","grid.linewidth":1.1,"grid.alpha":1.0,
        "grid.linestyle":(0,(4,3)),
    })
    from matplotlib.axes import Axes
    from matplotlib.text import Text
    import matplotlib.figure as mfig
    # --- grid: force white dashed at ticks whenever a script asks for a grid ---
    if not getattr(Axes,"_kv_gridpatched",False):
        _og=Axes.grid
        def grid(self,visible=True,which='major',axis='both',**kw):
            kw.pop("color",None); kw.pop("linestyle",None); kw.pop("ls",None)
            kw.pop("linewidth",None); kw.pop("lw",None); kw.pop("alpha",None)
            return _og(self,True,which=which,axis=axis,color="white",
                       linestyle=(0,(4,3)),linewidth=1.1,alpha=1.0,**kw)
        Axes.grid=grid; Axes._kv_gridpatched=True
    # --- text: sanitize hyphens everywhere text is set ---
    if not getattr(Text,"_kv_textpatched",False):
        _ost=Text.set_text
        def set_text(self,s):
            return _ost(self,sanitize(s) if isinstance(s,str) else s)
        Text.set_text=set_text; Text._kv_textpatched=True

apply_style()
