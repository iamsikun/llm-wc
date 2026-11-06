"""LaTeX to MathML for the site's formulas, at build time.

Covers the small subset the pages use: letters, numbers, operators, Greek letters, ``^`` and ``_``,
``\\sqrt{}``, ``\\hat{}``, ``\\frac{}{}``, ``\\lfloor``/``\\rfloor`` and a few relations. Browsers render
MathML natively, so the site needs no math library.
"""
from __future__ import annotations

import re
from html import escape

from markupsafe import Markup

SYMBOLS = {
    "alpha": ("mi", "α"), "beta": ("mi", "β"), "gamma": ("mi", "γ"), "delta": ("mi", "δ"),
    "varepsilon": ("mi", "ε"), "epsilon": ("mi", "ϵ"), "sigma": ("mi", "σ"), "mu": ("mi", "μ"),
    "lfloor": ("mo", "⌊"), "rfloor": ("mo", "⌋"), "gtrsim": ("mo", "≳"), "approx": ("mo", "≈"),
    "cdot": ("mo", "·"), "times": ("mo", "×"), "geq": ("mo", "≥"), "leq": ("mo", "≤"), "to": ("mo", "→"),
    ",": ("mspace", ""), "quad": ("mspace", ""),
}
TOKEN = re.compile(r"\\[A-Za-z]+|\\,|\d+(?:\.\d+)?|[A-Za-z]|\S")


class _Parser:
    def __init__(self, src: str):
        self.toks = TOKEN.findall(src)
        self.i = 0

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self):
        tok = self.toks[self.i]
        self.i += 1
        return tok

    def group(self) -> str:
        """A braced group or a single atom."""
        if self.peek() == "{":
            self.take()
            out = self.seq(stop="}")
            self.take()
            return f"<mrow>{out}</mrow>"
        return self.atom()

    def atom(self) -> str:
        tok = self.take()
        if tok == "{":
            out = self.seq(stop="}")
            self.take()
            return f"<mrow>{out}</mrow>"
        if tok.startswith("\\"):
            name = tok[1:]
            if name == "sqrt":
                return f"<msqrt>{self.group()}</msqrt>"
            if name == "hat":
                return f'<mover accent="true">{self.group()}<mo>\u02c6</mo></mover>'
            if name == "frac":
                return f"<mfrac>{self.group()}{self.group()}</mfrac>"
            tag, ch = SYMBOLS[name]
            if tag == "mspace":
                return '<mspace width="0.17em"/>' if name == "," else '<mspace width="1em"/>'
            if name in ("lfloor", "rfloor"):
                return f'<mo stretchy="false">{ch}</mo>'
            return f"<{tag}>{ch}</{tag}>"
        if re.fullmatch(r"\d+(?:\.\d+)?", tok):
            return f"<mn>{tok}</mn>"
        if re.fullmatch(r"[A-Za-z]", tok):
            return f"<mi>{tok}</mi>"
        if tok == "/":
            return '<mo lspace="0" rspace="0">/</mo>'
        if tok in "()[]":
            return f'<mo stretchy="false">{tok}</mo>'
        return f"<mo>{escape(tok)}</mo>"

    def seq(self, stop: str | None = None) -> str:
        out = []
        while self.peek() is not None and self.peek() != stop:
            base = self.atom()
            sub = sup = None
            while self.peek() in ("^", "_"):
                op = self.take()
                if op == "^":
                    sup = self.group()
                else:
                    sub = self.group()
            if sub and sup:
                base = f"<msubsup>{base}{sub}{sup}</msubsup>"
            elif sup:
                base = f"<msup>{base}{sup}</msup>"
            elif sub:
                base = f"<msub>{base}{sub}</msub>"
            out.append(base)
        return "".join(out)


def tex(src, display: bool = False) -> Markup:
    """Render a LaTeX formula as a MathML element (inline unless ``display``)."""
    body = _Parser(str(src)).seq()
    mode = ' display="block"' if display else ""
    return Markup(f"<math{mode}>{body}</math>")
