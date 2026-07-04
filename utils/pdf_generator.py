"""Render structured practice paper data as a formatted PDF.

Uses Arial Unicode MS (macOS system font) for full Unicode math coverage:
Greek letters (α, β, Σ, μ, π…), math operators (∫, √, ∞, ±, ≤, ≥…),
arrows (→, ⇒…), and proper typographic punctuation (—, •…).
Falls back to Helvetica + ASCII sanitization if the font is unavailable.
"""

import re
import os
from fpdf import FPDF

# ── Font resolution ────────────────────────────────────────

_UNICODE_FONT_PATH = None  # cached result


def _find_unicode_font() -> str | None:
    """Find a TTF font with broad Unicode coverage on the system."""
    global _UNICODE_FONT_PATH
    if _UNICODE_FONT_PATH is not None:
        return _UNICODE_FONT_PATH or None

    candidates = [
        # macOS
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        # Windows
        "C:\\Windows\\Fonts\\arialuni.ttf",
        "C:\\Windows\\Fonts\\Arial.ttf",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/dejavu/DejaVuSans.ttf",
        # Fallback: any DejaVu Sans on the system
    ]

    for path in candidates:
        if os.path.isfile(path):
            _UNICODE_FONT_PATH = path
            return path

    _UNICODE_FONT_PATH = ""
    return None


# ── LaTeX → Unicode conversion ─────────────────────────────

LATEX_TO_UNICODE = {
    r"\alpha": "α",
    r"\beta": "β",
    r"\gamma": "γ",
    r"\delta": "δ",
    r"\epsilon": "ε",
    r"\theta": "θ",
    r"\lambda": "λ",
    r"\mu": "μ",
    r"\pi": "π",
    r"\sigma": "σ",
    r"\tau": "τ",
    r"\phi": "φ",
    r"\omega": "ω",
    r"\Sigma": "Σ",
    r"\int": "∫",
    r"\sqrt": "√",
    r"\infty": "∞",
    r"\pm": "±",
    r"\leq": "≤",
    r"\geq": "≥",
    r"\neq": "≠",
    r"\approx": "≈",
    r"\times": "×",
    r"\div": "÷",
    r"\cdot": "·",
    r"\rightarrow": "→",
    r"\Rightarrow": "⇒",
    r"\leftarrow": "←",
    r"\Leftarrow": "⇐",
    r"\prod": "∏",
    r"\sum": "∑",
    r"\partial": "∂",
    r"\nabla": "∇",
}

# Unicode superscript / subscript mappings — restricted to characters we've
# verified have real glyphs in the installed font (digits + a few symbols).
# Letter glyphs are missing from most Unicode-coverage fonts, including
# Arial Unicode MS (confirmed via fontTools: 100% of subscript letters and
# 64% of superscript letters have no glyph) — so letters are deliberately
# NOT mapped here. See _convert_script_group for how mixed/letter groups
# are handled instead of silently emitting an unsupported glyph.
_SAFE_SCRIPT_CHARS = "0123456789+-=()"
_SUPER_MAP = str.maketrans(dict(zip(_SAFE_SCRIPT_CHARS, "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾")))
_SUB_MAP = str.maketrans(dict(zip(_SAFE_SCRIPT_CHARS, "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎")))
_SAFE_SCRIPT_SET = set(_SAFE_SCRIPT_CHARS)


def _normalize_script_delimiters(text: str, marker: str) -> str:
    """Collapse the different ways exponents/subscripts show up in generated
    text — bare (X^n), parenthesized (X^(n-1)), or braced (X^{n-1}) — into
    one canonical `marker{...}` form, so conversion only has to handle a
    single input shape instead of special-casing each style.
    """
    esc = re.escape(marker)
    # marker(...) -> marker{...}  (explicit parens, unambiguous boundary)
    text = re.sub(rf"{esc}\(([^()]*)\)", rf"{marker}{{\1}}", text)
    # bare marker + signed digit-run or single letter, not already braced
    text = re.sub(rf"{esc}(?!\{{)(-?(?:\d+|[A-Za-z]))", rf"{marker}{{\1}}", text)
    return text


def _convert_script_group(text: str, marker: str, unicode_map) -> str:
    """Convert normalized `marker{...}` groups to Unicode super/subscript,
    but only when every character in the group has a verified-safe glyph
    mapping. A group containing any unsafe character (typically a letter)
    falls back to plain `marker...` text as a whole, instead of silently
    dropping the raised formatting for just that character or emitting a
    codepoint the font can't render.
    """
    esc = re.escape(marker)

    def _replace(m):
        inner = m.group(1)
        if inner and all(ch in _SAFE_SCRIPT_SET for ch in inner):
            return inner.translate(unicode_map)
        # multi-char fallback keeps an explicit boundary (marker(...)) so it
        # stays unambiguous; a single character reads fine bare (X^n).
        return f"{marker}{inner}" if len(inner) == 1 else f"{marker}({inner})"

    return re.sub(rf"{esc}\{{([^}}]+)\}}", _replace, text)


def _convert_superscripts(text: str) -> str:
    """Convert power notation (X^2, X^(2), X^{2}, X^n, e^{-x}, ...) to Unicode
    superscript where every character is safely mappable, else a plain
    readable `^` fallback (e.g. X^{n} -> X^n, never a partial/blank glyph)."""
    text = _normalize_script_delimiters(text, "^")
    return _convert_script_group(text, "^", _SUPER_MAP)


def _convert_subscripts(text: str) -> str:
    """Convert subscript notation (X_1, X_(1), X_{1}, X_i, ...) to Unicode
    subscript where every character is safely mappable, else a plain
    readable `_` fallback (e.g. X_{i} -> X_i)."""
    text = _normalize_script_delimiters(text, "_")
    return _convert_script_group(text, "_", _SUB_MAP)



# Fallback ASCII mapping — only used when no Unicode font is available
_UNICODE_FALLBACK = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta",
    "ε": "epsilon", "θ": "theta", "λ": "lambda", "μ": "mu",
    "π": "pi", "σ": "sigma", "τ": "tau", "φ": "phi", "ω": "omega",
    "Σ": "Sigma", "∫": "∫(integral)", "√": "sqrt", "∞": "inf",
    "±": "+/-", "≤": "<=", "≥": ">=", "≠": "!=", "≈": "~=",
    "×": "*", "÷": "/", "·": "*", "→": "->", "⇒": "=>",
    "←": "<-", "—": "--", "–": "-", "•": "*",
    "∏": "Pi", "∑": "Sigma", "∂": "d", "∇": "nabla",
}


# Vulgar fractions — common fractions → single Unicode character
_VULGAR_FRACTIONS = {
    "1/2": "½", "1/3": "⅓", "2/3": "⅔",
    "1/4": "¼", "3/4": "¾",
    "1/5": "⅕", "2/5": "⅖", "3/5": "⅗", "4/5": "⅘",
    "1/6": "⅙", "5/6": "⅚",
    "1/8": "⅛", "3/8": "⅜", "5/8": "⅝", "7/8": "⅞",
    "1/7": "⅐", "1/9": "⅑", "1/10": "⅒",
}

# ASCII math sequences → Unicode symbols (applied before fractions so -> doesn't break)
_ASCII_MATH = [
    ("<=", "≤"), (">=", "≥"), ("!=", "≠"), ("~=", "≈"),
    ("->", "→"), ("=>", "⇒"), ("<-", "←"),
    ("+/-", "±"), ("+-", "±"),
    ("...", "…"),
]


def _convert_math_ascii(text: str) -> str:
    """Convert ASCII math sequences and fractions to proper Unicode symbols."""
    # Fractions (e.g., "1/2" → "½") — avoid converting URLs or dates
    for ascii_frac, uni_frac in _VULGAR_FRACTIONS.items():
        # Only replace when bounded by spaces, operators, or punctuation (not inside numbers like 21/2)
        text = re.sub(
            r"(?<=[\s(\[=+\-*/∀-⋿])" + re.escape(ascii_frac) + r"(?=[\s)\]=,;.+\-*/]|$)",
            uni_frac, text,
        )
    # Arrow / comparison symbols
    for ascii_sym, uni_sym in _ASCII_MATH:
        text = text.replace(ascii_sym, uni_sym)
    return text


def _clean_latex(text: str) -> str:
    """Convert LaTeX commands to Unicode, strip formatting wrappers, keep content.
    Also converts caret-power (X^2), underscore-subscript (X_0), fractions (1/2→½),
    and ASCII math (<=→≤, ->→→, etc.) to proper Unicode for PDF rendering.
    """
    # LaTeX commands → Unicode symbols
    for cmd, unicode_char in LATEX_TO_UNICODE.items():
        text = text.replace(cmd, unicode_char)

    # Strip formatting wrappers, keep inner content
    text = re.sub(r"\\textbf\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", r"\1", text)
    text = re.sub(r"\\textit\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", r"\1", text)
    text = re.sub(r"\\text\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", r"\1", text)
    text = re.sub(r"\\underline\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", r"\1", text)
    # \frac{a}{b} → (a)/(b)
    text = re.sub(
        r"\\frac\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}",
        r"(\1)/(\2)", text,
    )
    # Math mode delimiters
    text = re.sub(r"\\\[([^\]]*)\\\]", r"\1", text)
    text = re.sub(r"\$([^$]*)\$", r"\1", text)
    # Stray backslashes
    text = text.replace("\\", "")

    # Convert caret-power and underscore-subscript to Unicode
    text = _convert_superscripts(text)
    text = _convert_subscripts(text)

    # Convert ASCII math and fractions to Unicode symbols
    text = _convert_math_ascii(text)

    return text.strip()


def _maybe_sanitize(text: str) -> str:
    """If no Unicode font is available, fall back to ASCII. Otherwise pass through."""
    if _find_unicode_font():
        return text
    # Fallback: map Unicode → ASCII
    result = []
    for ch in text:
        if ord(ch) < 128:
            result.append(ch)
        else:
            result.append(_UNICODE_FALLBACK.get(ch, "?"))
    return "".join(result)


def _infer_time(total_marks: int) -> str:
    hours = max(1, total_marks // 30)
    return f"{hours} hour{'s' if hours > 1 else ''}"


# ── Public API ─────────────────────────────────────────────


def generate_practice_paper_pdf(
    subject: str,
    questions: list[dict],
    style: dict,
    has_slides: bool = True,
) -> bytes:
    """Render a practice exam paper as a formatted PDF.

    Questions appear first (exam paper), answer key follows at the back.
    Uses Arial Unicode MS for proper math notation (Greek, symbols, etc.),
    with transparent fallback if the font is unavailable.
    """
    font_path = _find_unicode_font()
    pdf = _build_pdf(subject, questions, style, has_slides, font_path)
    return bytes(pdf.output())


# ── PDF construction ───────────────────────────────────────


def _build_pdf(subject, questions, style, has_slides, font_path):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)

    if font_path:
        pdf.add_font("Uni", "", font_path, uni=True)
        pdf.add_font("Uni", "B", font_path, uni=True)  # same file, fpdf2 fakes bold
        pdf.add_font("Uni", "I", font_path, uni=True)  # same file, fpdf2 fakes italic
        font_name = "Uni"
    else:
        font_name = "Helvetica"

    total_marks = style.get("estimated_total_marks", 100)

    # ── Cover page / Header ────────────────────────────────
    pdf.add_page()
    pdf.set_font(font_name, "B", 20)
    pdf.cell(0, 14, _maybe_sanitize(f"PRACTICE EXAM — {subject.upper()}"), ln=True, align="C")
    pdf.ln(3)

    if not has_slides:
        pdf.set_font(font_name, "I", 10)
        pdf.set_text_color(180, 100, 0)
        pdf.multi_cell(0, 6, _maybe_sanitize(
            "Note: Generated without lecture notes — topic predictions may be less accurate."
        ), align="C")
        pdf.set_text_color(0, 0, 0)
        pdf.ln(3)

    pdf.set_font(font_name, "", 11)
    pdf.cell(0, 8, _maybe_sanitize(
        f"Total Marks: {total_marks}    |    Time Allowed: {_infer_time(total_marks)}"
    ), ln=True, align="C")
    pdf.ln(1)

    pdf.set_font(font_name, "", 10)
    pdf.cell(0, 7, "Instructions: Answer ALL questions. Marks are shown in brackets.", ln=True, align="C")
    pdf.ln(6)

    # ── Questions section ──────────────────────────────────
    pdf.set_font(font_name, "B", 14)
    pdf.cell(0, 10, "QUESTIONS", ln=True, align="C")
    pdf.ln(4)

    for q in questions:
        _render_question(pdf, q, font_name, show_answer=False)
        pdf.ln(2)

    # ── Answer Key section (new page) ──────────────────────
    pdf.add_page()
    pdf.set_font(font_name, "B", 16)
    pdf.cell(0, 12, "ANSWER KEY", ln=True, align="C")
    pdf.ln(4)

    pdf.set_font(font_name, "I", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, "Model answers for reference — do not distribute to candidates.", ln=True, align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(6)

    for q in questions:
        _render_answer_key_entry(pdf, q, font_name)
        pdf.ln(2)

    # ── End marker ─────────────────────────────────────────
    pdf.ln(4)
    pdf.set_draw_color(60, 60, 60)
    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
    pdf.ln(6)
    pdf.set_font(font_name, "B", 12)
    pdf.cell(0, 10, "--- END OF PAPER ---", ln=True, align="C")

    return pdf


# ── Question rendering (exam section — no answer) ──────────


def _render_question(pdf, q, font_name, show_answer=False):
    """Render a question for the exam paper section (no model answer visible)."""
    # Header: "Question N  [M marks]"
    pdf.set_font(font_name, "B", 12)
    header = f"Question {q.get('question_number', '?')}"
    marks = q.get("marks")
    if marks:
        header += f"  [{marks} marks]"
    pdf.cell(0, 9, _maybe_sanitize(header), ln=True)
    pdf.ln(3)

    # Body
    pdf.set_font(font_name, "", 11)
    body = _maybe_sanitize(_clean_latex(q.get("question_text", "")))
    pdf.multi_cell(0, 6.5, body)
    pdf.ln(2)

    # Subparts
    for sp in q.get("subparts", []):
        pdf.set_font(font_name, "", 11)
        sp_text = f"({sp.get('label', '?')}) {_maybe_sanitize(_clean_latex(sp.get('text', '')))}"
        sp_marks = sp.get("marks")
        if sp_marks:
            sp_text += f"  [{sp_marks} marks]"
        pdf.set_x(30)
        pdf.multi_cell(155, 6, sp_text)
        pdf.ln(2)

    # Separator
    pdf.set_draw_color(180, 180, 180)
    y = pdf.get_y()
    pdf.line(20, y, 190, y)
    pdf.ln(6)


# ── Answer key rendering ───────────────────────────────────


def _render_answer_key_entry(pdf, q, font_name):
    """Render one answer-key entry: question number + step-by-step model answer."""
    q_num = q.get("question_number", "?")

    # Question number header
    pdf.set_font(font_name, "B", 12)
    pdf.cell(0, 9, _maybe_sanitize(f"Question {q_num} — Model Answer"), ln=True)
    pdf.ln(2)

    # Repeat question text in small grey for context
    pdf.set_font(font_name, "I", 9)
    pdf.set_text_color(120, 120, 120)
    body_preview = _maybe_sanitize(_clean_latex(q.get("question_text", "")))
    if len(body_preview) > 300:
        body_preview = body_preview[:300].rsplit(" ", 1)[0] + "..."
    pdf.multi_cell(0, 5, body_preview)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    # Step-by-step explanation
    bullets = q.get("model_answer", [])
    if bullets:
        for point in bullets:
            clean = _maybe_sanitize(_clean_latex(point))
            # Detect "Step N:" prefix for bold step label
            step_match = re.match(r"(Step\s*\d+)\s*[:.\-—]\s*(.*)", clean)
            if step_match:
                step_num = step_match.group(1)
                step_text = step_match.group(2)
                pdf.set_x(20)
                pdf.set_font(font_name, "B", 10)
                pdf.cell(20, 5.5, _maybe_sanitize(step_num + ":"))
                pdf.set_font(font_name, "", 10)
                pdf.multi_cell(145, 5.5, step_text)
            else:
                pdf.set_x(22)
                pdf.set_font(font_name, "", 10)
                pdf.multi_cell(163, 5.5, _maybe_sanitize(f"• {clean}"))
            pdf.ln(1.5)
    else:
        pdf.set_x(22)
        pdf.set_font(font_name, "", 10)
        pdf.cell(0, 5.5, _maybe_sanitize("(No model answer available)"), ln=True)

    # Separator
    pdf.ln(2)
    pdf.set_draw_color(200, 200, 200)
    y = pdf.get_y()
    pdf.line(20, y, 190, y)
    pdf.ln(5)
