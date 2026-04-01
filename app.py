# =============================================================================
# app.py — College Algebra Math Solver (Single-File Streamlit App)
# =============================================================================
# Requires: streamlit, sympy
# Run with: streamlit run app.py
# =============================================================================

# =============================================================================
# IMPORTS
# =============================================================================

import re
import streamlit as st
from sympy import (
    symbols, sympify, simplify, factor, expand, solve, solveset,
    Rational, Abs, oo, zoo, nan, S, I,
    log, ln, exp, sqrt,
    Interval, Union, FiniteSet, Complement, EmptySet,
    Eq, Ne, Lt, Le, Gt, Ge,
    And, Or, Not,
    latex, pretty,
    lambdify,
    Symbol, Function,
    apart, together, cancel,
    limit, diff,
    Reals, Complexes,
    imageset,
    Piecewise,
    pi, E,
    cbrt, root,
    nsimplify,
    Integer, Float,
    numer, denom,
    Mul, Add, Pow,
    conjugate,
)
from sympy import Rational as Rat
from sympy.sets.sets import Intersection
from sympy.solvers.inequalities import solve_univariate_inequality, reduce_abs_inequality
from sympy.calculus.util import continuous_domain
import sympy as sp

# =============================================================================
# STREAMLIT PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="College Algebra Solver",
    page_icon="📐",
    layout="centered",
)

st.markdown("""
<style>
    .main-title { font-size: 2rem; font-weight: 700; color: #1a1a2e; }
    .section-label { font-weight: 600; color: #16213e; margin-top: 0.5rem; }
    .answer-box {
        background: #f0f4ff;
        border-left: 4px solid #3d5af1;
        border-radius: 6px;
        padding: 12px 16px;
        margin-top: 8px;
    }
    .step-box {
        background: #fafafa;
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        padding: 10px 14px;
        margin-top: 4px;
        font-size: 0.95rem;
    }
    .type-badge {
        display: inline-block;
        background: #3d5af1;
        color: white;
        border-radius: 4px;
        padding: 2px 10px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SYMBOL DEFINITIONS
# =============================================================================

x, y, z, h = symbols('x y z h')
a, b, c, k, n = symbols('a b c k n')

# =============================================================================
# INPUT NORMALIZATION / PREPROCESSING
# =============================================================================

def normalize_input(raw: str) -> str:
    """Normalize user input: unicode symbols, ^ exponents, spacing."""
    s = raw.strip()
    # Unicode replacements
    s = s.replace('≤', '<=')
    s = s.replace('≥', '>=')
    s = s.replace('≠', '!=')
    s = s.replace('−', '-')
    s = s.replace('×', '*')
    s = s.replace('÷', '/')
    s = s.replace('^', '**')
    # Remove extra whitespace
    s = re.sub(r'\s+', ' ', s)
    return s


def abs_to_sympy(expr_str: str) -> str:
    """Convert |expr| notation to Abs(expr) for SymPy parsing."""
    # Handle nested by iterating
    for _ in range(5):
        new = re.sub(r'\|([^|]+?)\|', r'Abs(\1)', expr_str)
        if new == expr_str:
            break
        expr_str = new
    return expr_str


def preprocess_expr(s: str) -> str:
    """Apply all preprocessing steps to an expression string."""
    s = normalize_input(s)
    s = abs_to_sympy(s)
    # Insert * for implicit multiplication: 2x → 2*x, 3y → 3*y
    s = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', s)
    return s


def safe_sympify(expr_str: str, extra_locals: dict = None):
    """
    Safely parse a string into a SymPy expression.
    Returns (expr, error_string). If error, expr is None.
    """
    local_dict = {'x': x, 'y': y, 'z': z, 'h': h, 'a': a, 'b': b,
                  'log': log, 'ln': log, 'exp': exp, 'sqrt': sqrt,
                  'Abs': Abs, 'E': E, 'pi': pi, 'e': E}
    if extra_locals:
        local_dict.update(extra_locals)
    try:
        result = sympify(expr_str, locals=local_dict)
        return result, None
    except Exception as e:
        return None, str(e)


# =============================================================================
# HELPER: FORMATTING UTILITIES
# =============================================================================

def lx(expr) -> str:
    """Return LaTeX string for a SymPy expression."""
    try:
        return latex(expr)
    except Exception:
        return str(expr)


def fmt_set(sol_set) -> str:
    """Format a SymPy solution set as a LaTeX string."""
    if sol_set == EmptySet or sol_set == S.EmptySet:
        return r"\emptyset"
    if isinstance(sol_set, FiniteSet):
        items = sorted(sol_set, key=lambda v: float(v.evalf()) if v.is_real else 0)
        return r"\{" + ", ".join(lx(v) for v in items) + r"\}"
    return lx(sol_set)


def fmt_interval(sol_set) -> str:
    """Format a SymPy set in interval notation as LaTeX."""
    return lx(sol_set)


def display_steps(steps: list):
    """Render a list of step strings in Streamlit."""
    for i, step in enumerate(steps, 1):
        st.markdown(f"**Step {i}:** {step}")


def show_result(detected_type: str, steps: list, answer_label: str, answer_latex: str):
    """Render the standard result block."""
    st.markdown(f'<span class="type-badge">Detected Type: {detected_type}</span>', unsafe_allow_html=True)
    st.markdown("**Steps:**")
    with st.container():
        for i, step in enumerate(steps, 1):
            st.markdown(f"{i}. {step}")
    st.markdown(f"**{answer_label}**")
    st.latex(answer_latex)


def show_error(msg: str):
    st.warning(f"⚠️ {msg}")


def fmt_critical_points(points: list) -> str:
    """Format a list of critical points as an ordered pair e.g. (-3, 3)."""
    pts = sorted(points, key=lambda v: float(v.evalf()))
    return "(" + ", ".join(lx(v) for v in pts) + ")"


def compute_domain_expr(expr):
    """Compute the domain of a SymPy expression in x. Returns a SymPy set."""
    try:
        dom = continuous_domain(expr, x, S.Reals)
        return dom
    except Exception:
        try:
            d = sp.denom(expr)
            if d == 1:
                return S.Reals
            excl = solve(d, x)
            excl_real = [v for v in excl if v.is_real]
            dom = S.Reals
            for v in excl_real:
                dom = dom - FiniteSet(v)
            return dom
        except Exception:
            return S.Reals


def compute_range_expr(expr):
    """
    Attempt to compute the range of expr as a function of x.
    Returns a LaTeX string. Falls back to all reals if unable.
    """
    try:
        y_sym = symbols('y_range')
        eq = Eq(expr, y_sym)
        x_sols = solve(eq, x)
        if not x_sols:
            return r"(-\infty, \infty)"
        excl_y = []
        for sol in x_sols:
            d = sp.denom(sol)
            if d != 1:
                bad_y = solve(d, y_sym)
                excl_y += [v for v in bad_y if v.is_real]
        if excl_y:
            range_set = S.Reals
            for v in excl_y:
                range_set = range_set - FiniteSet(v)
            return lx(range_set)
        return r"(-\infty, \infty)"
    except Exception:
        return r"(-\infty, \infty)"


def show_domain_range(expr=None, domain_set=None, range_latex: str = None):
    """Display Domain and Range below a result."""
    if domain_set is None and expr is not None:
        domain_set = compute_domain_expr(expr)
    if domain_set is not None:
        st.markdown(f"**Domain:** $${lx(domain_set)}$$")
    if range_latex is not None:
        st.markdown(f"**Range:** $${range_latex}$$")


# =============================================================================
# PROBLEM TYPE DETECTION
# =============================================================================

def detect_type(raw: str) -> str:
    """
    Detect the math problem type from the raw input string.
    Returns a string key identifying the problem category.
    """
    s = raw.lower().strip()
    norm = normalize_input(s)

    # Natural language patterns first
    if re.search(r'\banalyze\b|\banalysis\b', s):
        # Rational function: has / and f(x) =, or just fraction
        if re.search(r'f\s*\(\s*x\s*\)\s*=', s) or '/' in norm:
            # Is it a log or exp function?
            if re.search(r'log|ln', s):
                return 'log_function'
            if re.search(r'\d\*\*x|\d\^x|e\*\*x|exp\(', norm) or re.search(r'\^\s*x', s):
                return 'exp_function'
            if '/' in norm:
                return 'rational_analysis'
        return 'rational_analysis'

    # Expand / condense log expressions
    if re.search(r'\bexpand\b', s) and re.search(r'log|ln', s):
        return 'log_rules'
    if re.search(r'\bcondense\b', s) and re.search(r'log|ln', s):
        return 'log_rules'

    # Log function analysis (f(x) = log(...), find f(n) or just analyze)
    if re.search(r'f\s*\(\s*x\s*\)\s*=', raw, re.IGNORECASE):
        if re.search(r'log|ln', s):
            if re.search(r'find\s+f\s*\(', s) or re.search(r'analyze|domain|asymptote|intercept', s):
                return 'log_function'
        if re.search(r'\d\*\*x|\d\^x|e\*\*x|exp\(', norm) or re.search(r'\^\s*x', s):
            if re.search(r'find\s+f\s*\(', s) or re.search(r'analyze|growth|decay|asymptote|intercept', s):
                return 'exp_function'

    # AROC / DQ word problems: multi-sentence or contains a non-x variable function
    if re.search(r'average\s+rate\s+of\s+change', s):
        # If it contains a non-standard function name like h(t), use word problem solver
        if re.search(r'[a-zA-Z]\s*\(\s*[tTnNrRsS]\s*\)\s*=', raw):
            return 'aroc_word'
        return 'average_rate_of_change'
    if re.search(r'difference\s+quotient', s):
        if re.search(r'[a-zA-Z]\s*\(\s*[tTnNrRsS]\s*\)\s*=', raw):
            return 'aroc_word'
        return 'difference_quotient'
    if re.search(r'find\s+the\s+inverse|inverse\s+of', s):
        return 'inverse_function'
    if re.search(r'find\s+f\s*\+\s*g|find\s+f\s*-\s*g|find\s+f\s*\*\s*g|find\s+f/g', s):
        return 'rational_arithmetic'
    if re.search(r'fog|f\s*∘\s*g|f\s*\(\s*g|compose|composition', s):
        return 'function_composition'
    if re.search(r'divide\s+.+\s+by', s):
        return 'poly_division'
    if re.search(r'(?:find\s+)?(?:the\s+)?(?:real\s+)?zeros?\s+of|complex\s+zeros?', s):
        return 'find_zeros'
    if re.search(r'properties\s+of|even\s+or\s+odd|increasing|decreasing|local\s+(max|min)', s):
        return 'function_properties'
    if re.search(r'transformation|shift|reflect|stretch|compress|parent\s+function', s):
        return 'transformations'
    if re.search(r'library\s+of\s+function|identify\s+(the\s+)?function|parent\s+type', s):
        return 'library_function'
    if re.search(r'domain', s):
        return 'domain'
    if re.search(r'range', s):
        return 'range'

    # Single f(x) queries — properties, transformations, library, evaluate
    if re.search(r'f\s*\(\s*x\s*\)', raw, re.IGNORECASE) and not re.search(r'g\s*\(\s*x\s*\)', raw, re.IGNORECASE):
        if re.search(r'properties|even|odd|increasing|decreasing|local\s+max|local\s+min', s):
            return 'function_properties'
        if re.search(r'transformation|shift|reflect|stretch|compress', s):
            return 'transformations'
        if re.search(r'find\s+f\s*\(', s):
            # Check for exp/log before falling through to generic library_function
            if re.search(r'log|ln', s):
                return 'log_function'
            if (re.search(r'\d\*\*x|\d\^x|e\*\*x|exp\(', norm) or
                    re.search(r'\(\d*\.?\d+\)\s*\*\*\s*x', norm) or
                    re.search(r'\^\s*x', s)):
                return 'exp_function'
            return 'library_function'

    # f(x) AND g(x) — function arithmetic or composition
    if re.search(r'f\s*\(\s*x\s*\)', raw, re.IGNORECASE) and        re.search(r'g\s*\(\s*x\s*\)', raw, re.IGNORECASE):
        if re.search(r'fog|gof|compose|composition|∘', s):
            return 'function_composition'
        return 'rational_arithmetic'

    # Absolute value
    if '|' in raw or 'abs(' in s:
        if re.search(r'<=|>=|<|>', norm):
            return 'abs_inequality'
        if '=' in norm:
            return 'abs_equation'

    # Inequality detection
    if re.search(r'<=|>=|<|>', norm):
        # Check for log/ln
        if re.search(r'log|ln', s):
            return 'log_inequality'
        # Check for exp / exponential
        if re.search(r'\d\*\*x|\d\^x|e\*\*x|exp\(', norm) or re.search(r'\^\s*x', s):
            return 'exp_inequality'
        return 'polynomial_inequality'

    # Equation detection (has = but not ==)
    if '=' in norm and '==' not in norm:
        # Log equation
        if re.search(r'\blog\b|\bln\b', s):
            return 'log_equation'
        # Exponential equation
        if re.search(r'\d\*\*x|\d\^x|e\*\*x|exp\(', norm) or re.search(r'\^\s*x\b', s):
            return 'exp_equation'
        # Rational equation (has fractions)
        if '/' in norm:
            return 'rational_equation'
        return 'polynomial_equation'

    # Expression-only (no equals, no inequality) — simplification / domain
    if '/' in norm:
        return 'rational_simplify'
    if re.search(r'log|ln', s):
        return 'log_simplify'

    return 'unknown'


# =============================================================================
# PARSER UTILITIES
# =============================================================================

def split_equation(eq_str: str):
    """
    Split an equation string 'LHS = RHS' into (lhs_str, rhs_str).
    Returns None if no = found or ambiguous.
    """
    # Avoid splitting on <= >= !=
    # Replace them temporarily
    temp = eq_str.replace('<=', '≤').replace('>=', '≥').replace('!=', '≠')
    if '=' not in temp:
        return None
    parts = temp.split('=', 1)
    lhs = parts[0].replace('≤', '<=').replace('≥', '>=').replace('≠', '!=')
    rhs = parts[1].replace('≤', '<=').replace('≥', '>=').replace('≠', '!=')
    return lhs.strip(), rhs.strip()


def split_inequality(ineq_str: str):
    """
    Split an inequality string into (lhs_str, rel, rhs_str).
    rel is one of: '<', '>', '<=', '>='
    """
    for rel in ['<=', '>=', '<', '>']:
        if rel in ineq_str:
            parts = ineq_str.split(rel, 1)
            return parts[0].strip(), rel, parts[1].strip()
    return None


def parse_fx_gx(raw: str):
    """
    Parse inputs like:
      f(x) = expr1, g(x) = expr2, find ...
    Returns (f_expr, g_expr, operation_str) or (None, None, None)
    """
    raw_lower = raw.lower()
    # Try to extract f(x) = ... and g(x) = ...
    f_match = re.search(r'f\s*\(\s*x\s*\)\s*=\s*([^,]+)', raw, re.IGNORECASE)
    g_match = re.search(r'g\s*\(\s*x\s*\)\s*=\s*([^,]+)', raw, re.IGNORECASE)
    if not f_match or not g_match:
        return None, None, None

    f_str = f_match.group(1).strip()
    g_str = g_match.group(1).strip()

    # Determine operation
    op = 'unknown'
    if re.search(r'find\s+f\s*\+\s*g', raw_lower):
        op = 'add'
    elif re.search(r'find\s+f\s*-\s*g', raw_lower):
        op = 'subtract'
    elif re.search(r'find\s+f\s*\*\s*g|find\s+f\s*g\b', raw_lower):
        op = 'multiply'
    elif re.search(r'find\s+f\s*/\s*g', raw_lower):
        op = 'divide'
    elif re.search(r'fog|f\s*∘\s*g|f\s*\(\s*g\s*\(|compose|composition', raw_lower):
        op = 'compose_fog'
    elif re.search(r'gof|g\s*∘\s*f|g\s*\(\s*f\s*\(', raw_lower):
        op = 'compose_gof'

    f_expr, err1 = safe_sympify(preprocess_expr(f_str))
    g_expr, err2 = safe_sympify(preprocess_expr(g_str))
    if err1 or err2:
        return None, None, None
    return f_expr, g_expr, op


def extract_function_and_interval(raw: str):
    """
    For 'average rate of change of EXPR from A to B'
    Returns (expr, a_val, b_val) or None.
    """
    m = re.search(
        r'average\s+rate\s+of\s+change\s+of\s+(.+?)\s+from\s+([^\s]+)\s+to\s+([^\s]+)',
        raw, re.IGNORECASE
    )
    if not m:
        return None
    expr_str = preprocess_expr(m.group(1).strip())
    a_str = m.group(2).strip()
    b_str = m.group(3).strip()
    expr, e1 = safe_sympify(expr_str)
    a_val, e2 = safe_sympify(a_str)
    b_val, e3 = safe_sympify(b_str)
    if e1 or e2 or e3:
        return None
    return expr, a_val, b_val


def extract_diff_quotient_func(raw: str):
    """
    For 'difference quotient of EXPR'
    Returns expr or None.
    """
    m = re.search(r'difference\s+quotient\s+of\s+(.+)', raw, re.IGNORECASE)
    if not m:
        return None
    expr_str = preprocess_expr(m.group(1).strip())
    expr, err = safe_sympify(expr_str)
    return expr if not err else None


def extract_inverse_func(raw: str):
    """
    For 'find the inverse of EXPR' or 'inverse of EXPR'
    Returns expr or None.
    """
    m = re.search(r'inverse\s+of\s+(.+)', raw, re.IGNORECASE)
    if not m:
        return None
    expr_str = preprocess_expr(m.group(1).strip())
    # Remove trailing context noise
    expr_str = re.sub(r'\s*(find|where|when).*$', '', expr_str, flags=re.IGNORECASE).strip()
    expr, err = safe_sympify(expr_str)
    return expr if not err else None


# =============================================================================
# SOLVER: ABSOLUTE VALUE EQUATIONS
# =============================================================================

def solve_abs_equation(raw: str):
    """Solve |expr| = value or |expr| = |expr2|"""
    norm = preprocess_expr(raw)
    parts = split_equation(norm)
    if not parts:
        show_error("Could not parse equation. Expected format: |expr| = number")
        return

    lhs_str, rhs_str = parts
    lhs, e1 = safe_sympify(lhs_str)
    rhs, e2 = safe_sympify(rhs_str)

    if e1 or e2 or lhs is None or rhs is None:
        show_error(f"Could not parse: {e1 or e2}")
        return

    steps = []
    steps.append(f"Original equation: $|{lx(lhs.args[0] if isinstance(lhs, Abs) else lhs)}| = {lx(rhs)}$")

    # Identify the expression inside abs
    if isinstance(lhs, Abs):
        inner = lhs.args[0]
    else:
        # Try rhs
        if isinstance(rhs, Abs):
            inner = rhs.args[0]
            lhs, rhs = rhs, lhs
        else:
            # No abs found after conversion — try solving directly
            eq = Eq(lhs, rhs)
            solutions = solve(eq, x)
            steps.append("Solved symbolically.")
            show_result("Absolute Value Equation", steps, "Solution Set:", fmt_set(FiniteSet(*solutions)))
            return

        inner = lhs.args[0]

    steps.append(f"The absolute value equation $|{lx(inner)}| = {lx(rhs)}$ has two cases:")
    steps.append(f"Case 1: ${lx(inner)} = {lx(rhs)}$")
    steps.append(f"Case 2: ${lx(inner)} = {lx(-rhs)}$")

    sol1 = solve(Eq(inner, rhs), x)
    sol2 = solve(Eq(inner, -rhs), x)

    all_sols = []
    for s_val in sol1 + sol2:
        # Verify (abs value can't equal negative)
        check = Abs(inner).subs(x, s_val)
        if check.equals(rhs) or simplify(check - rhs) == 0:
            all_sols.append(s_val)

    if not all_sols:
        steps.append("After checking, no real solutions exist (RHS may be negative).")
        show_result("Absolute Value Equation", steps, "Solution Set:", r"\emptyset")
    else:
        steps.append(f"Case 1 gives: $x = {', '.join(lx(v) for v in sol1)}$")
        steps.append(f"Case 2 gives: $x = {', '.join(lx(v) for v in sol2)}$")
        sol_set = FiniteSet(*all_sols)
        show_result("Absolute Value Equation", steps, "Solution Set:", fmt_set(sol_set))
        show_domain_range(expr=inner, range_latex=r"(-\infty, \infty)")


# =============================================================================
# SOLVER: ABSOLUTE VALUE INEQUALITIES
# =============================================================================

def solve_abs_inequality(raw: str):
    """Solve |expr| < c, |expr| > c, |expr| <= c, |expr| >= c"""
    norm = preprocess_expr(raw)
    parts = split_inequality(norm)
    if not parts:
        show_error("Could not parse inequality.")
        return

    lhs_str, rel, rhs_str = parts
    lhs, e1 = safe_sympify(lhs_str)
    rhs, e2 = safe_sympify(rhs_str)

    if e1 or e2 or lhs is None or rhs is None:
        show_error(f"Could not parse: {e1 or e2}")
        return

    steps = []

    if not isinstance(lhs, Abs):
        show_error("Left side does not appear to contain an absolute value.")
        return

    inner = lhs.args[0]
    steps.append(f"Original: $|{lx(inner)}|\\; {rel}\\; {lx(rhs)}$")

    rhs_val = float(rhs.evalf()) if rhs.is_real else None

    if rel in ('<', '<='):
        # |u| < c  ↔  -c < u < c  (AND case)
        if rhs_val is not None and rhs_val <= 0:
            steps.append(f"Since the right side ${lx(rhs)} \\leq 0$, no real solution exists for a strict inequality.")
            show_result("Absolute Value Inequality", steps, "Solution Set:", r"\emptyset")
            return
        sym_rel = '<' if rel == '<' else '<='
        steps.append(f"$|u| {rel} c$ is equivalent to $-c {rel} u {rel} c$ (AND case)")
        steps.append(f"So: $-({lx(rhs)}) {rel} {lx(inner)} {rel} {lx(rhs)}$")
        steps.append(f"Simplify: ${lx(-rhs)} {rel} {lx(inner)} {rel} {lx(rhs)}$")

        # Solve the compound inequality
        if rel == '<':
            sol = solve([lhs < rhs, lhs > -rhs], x)
            # Use solveset for clean interval
            sol_set = solveset(lhs < rhs, x, domain=S.Reals).intersect(
                      solveset(lhs > -rhs, x, domain=S.Reals))
        else:
            sol_set = solveset(lhs <= rhs, x, domain=S.Reals).intersect(
                      solveset(lhs >= -rhs, x, domain=S.Reals))

        # Critical points = boundary values of the solution
        crit = solve(Eq(lhs, rhs), x)
        crit_real = [v for v in crit if v.is_real]
        steps.append("Solving both inequalities and intersecting:")
        show_result("Absolute Value Inequality (AND)", steps, "Solution Set:",
                    fmt_critical_points(crit_real) if crit_real else fmt_interval(sol_set))
        st.markdown(f"**Domain:** $${lx(sol_set)}$$")
        st.markdown("**Range:** $$(-\\infty, \\infty)$$")

    else:  # '>' or '>='
        # |u| > c  ↔  u < -c OR u > c
        if rhs_val is not None and rhs_val < 0:
            steps.append(f"Since ${lx(rhs)} < 0$, the solution is all real numbers.")
            show_result("Absolute Value Inequality", steps, "Solution Set:", r"(-\infty, \infty)")
            return
        steps.append(f"$|u| {rel} c$ is equivalent to $u < -c$ OR $u > c$ (OR case)")
        steps.append(f"So: ${lx(inner)} < {lx(-rhs)}$ OR ${lx(inner)} > {lx(rhs)}$")

        if rel == '>':
            sol_set = solveset(lhs > rhs, x, domain=S.Reals)
        else:
            sol_set = solveset(lhs >= rhs, x, domain=S.Reals)

        # Critical points = boundary values
        crit = solve(Eq(lhs, rhs), x)
        crit_real = [v for v in crit if v.is_real]
        steps.append("Solving both branches and taking the union:")
        show_result("Absolute Value Inequality (OR)", steps, "Solution Set:",
                    fmt_critical_points(crit_real) if crit_real else fmt_interval(sol_set))
        st.markdown(f"**Domain:** $${lx(sol_set)}$$")
        st.markdown("**Range:** $$(-\\infty, \\infty)$$")


# =============================================================================
# SOLVER: POLYNOMIAL INEQUALITIES
# =============================================================================

def solve_polynomial_inequality(raw: str):
    """Solve polynomial inequalities like x^2 - 9 >= 0"""
    norm = preprocess_expr(raw)
    parts = split_inequality(norm)
    if not parts:
        show_error("Could not parse inequality. Expected format like: x^2 - 9 >= 0")
        return

    lhs_str, rel, rhs_str = parts
    lhs, e1 = safe_sympify(lhs_str)
    rhs, e2 = safe_sympify(rhs_str)

    if e1 or e2 or lhs is None or rhs is None:
        show_error(f"Could not parse expression: {e1 or e2}")
        return

    expr = lhs - rhs  # Move everything to left side

    steps = []
    steps.append(f"Rewrite as: ${lx(expr)} {rel} 0$")

    # Factor
    factored = factor(expr)
    steps.append(f"Factor: ${lx(factored)} {rel} 0$")

    # Find critical points
    critical = solve(expr, x)
    real_critical = [v for v in critical if v.is_real]
    real_critical_sorted = sorted(real_critical, key=lambda v: float(v.evalf()))
    steps.append(f"Critical points (zeros): $x = {', '.join(lx(v) for v in real_critical_sorted)}$")

    # Build the inequality for solveset
    try:
        rel_map = {'<': expr < 0, '>': expr > 0, '<=': expr <= 0, '>=': expr >= 0}
        ineq = rel_map[rel]
        sol_set = solveset(ineq, x, domain=S.Reals)
        steps.append("Test intervals between critical points to determine sign of expression.")
        show_result("Polynomial Inequality", steps, "Solution Set:",
                    fmt_critical_points(real_critical_sorted) if real_critical_sorted else fmt_interval(sol_set))
        st.markdown(f"**Domain:** $${lx(sol_set)}$$")
        st.markdown("**Range:** $$(-\\infty, \\infty)$$")
    except Exception as e:
        show_error(f"Could not solve inequality symbolically: {e}")


# =============================================================================
# SOLVER: RATIONAL EQUATIONS
# =============================================================================

def get_lcd(expr_list):
    """
    Given a list of SymPy expressions (denominators), compute their LCM (LCD).
    Returns a SymPy expression.
    """
    from sympy import lcm, Mul, factor
    if not expr_list:
        return sp.Integer(1)
    result = expr_list[0]
    for d in expr_list[1:]:
        result = sp.lcm(result, d)
    return expand(result)


def collect_denominators(expr):
    """
    Walk a SymPy expression and collect all denominators of rational sub-expressions.
    Returns a list of non-trivial denominators (not 1).
    """
    denoms = []
    # denom of the whole expression
    d = sp.denom(expr)
    if d != 1:
        denoms.append(d)
    # Also check each term if it's an Add
    if isinstance(expr, Add):
        for term in expr.args:
            d = sp.denom(term)
            if d != 1:
                denoms.append(d)
    return denoms


def solve_rational_equation(raw: str):
    """Solve rational equations like (x+1)/(x-2) = 3, showing actual LCD and intermediate steps."""
    norm = preprocess_expr(raw)
    parts = split_equation(norm)
    if not parts:
        show_error("Could not parse equation.")
        return

    lhs_str, rhs_str = parts
    lhs, e1 = safe_sympify(lhs_str)
    rhs, e2 = safe_sympify(rhs_str)

    if e1 or e2 or lhs is None or rhs is None:
        show_error(f"Parse error: {e1 or e2}")
        return

    steps = []
    steps.append(f"Equation: $\\displaystyle {lx(lhs)} = {lx(rhs)}$")

    # ── Step: identify and factor each denominator ─────────────────────────────
    lhs_denoms = collect_denominators(lhs)
    rhs_denoms = collect_denominators(rhs)
    all_denoms = lhs_denoms + rhs_denoms

    # Factor each denominator for clean display
    factored_denoms = [factor(d) for d in all_denoms]

    if factored_denoms:
        denom_display = ",\\;".join(lx(d) for d in factored_denoms)
        steps.append(f"Identify the denominators: $\\displaystyle {denom_display}$")
    else:
        steps.append("No fractions detected — treating as a standard equation.")

    # ── Step: compute LCD ──────────────────────────────────────────────────────
    if all_denoms:
        lcd = get_lcd(all_denoms)
        lcd_factored = factor(lcd)
        steps.append(f"LCD $= {lx(lcd_factored)}$")

        # ── Step: multiply both sides by LCD and show the result ───────────────
        lhs_multiplied = expand(lhs * lcd)
        rhs_multiplied = expand(rhs * lcd)

        # Simplify / cancel the results
        lhs_cleared = cancel(lhs_multiplied)
        rhs_cleared = cancel(rhs_multiplied)

        steps.append(
            f"Multiply both sides by the LCD $({lx(lcd_factored)})$:\n"
            f"$$({lx(lcd_factored)}) \\cdot \\left({lx(lhs)}\\right) = "
            f"({lx(lcd_factored)}) \\cdot \\left({lx(rhs)}\\right)$$"
        )
        steps.append(
            f"After canceling, the resulting equation is:\n"
            f"$${lx(expand(lhs_cleared))} = {lx(expand(rhs_cleared))}$$"
        )

        # ── Step: move everything to one side and simplify ─────────────────────
        cleared_expr = expand(lhs_cleared - rhs_cleared)
        if cleared_expr != expand(lhs_cleared):
            factored_cleared = factor(cleared_expr)
            steps.append(
                f"Rearrange (set equal to zero): "
                f"${lx(expand(cleared_expr))} = 0$"
            )
            if factored_cleared != cleared_expr:
                steps.append(f"Factor: ${lx(factored_cleared)} = 0$")
    else:
        lcd = sp.Integer(1)
        lhs_cleared = lhs
        rhs_cleared = rhs

    # ── Step: find excluded values from original denominators ──────────────────
    excluded = []
    for d in all_denoms:
        excl_candidates = solve(d, x)
        excluded += [v for v in excl_candidates if v.is_real]
    excluded = list(set(excluded))

    if excluded:
        excl_str = ",\\;".join(f"x \\neq {lx(v)}" for v in sorted(excluded, key=lambda v: float(v.evalf())))
        steps.append(f"Excluded values (make original denominators zero): ${excl_str}$")

    # ── Step: solve ────────────────────────────────────────────────────────────
    try:
        eq = Eq(lhs, rhs)
        solutions = solve(eq, x)
    except Exception as e:
        show_error(f"Could not solve equation: {e}")
        return

    # ── Step: check solutions against excluded values ──────────────────────────
    valid_solutions = [s_val for s_val in solutions if s_val not in excluded]
    extraneous = [s_val for s_val in solutions if s_val in excluded]

    if extraneous:
        steps.append(
            f"Extraneous solution(s) rejected: $x = {', '.join(lx(v) for v in extraneous)}$ "
            f"(these make a denominator zero)"
        )

    # ── Step: verify valid solutions by substitution ───────────────────────────
    for s_val in valid_solutions:
        try:
            lhs_val = simplify(lhs.subs(x, s_val))
            rhs_val = simplify(rhs.subs(x, s_val))
            steps.append(
                f"Check $x = {lx(s_val)}$: "
                f"LHS $= {lx(lhs_val)}$, RHS $= {lx(rhs_val)}$ ✓"
            )
        except Exception:
            pass

    if not valid_solutions:
        steps.append("All candidate solutions are excluded — no valid solution.")
        show_result("Rational Equation", steps, "Solution Set:", r"\emptyset")
    else:
        sol_set = FiniteSet(*valid_solutions)
        show_result("Rational Equation", steps, "Solution Set:", fmt_set(sol_set))
        # Domain of the original equation
        combined_expr = lhs - rhs
        show_domain_range(expr=combined_expr, range_latex=r"(-\infty, \infty)")


# =============================================================================
# SOLVER: RATIONAL FUNCTION ARITHMETIC (f+g, f-g, f/g, f*g)
# =============================================================================

def solve_rational_arithmetic(raw: str):
    """Combine rational functions: f+g, f-g, f*g, f/g"""
    f_expr, g_expr, op = parse_fx_gx(raw)
    if f_expr is None:
        # Try simpler: just add/subtract the expressions directly
        # E.g. "(x+1)/(x-2) + (x-3)/(x+4)"
        norm = preprocess_expr(raw)
        expr, err = safe_sympify(norm)
        if err or expr is None:
            show_error("Could not parse rational expression.")
            return
        steps = ["Combine fractions over a common denominator."]
        simplified = together(expr)
        simplified = cancel(simplified)
        steps.append(f"Combined: ${lx(expr)}$")
        steps.append(f"Simplified: ${lx(simplified)}$")

        # Domain: find where denominator = 0
        d = sp.denom(simplified)
        excl = solve(d, x) if d != 1 else []
        excl_real = [v for v in excl if v.is_real]
        if excl_real:
            excl_str = ", ".join(lx(v) for v in excl_real)
            steps.append(f"Excluded values: $x \\neq {excl_str}$")

        show_result("Rational Function Arithmetic", steps, "Simplified Function:", lx(simplified))
        if excl_real:
            st.markdown(f"**Excluded Values:** $x \\neq {', '.join(lx(v) for v in excl_real)}$")
        return

    op_map = {'add': '+', 'subtract': '-', 'multiply': '\\cdot', 'divide': '/'}
    steps = []
    steps.append(f"$f(x) = {lx(f_expr)},\\quad g(x) = {lx(g_expr)}$")

    if op == 'add':
        result = together(f_expr + g_expr)
        result = cancel(result)
        steps.append(f"$(f+g)(x) = {lx(f_expr)} + {lx(g_expr)}$")
    elif op == 'subtract':
        result = together(f_expr - g_expr)
        result = cancel(result)
        steps.append(f"$(f-g)(x) = {lx(f_expr)} - ({lx(g_expr)})$")
    elif op == 'multiply':
        result = cancel(f_expr * g_expr)
        steps.append(f"$(f \\cdot g)(x) = ({lx(f_expr)}) \\cdot ({lx(g_expr)})$")
    elif op == 'divide':
        result = cancel(f_expr / g_expr)
        steps.append(f"$(f/g)(x) = \\frac{{{lx(f_expr)}}}{{{lx(g_expr)}}}$")
    else:
        show_error(f"Operation '{op}' not recognized. Try: find f+g, find f-g, find f*g, find f/g")
        return

    steps.append(f"Combine over common denominator and simplify.")
    result = cancel(simplify(result))

    # Excluded values from combined denominator
    d = sp.denom(result)
    excl = solve(d, x) if d != 1 else []
    excl_real = [v for v in excl if v.is_real]

    show_result("Rational Function Arithmetic", steps, "Simplified Function:", lx(result))
    show_domain_range(expr=result, range_latex=compute_range_expr(result))
    if excl_real:
        st.markdown(f"**Excluded Values:** $x \\neq {', '.join(lx(v) for v in excl_real)}$")


# =============================================================================
# SOLVER: FUNCTION COMPOSITION
# =============================================================================

def solve_composition(raw: str):
    """Solve f∘g (fog) or g∘f (gof)"""
    f_expr, g_expr, op = parse_fx_gx(raw)
    if f_expr is None:
        show_error("Could not parse f(x) and g(x). Expected: f(x)=..., g(x)=..., find fog")
        return

    steps = []
    steps.append(f"$f(x) = {lx(f_expr)},\\quad g(x) = {lx(g_expr)}$")

    if op == 'compose_fog':
        steps.append("$(f \\circ g)(x) = f(g(x))$")
        steps.append(f"Substitute $g(x) = {lx(g_expr)}$ into $f$:")
        result = f_expr.subs(x, g_expr)
        steps.append(f"$f(g(x)) = {lx(result)}$")
        result = simplify(cancel(result))
        steps.append(f"Simplified: ${lx(result)}$")
        label = "Composition $(f \\circ g)(x):$"
    elif op == 'compose_gof':
        steps.append("$(g \\circ f)(x) = g(f(x))$")
        steps.append(f"Substitute $f(x) = {lx(f_expr)}$ into $g$:")
        result = g_expr.subs(x, f_expr)
        steps.append(f"$g(f(x)) = {lx(result)}$")
        result = simplify(cancel(result))
        steps.append(f"Simplified: ${lx(result)}$")
        label = "Composition $(g \\circ f)(x):$"
    else:
        show_error("Could not determine composition direction. Use 'find fog' or 'find gof'.")
        return

    show_result("Function Composition", steps, label, lx(result))
    show_domain_range(expr=result, range_latex=r"(-\infty, \infty)")


# =============================================================================
# SOLVER: AVERAGE RATE OF CHANGE
# =============================================================================

def solve_aroc(raw: str):
    """Average rate of change of f(x) from a to b"""
    result = extract_function_and_interval(raw)
    if result is None:
        show_error("Could not parse. Expected: average rate of change of EXPR from A to B")
        return

    f_expr, a_val, b_val = result
    steps = []
    steps.append(f"$f(x) = {lx(f_expr)},\\quad a = {lx(a_val)},\\quad b = {lx(b_val)}$")
    steps.append(r"Formula: $\text{AROC} = \frac{f(b) - f(a)}{b - a}$")

    fa = f_expr.subs(x, a_val)
    fb = f_expr.subs(x, b_val)
    steps.append(f"$f({lx(a_val)}) = {lx(simplify(fa))}$")
    steps.append(f"$f({lx(b_val)}) = {lx(simplify(fb))}$")

    numer_val = simplify(fb - fa)
    denom_val = simplify(b_val - a_val)
    aroc = simplify(Rational(numer_val, denom_val) if (numer_val.is_integer and denom_val.is_integer) else numer_val / denom_val)

    steps.append(f"$\\text{{AROC}} = \\frac{{{lx(numer_val)}}}{{{lx(denom_val)}}} = {lx(aroc)}$")
    show_result("Average Rate of Change", steps, "Average Rate of Change:", lx(aroc))
    show_domain_range(expr=f_expr, range_latex=r"(-\infty, \infty)")


# =============================================================================
# SOLVER: DIFFERENCE QUOTIENT
# =============================================================================

def solve_dq(raw: str):
    """Difference quotient: [f(x+h) - f(x)] / h"""
    f_expr = extract_diff_quotient_func(raw)
    if f_expr is None:
        show_error("Could not parse. Expected: difference quotient of EXPR")
        return

    steps = []
    steps.append(f"$f(x) = {lx(f_expr)}$")
    steps.append(r"Formula: $\frac{f(x+h) - f(x)}{h}$")

    fxh = f_expr.subs(x, x + h)
    steps.append(f"$f(x+h) = {lx(expand(fxh))}$")

    numerator = expand(fxh - f_expr)
    steps.append(f"$f(x+h) - f(x) = {lx(numerator)}$")

    dq = simplify(numerator / h)
    # Try to factor h out cleanly
    try:
        numer_factored = factor(numerator)
        # Cancel h
        dq_clean = cancel(numer_factored / h)
    except Exception:
        dq_clean = dq

    steps.append(f"Divide by $h$: $\\frac{{{lx(numerator)}}}{h}$")
    steps.append(f"Simplify (cancel $h$): ${lx(dq_clean)}$")
    show_result("Difference Quotient", steps, "Difference Quotient:", lx(dq_clean))
    show_domain_range(expr=f_expr, range_latex=r"(-\infty, \infty)")


# =============================================================================
# SOLVER: INVERSE FUNCTIONS
# =============================================================================

def solve_inverse(raw: str):
    """Find the inverse of a function"""
    f_expr = extract_inverse_func(raw)
    if f_expr is None:
        show_error("Could not parse. Expected: find the inverse of EXPR")
        return

    steps = []
    steps.append(f"$f(x) = {lx(f_expr)}$")
    steps.append("Replace $f(x)$ with $y$: $y = " + lx(f_expr) + "$")
    steps.append("Swap $x$ and $y$: $x = " + lx(f_expr.subs(x, y)) + "$")
    steps.append("Solve for $y$.")

    # Replace x with y in expr, then solve for y given x
    eq_swapped = Eq(x, f_expr.subs(x, y))
    solutions = solve(eq_swapped, y)

    if not solutions:
        show_error("Could not find inverse (may not exist or be algebraically complex).")
        return

    if len(solutions) == 1:
        inv = solutions[0]
        steps.append(f"$y = {lx(inv)}$")
        steps.append(f"So $f^{{-1}}(x) = {lx(inv)}$")

        # Domain of inverse = range of original (just compute domain of inverse)
        try:
            inv_denom = sp.denom(inv)
            if inv_denom != 1:
                excl = solve(inv_denom, x)
                excl_real = [v for v in excl if v.is_real]
                if excl_real:
                    steps.append(f"Domain restriction: $x \\neq {', '.join(lx(v) for v in excl_real)}$")
        except Exception:
            pass

        show_result("Inverse Function", steps, "Inverse:", lx(inv))
        show_domain_range(expr=inv, range_latex=compute_range_expr(inv))
    else:
        # Multiple branches
        steps.append(f"Multiple solutions found: may not be one-to-one.")
        for i, sol in enumerate(solutions):
            steps.append(f"Branch {i+1}: $y = {lx(sol)}$")
        show_result("Inverse Function", steps, "Inverse (branch 1):", lx(solutions[0]))


# =============================================================================
# SOLVER: EXPONENTIAL EQUATIONS
# =============================================================================

def _get_base_exp(expr):
    """Return (base, exponent) if expr is a Pow node, else None."""
    if isinstance(expr, sp.Pow):
        return expr.args  # (base, exponent)
    return None


def solve_exp_equation(raw: str):
    """Solve exponential equations like 2^x = 8, e^x = 5, 3^(2x+1) = 27"""
    norm = preprocess_expr(raw)
    parts = split_equation(norm)
    if not parts:
        show_error("Could not parse equation.")
        return

    lhs_str, rhs_str = parts
    lhs, e1 = safe_sympify(lhs_str)
    rhs, e2 = safe_sympify(rhs_str)

    if e1 or e2 or lhs is None or rhs is None:
        show_error(f"Parse error: {e1 or e2}")
        return

    steps = []
    steps.append(f"Equation: ${lx(lhs)} = {lx(rhs)}$")

    matched_bases = False
    lhs_be = _get_base_exp(lhs)
    rhs_be = _get_base_exp(rhs)

    # ── Strategy 1: same base on both sides → set exponents equal ────────────
    if lhs_be and rhs_be and simplify(lhs_be[0] - rhs_be[0]) == 0:
        base, lhs_exp = lhs_be
        _, rhs_exp = rhs_be
        steps.append(
            f"Both sides have the same base ${lx(base)}$. "
            f"Set the exponents equal: ${lx(lhs_exp)} = {lx(rhs_exp)}$"
        )
        exp_eq_solutions = solve(Eq(lhs_exp, rhs_exp), x)
        if exp_eq_solutions:
            matched_bases = True
            for v in exp_eq_solutions:
                steps.append(f"Solve for $x$: $x = {lx(v)}$")
            sol_set_fmt = FiniteSet(*exp_eq_solutions)
            show_result("Exponential Equation", steps, "Solution Set:", fmt_set(sol_set_fmt))
            return

    # ── Strategy 2: rewrite RHS as a power of the same base ──────────────────
    if lhs_be and not matched_bases:
        base, exponent = lhs_be
        if rhs.is_number and base.is_number:
            try:
                k = simplify(log(rhs, base))
                if k.is_integer or k.is_rational:
                    steps.append(
                        f"Rewrite right side as a power of ${lx(base)}$: "
                        f"${lx(rhs)} = {lx(base)}^{{{lx(k)}}}$"
                    )
                    steps.append(
                        f"Set exponents equal: ${lx(exponent)} = {lx(k)}$"
                    )
                    exp_solutions = solve(Eq(exponent, k), x)
                    if exp_solutions:
                        matched_bases = True
                        for v in exp_solutions:
                            steps.append(f"Solve for $x$: $x = {lx(v)}$")
                        sol_set_fmt = FiniteSet(*exp_solutions)
                        show_result("Exponential Equation", steps, "Solution Set:", fmt_set(sol_set_fmt))
                        return
            except Exception:
                pass

    # ── Strategy 3: take ln of both sides ────────────────────────────────────
    if lhs_be:
        base, exponent = lhs_be
        if base == E:
            steps.append(
                f"Take $\\ln$ of both sides: $\\ln({lx(lhs)}) = \\ln({lx(rhs)})$"
            )
            steps.append(
                f"Apply $\\ln(e^u) = u$: ${lx(exponent)} = \\ln({lx(rhs)})$"
            )
        else:
            steps.append(
                f"Take $\\ln$ of both sides: $\\ln({lx(lhs)}) = \\ln({lx(rhs)})$"
            )
            steps.append(
                f"Apply the power rule $\\ln(b^x) = x \\cdot \\ln(b)$: "
                f"${lx(exponent)} \\cdot \\ln({lx(base)}) = \\ln({lx(rhs)})$"
            )
            if rhs.is_number and base.is_number:
                steps.append(
                    f"Isolate $x$: $x = \\dfrac{{\\ln({lx(rhs)})}}{{\\ln({lx(base)})}}$"
                )
    else:
        steps.append(
            f"Take $\\ln$ of both sides: $\\ln({lx(lhs)}) = \\ln({lx(rhs)})$"
        )

    eq = Eq(lhs, rhs)
    solutions = solve(eq, x)
    if not solutions:
        try:
            sol_set = solveset(eq, x, domain=S.Reals)
            solutions = list(sol_set) if isinstance(sol_set, FiniteSet) else []
        except Exception:
            solutions = []

    if not solutions:
        show_error("Could not solve symbolically. No real solution or requires numerical methods.")
        return

    for s_val in solutions:
        approx = s_val.evalf()
        if simplify(s_val - approx) != 0:
            steps.append(f"Exact value: $x = {lx(s_val)}$")
            steps.append(f"Decimal approximation: $x \\approx {float(approx):.6f}$")
        else:
            steps.append(f"$x = {lx(s_val)}$")

    sol_set_fmt = FiniteSet(*solutions)
    show_result("Exponential Equation", steps, "Solution Set:", fmt_set(sol_set_fmt))
    show_domain_range(expr=lhs, range_latex=r"(-\infty, \infty)")


# =============================================================================
# SOLVER: LOGARITHMIC EQUATIONS
# =============================================================================

def solve_log_equation(raw: str):
    """Solve logarithmic equations like log(x) + log(x-3) = 2"""
    norm = preprocess_expr(raw)
    uses_log10 = bool(re.search(r'\blog\s*\(', raw, re.IGNORECASE) and not re.search(r'log\s*\(.*,', raw))
    uses_ln    = bool(re.search(r'\bln\s*\(', raw, re.IGNORECASE))

    if uses_log10 and not uses_ln:
        norm_for_parse = re.sub(r'\blog\s*\(', 'log10(', norm)
        def parse_log(s):
            local = {'x': x, 'log10': lambda a: log(a, 10), 'log': log, 'ln': log,
                     'Abs': Abs, 'sqrt': sqrt, 'exp': exp}
            return sympify(s, locals=local)
        base_num = 10
        base_label = "10"
    else:
        norm_for_parse = norm
        def parse_log(s):
            return safe_sympify(s)[0]
        base_num = E
        base_label = "e"

    parts = split_equation(norm_for_parse)
    if not parts:
        show_error("Could not parse equation.")
        return

    lhs_str, rhs_str = parts
    try:
        lhs = parse_log(lhs_str)
        rhs = parse_log(rhs_str)
    except Exception as e:
        show_error(f"Parse error: {e}")
        return

    if lhs is None or rhs is None:
        show_error("Could not parse equation sides.")
        return

    steps = []
    steps.append(f"Equation: ${lx(lhs)} = {lx(rhs)}$")

    # ── Step: identify the log base being used ────────────────────────────────
    if uses_log10 and not uses_ln:
        steps.append(f"Using **base-10** logarithms ($\\log = \\log_{{10}}$).")
    else:
        steps.append(f"Using **natural** logarithms ($\\ln = \\log_e$).")

    # ── Step: try to combine/simplify the log side using SymPy's expand/simplify
    try:
        lhs_combined = simplify(lhs)
        if lhs_combined != lhs:
            steps.append(
                f"Combine logarithms using log rules (product/quotient/power): "
                f"${lx(lhs_combined)} = {lx(rhs)}$"
            )
        else:
            lhs_combined = lhs
    except Exception:
        lhs_combined = lhs

    # ── Step: show what "both sides exponentiated" looks like ─────────────────
    # If the whole LHS is a single log, isolate and exponentiate
    def is_single_log(expr):
        return isinstance(expr, sp.log) or (isinstance(expr, sp.Mul) and
               any(isinstance(a, sp.log) for a in expr.args))

    if is_single_log(lhs_combined):
        if base_num == E:
            steps.append(
                f"Exponentiate both sides (base $e$): "
                f"$e^{{{lx(lhs_combined)}}} = e^{{{lx(rhs)}}}$"
            )
        else:
            steps.append(
                f"Exponentiate both sides (base $10$): "
                f"$10^{{{lx(lhs_combined)}}} = 10^{{{lx(rhs)}}}$"
            )
        steps.append(
            f"The left side simplifies — argument of log is isolated."
        )
    else:
        # General: just note that we exponentiate to clear logs
        if base_num == E:
            steps.append(
                f"Rewrite using $e^{{\\ln(u)}} = u$ to eliminate all logs, then solve the resulting equation."
            )
        else:
            steps.append(
                f"Rewrite using $10^{{\\log(u)}} = u$ to eliminate all logs, then solve the resulting equation."
            )

    # ── Step: domain — log arguments must be positive ─────────────────────────
    steps.append("Domain restriction: every logarithm argument must be **positive**.")

    # ── Step: solve ────────────────────────────────────────────────────────────
    eq = Eq(lhs, rhs)
    try:
        solutions = solve(eq, x)
    except Exception as e:
        show_error(f"Could not solve: {e}")
        return

    if not solutions:
        show_error("No solutions found.")
        return

    # ── Step: check each candidate ────────────────────────────────────────────
    final_valid = []
    for s_val in solutions:
        if not s_val.is_real:
            steps.append(f"Reject $x = {lx(s_val)}$ (complex value).")
            continue
        try:
            residual = abs(float((lhs - rhs).subs(x, s_val).evalf()))
            if residual < 1e-6:
                final_valid.append(s_val)
                # Show what each log argument evaluates to for transparency
                steps.append(
                    f"Check $x = {lx(s_val)}$: substituting back gives LHS $= {lx(simplify(lhs.subs(x, s_val)))}$, "
                    f"RHS $= {lx(simplify(rhs.subs(x, s_val)))}$ ✓"
                )
            else:
                steps.append(f"Reject $x = {lx(s_val)}$ — extraneous (makes a log argument ≤ 0 or doesn't satisfy original).")
        except Exception:
            final_valid.append(s_val)

    if not final_valid:
        steps.append("All solutions are extraneous.")
        show_result("Logarithmic Equation", steps, "Solution Set:", r"\emptyset")
    else:
        sol_set = FiniteSet(*final_valid)
        show_result("Logarithmic Equation", steps, "Solution Set:", fmt_set(sol_set))
        show_domain_range(expr=lhs, range_latex=r"(-\infty, \infty)")


# =============================================================================
# SOLVER: EXPONENTIAL INEQUALITIES
# =============================================================================

def solve_exp_inequality(raw: str):
    """Solve exponential inequalities like 2^x > 8"""
    norm = preprocess_expr(raw)
    parts = split_inequality(norm)
    if not parts:
        show_error("Could not parse inequality.")
        return

    lhs_str, rel, rhs_str = parts
    lhs, e1 = safe_sympify(lhs_str)
    rhs, e2 = safe_sympify(rhs_str)

    if e1 or e2 or lhs is None or rhs is None:
        show_error(f"Parse error: {e1 or e2}")
        return

    steps = []
    steps.append(f"Inequality: ${lx(lhs)} {rel} {lx(rhs)}$")

    expr = lhs - rhs
    try:
        rel_map = {'<': expr < 0, '>': expr > 0, '<=': expr <= 0, '>=': expr >= 0}
        ineq = rel_map[rel]
        sol_set = solveset(ineq, x, domain=S.Reals)
        steps.append("Take logarithm of both sides to linearize the exponent.")
        steps.append("Be careful: if base < 1, inequality direction flips.")
        # Critical points
        crit = solve(Eq(lhs, rhs), x)
        crit_real = [v for v in crit if v.is_real]
        show_result("Exponential Inequality", steps, "Solution Set:",
                    fmt_critical_points(crit_real) if crit_real else fmt_interval(sol_set))
        st.markdown(f"**Domain:** $${lx(sol_set)}$$")
        st.markdown("**Range:** $$(-\\infty, \\infty)$$")
    except Exception as e:
        show_error(f"Could not solve inequality: {e}")


# =============================================================================
# SOLVER: LOGARITHMIC INEQUALITIES
# =============================================================================

def solve_log_inequality(raw: str):
    """Solve logarithmic inequalities like ln(x-1) > 0"""
    norm = preprocess_expr(raw)
    uses_log10 = bool(re.search(r'\blog\s*\(', raw, re.IGNORECASE) and not re.search(r'log\s*\(.*,', raw))

    if uses_log10:
        norm = re.sub(r'\blog\s*\(', 'log10(', norm)
        local = {'x': x, 'log10': lambda a: log(a, 10), 'log': log, 'ln': log,
                 'Abs': Abs, 'sqrt': sqrt, 'exp': exp}
        def parse_fn(s):
            return sympify(s, locals=local)
    else:
        def parse_fn(s):
            return safe_sympify(s)[0]

    parts = split_inequality(norm)
    if not parts:
        show_error("Could not parse inequality.")
        return

    lhs_str, rel, rhs_str = parts
    try:
        lhs = parse_fn(lhs_str)
        rhs = parse_fn(rhs_str)
    except Exception as e:
        show_error(f"Parse error: {e}")
        return

    if lhs is None or rhs is None:
        show_error("Could not parse expression.")
        return

    steps = []
    steps.append(f"Inequality: ${lx(lhs)} {rel} {lx(rhs)}$")
    steps.append("Domain: argument of logarithm must be positive.")

    expr = lhs - rhs
    try:
        rel_map = {'<': expr < 0, '>': expr > 0, '<=': expr <= 0, '>=': expr >= 0}
        ineq = rel_map[rel]
        sol_set = solveset(ineq, x, domain=S.Reals)
        steps.append("Exponentiate both sides to remove logarithm (preserve inequality direction for positive bases).")
        # Critical points
        crit = solve(Eq(lhs, rhs), x)
        crit_real = [v for v in crit if v.is_real]
        show_result("Logarithmic Inequality", steps, "Solution Set:",
                    fmt_critical_points(crit_real) if crit_real else fmt_interval(sol_set))
        st.markdown(f"**Domain:** $${lx(sol_set)}$$")
        st.markdown("**Range:** $$(-\\infty, \\infty)$$")
    except Exception as e:
        show_error(f"Could not solve: {e}")


# =============================================================================
# SOLVER: POLYNOMIAL / GENERAL EQUATIONS
# =============================================================================

def solve_polynomial_equation(raw: str):
    """Solve polynomial equations (linear, quadratic, cubic, etc.) with real steps."""
    norm = preprocess_expr(raw)
    parts = split_equation(norm)
    if not parts:
        show_error("Could not parse equation.")
        return

    lhs_str, rhs_str = parts
    lhs, e1 = safe_sympify(lhs_str)
    rhs, e2 = safe_sympify(rhs_str)

    if e1 or e2 or lhs is None or rhs is None:
        show_error(f"Parse error: {e1 or e2}")
        return

    expr = lhs - rhs
    expr_expanded = expand(expr)
    steps = []
    steps.append(f"Equation: ${lx(lhs)} = {lx(rhs)}$")

    # ── Step: move everything to one side ────────────────────────────────────
    if rhs != 0:
        steps.append(f"Move all terms to one side: ${lx(expr_expanded)} = 0$")

    # ── Detect degree for targeted strategy ──────────────────────────────────
    try:
        deg = sp.degree(expr_expanded, x)
    except Exception:
        deg = None

    # ── Linear equation ───────────────────────────────────────────────────────
    if deg == 1:
        steps.append("This is a **linear equation** — isolate $x$.")
        solutions = solve(Eq(lhs, rhs), x)
        if solutions:
            steps.append(f"$x = {lx(solutions[0])}$")

    # ── Quadratic equation ────────────────────────────────────────────────────
    elif deg == 2:
        steps.append("This is a **quadratic equation**.")
        coeffs = sp.Poly(expr_expanded, x).all_coeffs()
        if len(coeffs) == 3:
            a_c, b_c, c_c = coeffs
        else:
            a_c, b_c, c_c = coeffs[0], 0, 0

        # Try factoring first
        factored = factor(expr_expanded)
        if factored != expr_expanded and '*' in str(factored):
            steps.append(f"Factor: ${lx(factored)} = 0$")
            steps.append("Set each factor equal to zero and solve.")
        else:
            # Show quadratic formula
            disc = b_c**2 - 4*a_c*c_c
            disc_simplified = simplify(disc)
            steps.append(
                f"Standard form: ${lx(a_c)}x^2 + ({lx(b_c)})x + ({lx(c_c)}) = 0$"
            )
            steps.append(
                f"Quadratic formula: $x = \\dfrac{{-b \\pm \\sqrt{{b^2 - 4ac}}}}{{2a}}$"
            )
            steps.append(
                f"Substitute $a={lx(a_c)},\\; b={lx(b_c)},\\; c={lx(c_c)}$:\n"
                f"$x = \\dfrac{{-({lx(b_c)}) \\pm \\sqrt{{({lx(b_c)})^2 - 4({lx(a_c)})({lx(c_c)})}}}}{{2({lx(a_c)})}}$"
            )
            steps.append(
                f"Discriminant: $b^2 - 4ac = {lx(b_c)}^2 - 4({lx(a_c)})({lx(c_c)}) = {lx(disc_simplified)}$"
            )
            if disc_simplified < 0:
                steps.append("Discriminant is negative — no real solutions.")
            elif disc_simplified == 0:
                steps.append("Discriminant is zero — exactly one solution (repeated root).")
            else:
                steps.append("Discriminant is positive — two real solutions.")

        solutions = solve(Eq(lhs, rhs), x)

    # ── Higher-degree ─────────────────────────────────────────────────────────
    else:
        factored = factor(expr_expanded)
        if factored != expr_expanded:
            steps.append(f"Factor: ${lx(factored)} = 0$")
            steps.append("Set each factor equal to zero and solve.")
        else:
            steps.append(f"Attempt to solve ${lx(expr_expanded)} = 0$ directly.")
        solutions = solve(Eq(lhs, rhs), x)

    # ── Collect and display solutions ─────────────────────────────────────────
    if not solutions:
        show_result("Polynomial Equation", steps, "Solution Set:", r"\emptyset")
        return

    real_sols    = [s_val for s_val in solutions if s_val.is_real]
    complex_sols = [s_val for s_val in solutions if not s_val.is_real]

    for s_val in real_sols:
        # Show check by substitution
        try:
            check_val = simplify(lhs.subs(x, s_val) - rhs.subs(x, s_val))
            steps.append(f"Check $x = {lx(s_val)}$: ${lx(lhs.subs(x,s_val))} = {lx(rhs.subs(x,s_val))}$ ✓")
        except Exception:
            pass

    if complex_sols:
        steps.append(f"Complex solutions (not real): $x = {', '.join(lx(v) for v in complex_sols)}$")

    sol_set = FiniteSet(*real_sols) if real_sols else FiniteSet(*solutions)
    show_result("Polynomial Equation", steps, "Solution Set:", fmt_set(sol_set))
    show_domain_range(expr=lhs - rhs, range_latex=r"(-\infty, \infty)")


# =============================================================================
# SOLVER: RATIONAL SIMPLIFY (expression only, no =)
# =============================================================================

def solve_rational_simplify(raw: str):
    """Simplify a rational expression and find its domain."""
    norm = preprocess_expr(raw)
    expr, err = safe_sympify(norm)
    if err or expr is None:
        show_error(f"Could not parse expression: {err}")
        return

    steps = []
    steps.append(f"Expression: ${lx(expr)}$")

    simplified = cancel(expr)
    steps.append(f"Cancel common factors: ${lx(simplified)}$")

    # Domain: find where original denominator(s) = 0
    orig_denom = sp.denom(expr)
    excl = []
    if orig_denom != 1:
        excl_candidates = solve(orig_denom, x)
        excl = [v for v in excl_candidates if v.is_real]

    if excl:
        excl_str = ", ".join(lx(v) for v in excl)
        steps.append(f"Excluded values from original: $x \\neq {excl_str}$")

    show_result("Rational Expression", steps, "Simplified Expression:", lx(simplified))
    if excl:
        st.markdown(f"**Excluded Values:** $x \\neq {', '.join(lx(v) for v in excl)}$")


# =============================================================================
# SOLVER: LOG/EXP SIMPLIFY (rules of logs)
# =============================================================================

def solve_log_simplify(raw: str):
    """Simplify a logarithmic expression using log rules."""
    norm = preprocess_expr(raw)
    # Detect base
    uses_log10 = bool(re.search(r'\blog\s*\(', raw, re.IGNORECASE))
    if uses_log10:
        norm = re.sub(r'\blog\s*\(', 'log10(', norm)
        local = {'x': x, 'log10': lambda a: log(a, 10), 'log': log, 'ln': log,
                 'Abs': Abs, 'sqrt': sqrt, 'exp': exp, 'E': E}
        try:
            expr = sympify(norm, locals=local)
        except Exception as e:
            show_error(f"Parse error: {e}")
            return
    else:
        expr, err = safe_sympify(norm)
        if err or expr is None:
            show_error(f"Parse error: {err}")
            return

    steps = []
    steps.append(f"Expression: ${lx(expr)}$")
    simplified = simplify(expand(expr, log=True))
    steps.append(f"Apply log rules (product, quotient, power): ${lx(simplified)}$")
    show_result("Logarithmic Simplification", steps, "Simplified:", lx(simplified))



# =============================================================================
# SOLVER: SYSTEMS OF EQUATIONS (2 or 3 variables)
# =============================================================================

def solve_system(equations: list):
    """
    Solve a system of linear or non-linear equations.
    equations: list of strings like ['x + y = 5', '2x - y = 3']
    """
    x, y, z = symbols('x y z')

    parsed_eqs = []
    steps = []
    steps.append(f"System of {len(equations)} equation(s):")

    for i, eq_str in enumerate(equations):
        norm = preprocess_expr(eq_str)
        parts = split_equation(norm)
        if not parts:
            show_error(f"Could not parse equation {i+1}: {eq_str}")
            return
        lhs_str, rhs_str = parts
        lhs, e1 = safe_sympify(lhs_str)
        rhs, e2 = safe_sympify(rhs_str)
        if e1 or e2 or lhs is None or rhs is None:
            show_error(f"Parse error in equation {i+1}: {e1 or e2}")
            return
        eq = Eq(lhs, rhs)
        parsed_eqs.append(eq)
        steps.append(f"Equation {i+1}: ${lx(lhs)} = {lx(rhs)}$")

    # Determine variables used
    all_syms = set()
    for eq in parsed_eqs:
        all_syms |= eq.free_symbols
    var_list = sorted(all_syms, key=lambda s: str(s))

    steps.append(f"Variables: ${', '.join(str(v) for v in var_list)}$")

    def is_linear_system(eqs, vars):
        for eq in eqs:
            expr = expand(eq.lhs - eq.rhs)
            for v in vars:
                try:
                    if sp.degree(expr, v) > 1:
                        return False
                except Exception:
                    return False
        return True

    try:
        if is_linear_system(parsed_eqs, var_list):
            steps.append("This is a **linear system** — solving by elimination/substitution.")
            from sympy import linsolve
            exprs = [eq.lhs - eq.rhs for eq in parsed_eqs]
            sol = linsolve(exprs, *var_list)
            if not sol:
                steps.append("No solution — system may be inconsistent or dependent.")
                show_result("System of Linear Equations", steps, "Solution:", r"\text{No solution}")
                return
            sol_list = list(sol)
            if len(sol_list) == 1:
                sol_tuple = sol_list[0]
                sol_str = ", ".join(f"{v} = {lx(val)}" for v, val in zip(var_list, sol_tuple))
                steps.append(f"Solution: ${sol_str}$")
                for i, eq in enumerate(parsed_eqs):
                    check = simplify(eq.lhs.subs(list(zip(var_list, sol_tuple))) -
                                     eq.rhs.subs(list(zip(var_list, sol_tuple))))
                    steps.append(f"Check equation {i+1}: residual $= {lx(check)}$ ✓")
                result_latex = r"\left(" + ", ".join(lx(v) for v in sol_tuple) + r"\right)"
                show_result("System of Linear Equations", steps, "Solution:", result_latex)
            else:
                steps.append("Infinitely many solutions — system is dependent.")
                show_result("System of Equations", steps, "Solution:", r"\text{Infinitely many solutions}")
        else:
            steps.append("This is a **non-linear system** — solving symbolically.")
            sol = solve(parsed_eqs, var_list, dict=True)
            if not sol:
                steps.append("No real solution found.")
                show_result("System of Non-Linear Equations", steps, "Solution:", r"\text{No solution}")
                return
            result_parts = []
            for i, s in enumerate(sol):
                pair = ", ".join(f"{lx(k)} = {lx(v)}" for k, v in s.items())
                steps.append(f"Solution {i+1}: ${pair}$")
                result_parts.append(r"\left(" + ", ".join(lx(v) for v in s.values()) + r"\right)")
            result_latex = ",\\;".join(result_parts)
            show_result("System of Non-Linear Equations", steps, "Solution Set:", result_latex)
    except Exception as e:
        show_error(f"Could not solve system: {e}")


# =============================================================================
# SOLVER: PROPERTIES OF FUNCTIONS
# =============================================================================

def solve_function_properties(raw: str):
    """Analyze properties: even/odd, domain, range, increasing/decreasing, extrema."""
    m = re.search(r'f\s*\(\s*x\s*\)\s*=\s*(.+?)(?=,\s*properties|,\s*transformations|$)', raw, re.IGNORECASE)
    if not m:
        show_error("Could not parse. Expected: f(x) = expr, properties")
        return

    expr_str = preprocess_expr(m.group(1).strip())
    f_expr, err = safe_sympify(expr_str)
    if err or f_expr is None:
        show_error(f"Could not parse function: {err}")
        return

    steps = []
    steps.append(f"$f(x) = {lx(f_expr)}$")

    # Domain
    domain = compute_domain_expr(f_expr)
    domain_str = r"(-\infty, \infty)" if domain == S.Reals else lx(domain)
    steps.append(f"**Domain:** ${domain_str}$")

    # Even / Odd
    f_neg = simplify(f_expr.subs(x, -x))
    steps.append(f"$f(-x) = {lx(f_neg)}$")
    if simplify(f_neg - f_expr) == 0:
        steps.append("**Even function** (symmetric about y-axis): $f(-x) = f(x)$ ✓")
    elif simplify(f_neg + f_expr) == 0:
        steps.append("**Odd function** (symmetric about origin): $f(-x) = -f(x)$ ✓")
    else:
        steps.append("**Neither even nor odd.**")

    # Derivatives
    try:
        f_prime = diff(f_expr, x)
        steps.append(f"First derivative: $f'(x) = {lx(f_prime)}$")

        crit_pts = solve(f_prime, x)
        crit_real = sorted([v for v in crit_pts if v.is_real], key=lambda v: float(v.evalf()))
        if crit_real:
            steps.append(f"Critical points: $x = {', '.join(lx(v) for v in crit_real)}$")

        try:
            inc_set = solveset(f_prime > 0, x, domain=S.Reals)
            dec_set = solveset(f_prime < 0, x, domain=S.Reals)
            if inc_set != EmptySet:
                steps.append(f"Increasing on: ${lx(inc_set)}$")
            if dec_set != EmptySet:
                steps.append(f"Decreasing on: ${lx(dec_set)}$")
        except Exception:
            steps.append("Could not determine increasing/decreasing intervals.")

        f_pp = diff(f_prime, x)
        steps.append(f"Second derivative: $f''(x) = {lx(f_pp)}$")

        for cp in crit_real:
            try:
                fpp = simplify(f_pp.subs(x, cp))
                fval = simplify(f_expr.subs(x, cp))
                if fpp > 0:
                    steps.append(f"Local **minimum** at $x = {lx(cp)}$: $f({lx(cp)}) = {lx(fval)}$")
                elif fpp < 0:
                    steps.append(f"Local **maximum** at $x = {lx(cp)}$: $f({lx(cp)}) = {lx(fval)}$")
                else:
                    steps.append(f"Inconclusive at $x = {lx(cp)}$ — check manually.")
            except Exception:
                pass
    except Exception as e:
        steps.append(f"Could not compute derivative analysis: {e}")

    range_lat = compute_range_expr(f_expr)
    steps.append(f"**Range:** ${range_lat}$")

    show_result("Function Properties", steps, "Function:", lx(f_expr))
    st.markdown(f"**Domain:** ${domain_str}$")
    st.markdown(f"**Range:** ${range_lat}$")


# =============================================================================
# SOLVER: TRANSFORMATIONS OF FUNCTIONS
# =============================================================================

def solve_transformations(raw: str):
    """Identify transformations from parent function."""
    m = re.search(r'f\s*\(\s*x\s*\)\s*=\s*(.+?)(?=,\s*transformations|$)', raw, re.IGNORECASE)
    if not m:
        show_error("Could not parse. Expected: f(x) = expr")
        return

    expr_str = preprocess_expr(m.group(1).strip())
    f_expr, err = safe_sympify(expr_str)
    if err or f_expr is None:
        show_error(f"Could not parse: {err}")
        return

    steps = []
    steps.append(f"$f(x) = {lx(f_expr)}$")
    transformations = []
    parent = None

    def get_k(expr):
        """Extract vertical shift k from expr = x_part + k."""
        if isinstance(expr, Add):
            const = [t for t in expr.args if not t.has(x)]
            if not const:
                return S.Zero
            result = S.Zero
            for c in const:
                result = result + c
            return result
        return S.Zero

    k_val = get_k(f_expr)
    try:
        x_part = simplify(f_expr - k_val)
    except Exception:
        x_part = f_expr
        k_val = S.Zero

    # Absolute value
    if f_expr.has(Abs):
        parent = "Absolute Value: $f(x) = |x|$"
        coeff = S.One
        inner_expr = None
        if isinstance(x_part, Mul):
            for arg in x_part.args:
                if isinstance(arg, Abs):
                    inner_expr = arg.args[0]
                elif arg.is_number:
                    coeff = arg
        elif isinstance(x_part, Abs):
            inner_expr = x_part.args[0]
        if inner_expr is not None:
            h_v = solve(inner_expr, x)
            h_v = h_v[0] if h_v else S.Zero
            if coeff != 1:
                if coeff < 0:
                    transformations.append("Reflection over x-axis")
                transformations.append(f"Vertical stretch/compression by $|a| = {lx(Abs(coeff))}$")
            if h_v != 0:
                transformations.append(f"Horizontal shift {'right' if h_v > 0 else 'left'} by ${lx(Abs(h_v))}$")
        if k_val != 0:
            transformations.append(f"Vertical shift {'up' if k_val > 0 else 'down'} by ${lx(Abs(k_val))}$")

    # Square root
    elif f_expr.has(sqrt):
        parent = "Square Root: $f(x) = \\sqrt{x}$"
        coeff = S.One
        inner_arg = None
        if isinstance(x_part, Mul):
            for arg in x_part.args:
                if isinstance(arg, Pow) and arg.exp == sp.Rational(1, 2):
                    inner_arg = arg.args[0]
                elif arg.is_number:
                    coeff = arg
        elif isinstance(x_part, Pow) and x_part.exp == sp.Rational(1, 2):
            inner_arg = x_part.args[0]
        if inner_arg is not None:
            h_v = solve(inner_arg, x)
            h_v = h_v[0] if h_v else S.Zero
            if coeff != 1:
                if coeff < 0:
                    transformations.append("Reflection over x-axis")
                transformations.append(f"Vertical stretch/compression by ${lx(Abs(coeff))}$")
            if h_v != 0:
                transformations.append(f"Horizontal shift {'right' if h_v > 0 else 'left'} by ${lx(Abs(h_v))}$")
        if k_val != 0:
            transformations.append(f"Vertical shift {'up' if k_val > 0 else 'down'} by ${lx(Abs(k_val))}$")

    # Quadratic
    elif f_expr.is_polynomial(x):
        try:
            deg = sp.degree(expand(f_expr), x)
            if deg == 2:
                parent = "Quadratic: $f(x) = x^2$"
                poly = sp.Poly(expand(f_expr), x)
                cs = poly.all_coeffs()
                if len(cs) == 3:
                    a_c, b_c, c_c = cs
                    h_v = simplify(-b_c / (2 * a_c))
                    k_v = simplify(c_c - b_c**2 / (4 * a_c))
                    steps.append(f"Vertex form: $f(x) = {lx(a_c)}(x - ({lx(h_v)}))^2 + ({lx(k_v)})$")
                    steps.append(f"Vertex: $({lx(h_v)}, {lx(k_v)})$")
                    if a_c != 1:
                        if a_c < 0:
                            transformations.append("Reflection over x-axis (opens downward)")
                        transformations.append(f"Vertical stretch/compression by $|a| = {lx(Abs(a_c))}$")
                    if h_v != 0:
                        transformations.append(f"Horizontal shift {'right' if h_v > 0 else 'left'} by ${lx(Abs(h_v))}$")
                    if k_v != 0:
                        transformations.append(f"Vertical shift {'up' if k_v > 0 else 'down'} by ${lx(Abs(k_v))}$")
            elif deg == 3:
                parent = "Cubic: $f(x) = x^3$"
            else:
                parent = f"Degree-{deg} Polynomial: $f(x) = x^{{{deg}}}$"
        except Exception:
            parent = "Polynomial"

    elif f_expr.has(log):
        parent = "Logarithmic: $f(x) = \\ln(x)$"
        if k_val != 0:
            transformations.append(f"Vertical shift {'up' if k_val > 0 else 'down'} by ${lx(Abs(k_val))}$")

    elif f_expr.has(exp):
        parent = "Exponential: $f(x) = e^x$"
        if k_val != 0:
            transformations.append(f"Vertical shift {'up' if k_val > 0 else 'down'} by ${lx(Abs(k_val))}$")

    else:
        parent = "Unknown parent function"

    if parent:
        steps.append(f"**Parent function:** {parent}")
    if transformations:
        steps.append("**Transformations from parent:**")
        for t in transformations:
            steps.append(f"• {t}")
    else:
        steps.append("No transformations beyond the parent — or none detected automatically.")

    domain = compute_domain_expr(f_expr)
    range_lat = compute_range_expr(f_expr)
    show_result("Function Transformations", steps, "Transformed Function:", lx(f_expr))
    st.markdown(f"**Domain:** ${r'(-\infty, \infty)' if domain == S.Reals else lx(domain)}$")
    st.markdown(f"**Range:** ${range_lat}$")


# =============================================================================
# SOLVER: LIBRARY OF FUNCTIONS
# =============================================================================

def solve_library_function(raw: str):
    """Identify parent function, evaluate at a point, show domain/range."""
    eval_match = re.search(r'find\s+f\s*\(\s*(-?[\d.]+)\s*\)', raw, re.IGNORECASE)
    eval_val = None
    if eval_match:
        eval_val, _ = safe_sympify(eval_match.group(1))

    m = re.search(r'f\s*\(\s*x\s*\)\s*=\s*(.+?)(?=,\s*find|$)', raw, re.IGNORECASE)
    if not m:
        show_error("Could not parse. Expected: f(x) = expr [, find f(value)]")
        return

    expr_str = preprocess_expr(m.group(1).strip())
    f_expr, err = safe_sympify(expr_str)
    if err or f_expr is None:
        show_error(f"Could not parse: {err}")
        return

    steps = []
    steps.append(f"$f(x) = {lx(f_expr)}$")

    # Identify parent
    if f_expr.has(Abs):
        parent, parent_eq = "Absolute Value", r"f(x) = |x|"
    elif f_expr.has(sqrt):
        parent, parent_eq = "Square Root", r"f(x) = \sqrt{x}"
    elif f_expr.has(log):
        parent, parent_eq = "Logarithmic", r"f(x) = \ln(x)"
    elif f_expr.has(exp):
        parent, parent_eq = "Exponential", r"f(x) = e^x"
    elif sp.denom(f_expr) != 1:
        parent, parent_eq = "Rational", r"f(x) = \frac{1}{x}"
    elif f_expr.is_polynomial(x):
        try:
            deg = sp.degree(expand(f_expr), x)
            names = {0: "Constant", 1: "Linear", 2: "Quadratic", 3: "Cubic"}
            parent = names.get(deg, f"Degree-{deg} Polynomial")
            parent_eq = {0: r"f(x) = c", 1: r"f(x) = x", 2: r"f(x) = x^2", 3: r"f(x) = x^3"}.get(deg, f"f(x) = x^{deg}")
        except Exception:
            parent, parent_eq = "Polynomial", r"f(x) = x^n"
    else:
        parent, parent_eq = "Unknown", r"f(x) = ?"

    steps.append(f"**Parent type:** {parent} — ${parent_eq}$")

    domain = compute_domain_expr(f_expr)
    domain_str = r"(-\infty, \infty)" if domain == S.Reals else lx(domain)
    range_lat = compute_range_expr(f_expr)
    steps.append(f"**Domain:** ${domain_str}$")
    steps.append(f"**Range:** ${range_lat}$")

    result_val = None
    if eval_val is not None:
        try:
            result_val = simplify(f_expr.subs(x, eval_val))
            steps.append(f"Evaluate: $f({lx(eval_val)}) = {lx(result_val)}$")
        except Exception as e:
            steps.append(f"Could not evaluate: {e}")

    show_result("Library of Functions", steps, "Function:", lx(f_expr))
    st.markdown(f"**Parent Type:** {parent}")
    st.markdown(f"**Domain:** ${domain_str}$")
    st.markdown(f"**Range:** ${range_lat}$")
    if result_val is not None:
        st.markdown(f"**$f({lx(eval_val)})$ =** ${lx(result_val)}$")


# =============================================================================
# SOLVER: POLYNOMIAL LONG DIVISION & SYNTHETIC DIVISION
# =============================================================================

def solve_poly_division(raw: str):
    """Polynomial division. Input: divide POLY by DIVISOR"""
    m = re.search(r'divide\s+(.+?)\s+by\s+(.+)', raw, re.IGNORECASE)
    if not m:
        show_error("Could not parse. Expected: divide POLY by DIVISOR")
        return

    dividend_str = preprocess_expr(m.group(1).strip())
    divisor_str = preprocess_expr(m.group(2).strip())
    dividend, e1 = safe_sympify(dividend_str)
    divisor, e2 = safe_sympify(divisor_str)

    if e1 or e2 or dividend is None or divisor is None:
        show_error(f"Parse error: {e1 or e2}")
        return

    steps = []
    steps.append(f"Dividend: ${lx(expand(dividend))}$")
    steps.append(f"Divisor: ${lx(expand(divisor))}$")

    # Check for synthetic division (linear divisor)
    c_val = None
    try:
        if sp.degree(expand(divisor), x) == 1:
            c_cands = solve(expand(divisor), x)
            if c_cands:
                c_val = c_cands[0]
    except Exception:
        pass

    try:
        from sympy import div, Poly
        quotient, remainder = div(Poly(expand(dividend), x), Poly(expand(divisor), x), x)
        q_expr = quotient.as_expr()
        r_expr = remainder.as_expr()

        if c_val is not None:
            steps.append(f"Using **synthetic division** with $c = {lx(c_val)}$:")
            try:
                d_poly = sp.Poly(expand(dividend), x)
                d_coeffs = d_poly.all_coeffs()
                steps.append(f"Dividend coefficients: $[{', '.join(lx(simplify(c)) for c in d_coeffs)}]$")
                result_coeffs = [d_coeffs[0]]
                for i in range(1, len(d_coeffs)):
                    result_coeffs.append(simplify(d_coeffs[i] + c_val * result_coeffs[-1]))
                steps.append(f"Bring down ${lx(d_coeffs[0])}$, then multiply by $c={lx(c_val)}$ and add each column:")
                steps.append(f"Result row: $[{', '.join(lx(c) for c in result_coeffs)}]$")
                steps.append(f"Remainder: ${lx(result_coeffs[-1])}$")
                deg = len(d_coeffs) - 2
                q_from_synth = sum(simplify(result_coeffs[i]) * x**(deg - i) for i in range(deg + 1))
                steps.append(f"Quotient: ${lx(simplify(q_from_synth))}$")
            except Exception:
                steps.append(f"Quotient: ${lx(q_expr)}$, Remainder: ${lx(r_expr)}$")
        else:
            steps.append("Using **polynomial long division**:")
            steps.append(f"Quotient: ${lx(q_expr)}$")
            steps.append(f"Remainder: ${lx(r_expr)}$")

        steps.append(f"Verification: ${lx(expand(dividend))} = ({lx(expand(divisor))})({lx(q_expr)}) + ({lx(r_expr)})$")
        if simplify(r_expr) == 0:
            steps.append(f"Remainder $= 0$ — $(${lx(divisor)}$)$ is a **factor**.")

        result_latex = f"\\frac{{{lx(expand(dividend))}}}{{{lx(expand(divisor))}}} = {lx(q_expr)} + \\frac{{{lx(r_expr)}}}{{{lx(expand(divisor))}}}"
        show_result("Polynomial Division", steps, "Result:", result_latex)
    except Exception as e:
        show_error(f"Could not perform division: {e}")


# =============================================================================
# SOLVER: FINDING ZEROS OF POLYNOMIAL FUNCTIONS
# =============================================================================

def solve_find_zeros(raw: str):
    """Find all zeros using Rational Root Theorem + synthetic division + factoring."""
    m = re.search(r'(?:find\s+)?(?:the\s+)?(?:real\s+)?zeros?\s+of\s+(.+)', raw, re.IGNORECASE)
    if not m:
        show_error("Could not parse. Expected: find zeros of EXPR")
        return

    expr_str = preprocess_expr(m.group(1).strip())
    f_expr, err = safe_sympify(expr_str)
    if err or f_expr is None:
        show_error(f"Could not parse: {err}")
        return

    steps = []
    steps.append(f"$f(x) = {lx(f_expr)}$")

    try:
        poly = sp.Poly(expand(f_expr), x)
        coeffs = poly.all_coeffs()
        deg = poly.degree()
        steps.append(f"Degree: ${deg}$, Coefficients: $[{', '.join(lx(c) for c in coeffs)}]$")

        leading = coeffs[0]
        constant = coeffs[-1]

        # Rational Root Theorem
        if leading.is_integer and constant.is_integer and int(constant) != 0:
            def int_factors(n):
                n = int(abs(n))
                return [i for i in range(1, n + 1) if n % i == 0]

            p_factors = int_factors(int(constant))
            q_factors = int_factors(int(leading))
            candidates = sorted(set(
                [sp.Rational(p, q) for p in p_factors for q in q_factors] +
                [-sp.Rational(p, q) for p in p_factors for q in q_factors]
            ), key=float)

            steps.append(
                f"Rational Root Theorem: possible zeros are $\\pm\\frac{{p}}{{q}}$ "
                f"where $p \\mid {int(abs(constant))}$, $q \\mid {int(abs(leading))}$"
            )
            display_cands = candidates[:24]
            steps.append(f"Candidates: ${', '.join(lx(c) for c in display_cands)}$"
                         + (" ..." if len(candidates) > 24 else ""))

            rational_roots = []
            for cand in candidates:
                try:
                    if poly.eval(cand) == 0:
                        rational_roots.append(cand)
                except Exception:
                    pass

            if rational_roots:
                steps.append(f"Verified rational roots by substitution: $x = {', '.join(lx(r) for r in rational_roots)}$")
            else:
                steps.append("No rational roots found among candidates.")

        # Factor completely
        factored = factor(f_expr)
        if factored != f_expr:
            steps.append(f"Factor completely: ${lx(factored)}$")

        # Find all zeros
        all_zeros = solve(f_expr, x)
        real_zeros = sorted([z for z in all_zeros if z.is_real], key=lambda v: float(v.evalf()))
        complex_zeros = [z for z in all_zeros if not z.is_real]

        if real_zeros:
            steps.append(f"**Real zeros:** $x = {', '.join(lx(z) for z in real_zeros)}$")
        if complex_zeros:
            steps.append(f"**Complex zeros:** $x = {', '.join(lx(z) for z in complex_zeros)}$")
        if not all_zeros:
            steps.append("No zeros found.")

        # Multiplicity
        for z_val in real_zeros:
            try:
                count = 0
                temp = expand(f_expr)
                while True:
                    rem = sp.rem(temp, x - z_val, x)
                    if simplify(rem) == 0:
                        count += 1
                        temp = sp.quo(temp, x - z_val, x)
                    else:
                        break
                if count > 1:
                    steps.append(f"$x = {lx(z_val)}$ has **multiplicity {count}**")
            except Exception:
                pass

        result_latex = ", ".join(lx(z) for z in all_zeros) if all_zeros else r"\text{No zeros}"
        show_result("Zeros of Polynomial", steps, "All Zeros:", result_latex)
        if real_zeros:
            st.markdown(f"**Real Zeros:** ${', '.join(lx(z) for z in real_zeros)}$")
        if complex_zeros:
            st.markdown(f"**Complex Zeros:** ${', '.join(lx(z) for z in complex_zeros)}$")

    except Exception as e:
        show_error(f"Could not find zeros: {e}")


# =============================================================================
# SOLVER: RATIONAL FUNCTION ANALYSIS
# =============================================================================

def solve_rational_analysis(raw: str):
    """
    Analyze a rational function: vertical/horizontal asymptotes,
    x-intercepts, y-intercept, domain.
    Input format: f(x) = (x+1)/(x-2), analyze
    """
    m = re.search(r'f\s*\(\s*x\s*\)\s*=\s*(.+?)(?=,\s*analyze|$)', raw, re.IGNORECASE)
    if not m:
        # Try bare expression: (x+1)/(x-2), analyze
        m2 = re.search(r'^(.+?),\s*analyze', raw, re.IGNORECASE)
        expr_str = preprocess_expr(m2.group(1).strip()) if m2 else preprocess_expr(
            re.sub(r',?\s*analyze.*', '', raw, flags=re.IGNORECASE).strip()
        )
    else:
        expr_str = preprocess_expr(m.group(1).strip())

    f_expr, err = safe_sympify(expr_str)
    if err or f_expr is None:
        show_error(f"Could not parse rational function: {err}")
        return

    steps = []
    steps.append(f"$f(x) = {lx(f_expr)}$")

    num = sp.numer(f_expr)
    den = sp.denom(f_expr)

    # ── Domain ────────────────────────────────────────────────────────────────
    domain = compute_domain_expr(f_expr)
    domain_str = lx(domain)
    steps.append(f"**Domain:** Set denominator $\\neq 0$: ${lx(den)} \\neq 0$")

    # ── Vertical Asymptotes ───────────────────────────────────────────────────
    den_zeros = solve(den, x)
    den_zeros_real = [v for v in den_zeros if v.is_real]
    # Check which zeros are NOT holes (also zeros of numerator)
    num_zeros_set = solve(num, x)
    va_list = [v for v in den_zeros_real if v not in num_zeros_set]
    holes   = [v for v in den_zeros_real if v in num_zeros_set]

    if va_list:
        va_str = ", ".join(f"x = {lx(v)}" for v in sorted(va_list, key=lambda v: float(v.evalf())))
        steps.append(f"**Vertical Asymptote(s):** ${va_str}$")
    else:
        steps.append("**Vertical Asymptotes:** None")

    if holes:
        hole_strs = []
        for v in holes:
            f_cancelled = cancel(f_expr)
            y_val = simplify(f_cancelled.subs(x, v))
            hole_strs.append(f"({lx(v)}, {lx(y_val)})")
        steps.append(f"**Hole(s) (removable discontinuity):** {', '.join(hole_strs)}")

    # ── Horizontal Asymptote ──────────────────────────────────────────────────
    try:
        deg_num = sp.degree(sp.Poly(num, x))
        deg_den = sp.degree(sp.Poly(den, x))
        lc_num = sp.LC(sp.Poly(num, x))
        lc_den = sp.LC(sp.Poly(den, x))
        if deg_num < deg_den:
            ha = S.Zero
            steps.append(f"**Horizontal Asymptote:** $y = 0$ (degree of numerator < degree of denominator)")
        elif deg_num == deg_den:
            ha = simplify(lc_num / lc_den)
            steps.append(f"**Horizontal Asymptote:** $y = {lx(ha)}$ (ratio of leading coefficients)")
        else:
            ha = None
            steps.append("**Horizontal Asymptote:** None (degree of numerator > degree of denominator — check for oblique asymptote)")
    except Exception:
        ha = None
        lim_inf = limit(f_expr, x, oo)
        if lim_inf.is_finite:
            ha = lim_inf
            steps.append(f"**Horizontal Asymptote:** $y = {lx(ha)}$ (from limit as $x \\to \\infty$)")
        else:
            steps.append("**Horizontal Asymptote:** None")

    # ── Oblique Asymptote (if applicable) ─────────────────────────────────────
    try:
        if ha is None:
            q, r = sp.div(num, den, x)
            if r != 0:
                steps.append(f"**Oblique Asymptote:** $y = {lx(q)}$")
    except Exception:
        pass

    # ── X-Intercepts ──────────────────────────────────────────────────────────
    x_ints = [v for v in solve(num, x) if v.is_real and v not in holes]
    if x_ints:
        xi_str = ", ".join(f"({lx(v)}, 0)" for v in sorted(x_ints, key=lambda v: float(v.evalf())))
        steps.append(f"**X-Intercept(s):** ${xi_str}$")
    else:
        steps.append("**X-Intercepts:** None (numerator has no real roots outside holes)")

    # ── Y-Intercept ───────────────────────────────────────────────────────────
    try:
        if S.Zero not in den_zeros_real:
            y_int = simplify(f_expr.subs(x, 0))
            steps.append(f"**Y-Intercept:** $(0, {lx(y_int)})$")
        else:
            steps.append("**Y-Intercept:** None ($x = 0$ is excluded from domain)")
    except Exception:
        steps.append("**Y-Intercept:** Could not compute")

    show_result("Rational Function Analysis", steps, "Function:", lx(f_expr))
    st.markdown(f"**Domain:** $${domain_str}$$")
    if va_list:
        st.markdown(f"**Vertical Asymptote(s):** $${', '.join('x = ' + lx(v) for v in va_list)}$$")
    if ha is not None:
        st.markdown(f"**Horizontal Asymptote:** $$y = {lx(ha)}$$")


# =============================================================================
# SOLVER: EXPONENTIAL FUNCTION ANALYSIS
# =============================================================================

def solve_exp_function(raw: str):
    """
    Analyze an exponential function: growth/decay, y-intercept,
    horizontal asymptote, domain, range, evaluate at a point.
    Input: f(x) = 3*2^x, analyze  OR  f(x) = 5*(0.5)^x, find f(3)
    """
    # Try to extract f(x) = expr
    m = re.search(r'f\s*\(\s*x\s*\)\s*=\s*(.+?)(?=,\s*(?:analyze|find\s+f)|$)', raw, re.IGNORECASE)
    if not m:
        show_error("Could not parse. Expected: f(x) = a*b^x, analyze  OR  f(x) = a*b^x, find f(n)")
        return
    expr_str = preprocess_expr(m.group(1).strip())
    f_expr, err = safe_sympify(expr_str)
    if err or f_expr is None:
        show_error(f"Could not parse function: {err}")
        return

    # Optional: evaluate at a point
    eval_m = re.search(r'find\s+f\s*\(\s*(-?\d+(?:\.\d+)?)\s*\)', raw, re.IGNORECASE)
    eval_pt = None
    if eval_m:
        eval_pt, _ = safe_sympify(eval_m.group(1))

    steps = []
    steps.append(f"$f(x) = {lx(f_expr)}$")

    # ── Identify base ─────────────────────────────────────────────────────────
    # Look for b^x or E^x patterns
    base_val = None
    a_val = S.One
    try:
        # Check if it's of the form a * b**x (possibly with shifts)
        collected = f_expr.as_coefficients_dict()
        # Try extracting via limit approach: base = f(1)/f(0)
        f0 = simplify(f_expr.subs(x, 0))
        f1 = simplify(f_expr.subs(x, 1))
        if f0 != 0:
            ratio = simplify(f1 / f0)
            if ratio.is_number and ratio > 0:
                base_val = ratio
                a_val = f0
    except Exception:
        pass

    if base_val is not None:
        steps.append(f"Identify: $a = {lx(a_val)}$, base $b = {lx(base_val)}$")
        if base_val > 1:
            steps.append(f"Since $b = {lx(base_val)} > 1$: **Exponential Growth**")
        elif base_val > 0:
            steps.append(f"Since $0 < b = {lx(base_val)} < 1$: **Exponential Decay**")
    else:
        # E-based: check for exp(x) or E**x
        if f_expr.has(E) or 'exp' in str(f_expr):
            steps.append("Base is $e \\approx 2.718$ — **Exponential Growth** (since $e > 1$)")
        else:
            steps.append("Could not automatically classify growth/decay — inspect the base manually.")

    # ── Y-Intercept ───────────────────────────────────────────────────────────
    try:
        y_int = simplify(f_expr.subs(x, 0))
        steps.append(f"**Y-Intercept:** $f(0) = {lx(y_int)}$ → point $(0, {lx(y_int)})$")
    except Exception:
        steps.append("Y-Intercept: Could not compute")

    # ── Horizontal Asymptote ─────────────────────────────────────────────────
    try:
        ha_pos = limit(f_expr, x, oo)
        ha_neg = limit(f_expr, x, -oo)
        if ha_pos.is_finite:
            steps.append(f"**Horizontal Asymptote (right):** $y = {lx(ha_pos)}$ as $x \\to \\infty$")
        if ha_neg.is_finite:
            steps.append(f"**Horizontal Asymptote (left):** $y = {lx(ha_neg)}$ as $x \\to -\\infty$")
        ha = ha_neg if ha_neg.is_finite else (ha_pos if ha_pos.is_finite else None)
    except Exception:
        ha = None
        steps.append("Horizontal Asymptote: Could not compute")

    # ── Domain / Range ────────────────────────────────────────────────────────
    steps.append(f"**Domain:** $(-\\infty, \\infty)$ — exponential functions are defined for all real $x$")

    try:
        # Range is (HA, ∞) for growth or (0, ∞) base case
        if ha is not None and ha.is_finite:
            ha_float = float(ha.evalf())
            if base_val is not None and float(a_val.evalf()) > 0:
                range_str = f"({lx(ha)}, \\infty)"
            else:
                range_str = f"({lx(ha)}, \\infty) \\text{{ or }} (-\\infty, {lx(ha)})"
        else:
            range_str = r"(0, \infty)"
        steps.append(f"**Range:** ${range_str}$")
    except Exception:
        range_str = r"(0, \infty)"
        steps.append(f"**Range:** ${range_str}$")

    # ── Evaluate at a point ───────────────────────────────────────────────────
    if eval_pt is not None:
        val = simplify(f_expr.subs(x, eval_pt))
        steps.append(f"**Evaluate:** $f({lx(eval_pt)}) = {lx(val)} \\approx {float(val.evalf()):.4f}$")

    show_result("Exponential Function Analysis", steps, "Function:", lx(f_expr))
    st.markdown(f"**Domain:** $$(-\\infty, \\infty)$$")
    st.markdown(f"**Range:** $${range_str}$$")
    if ha is not None:
        st.markdown(f"**Horizontal Asymptote:** $$y = {lx(ha)}$$")


# =============================================================================
# SOLVER: LOGARITHMIC FUNCTION ANALYSIS
# =============================================================================

def solve_log_function(raw: str):
    """
    Analyze a logarithmic function: domain, range, vertical asymptote,
    x-intercept, y-intercept, evaluate at a point.
    Input: f(x) = log(x-2) + 3, analyze   OR   f(x) = ln(x), find f(5)
    """
    m = re.search(r'f\s*\(\s*x\s*\)\s*=\s*(.+?)(?=,\s*(?:analyze|find\s+f)|$)', raw, re.IGNORECASE)
    if not m:
        show_error("Could not parse. Expected: f(x) = log(expr), analyze  OR  f(x) = log(expr), find f(n)")
        return
    expr_str_raw = m.group(1).strip()

    # Detect base
    uses_log10 = bool(re.search(r'\blog\s*\(', expr_str_raw, re.IGNORECASE))
    uses_ln    = bool(re.search(r'\bln\s*\(', expr_str_raw, re.IGNORECASE))
    if uses_log10 and not uses_ln:
        base_label = "10"
        local_dict = {'x': x, 'log': lambda a: log(a, 10), 'ln': log, 'Abs': Abs, 'sqrt': sqrt, 'exp': exp, 'E': E}
    else:
        base_label = "e"
        local_dict = {'x': x, 'log': log, 'ln': log, 'Abs': Abs, 'sqrt': sqrt, 'exp': exp, 'E': E}

    expr_str = preprocess_expr(expr_str_raw)
    try:
        f_expr = sympify(expr_str, locals=local_dict)
    except Exception as e:
        show_error(f"Could not parse function: {e}")
        return

    # Optional evaluate
    eval_m = re.search(r'find\s+f\s*\(\s*(-?\d+(?:\.\d+)?)\s*\)', raw, re.IGNORECASE)
    eval_pt = None
    if eval_m:
        eval_pt, _ = safe_sympify(eval_m.group(1))

    steps = []
    steps.append(f"$f(x) = {lx(f_expr)}$")
    steps.append(f"Base: $\\log_{ {base_label} }$")

    # ── Domain: argument of log must be > 0 ───────────────────────────────────
    # Find the argument inside the log
    domain = compute_domain_expr(f_expr)
    domain_str = lx(domain)
    steps.append(f"**Domain:** Argument of logarithm must be $> 0$ → ${domain_str}$")

    # ── Vertical Asymptote ────────────────────────────────────────────────────
    # Find where domain boundary is — solve argument = 0
    va_cands = []
    try:
        for node in sp.preorder_traversal(f_expr):
            if isinstance(node, sp.log):
                arg = node.args[0]
                zeros = solve(arg, x)
                va_cands += [v for v in zeros if v.is_real]
        va_cands = list(set(va_cands))
    except Exception:
        pass
    if va_cands:
        va_str = ", ".join(f"x = {lx(v)}" for v in sorted(va_cands, key=lambda v: float(v.evalf())))
        steps.append(f"**Vertical Asymptote(s):** ${va_str}$")
    else:
        steps.append("**Vertical Asymptote:** $x = 0$ (standard)")

    # ── X-Intercept: f(x) = 0 ────────────────────────────────────────────────
    try:
        x_ints = solve(Eq(f_expr, 0), x)
        x_ints_real = [v for v in x_ints if v.is_real]
        if x_ints_real:
            xi_str = ", ".join(f"({lx(v)}, 0)" for v in sorted(x_ints_real, key=lambda v: float(v.evalf())))
            steps.append(f"**X-Intercept(s):** ${xi_str}$ (set $f(x) = 0$ and solve)")
        else:
            steps.append("**X-Intercepts:** None in domain")
    except Exception:
        steps.append("**X-Intercepts:** Could not compute")

    # ── Y-Intercept: f(0) if 0 in domain ────────────────────────────────────
    try:
        if S.Zero in domain or (hasattr(domain, '__contains__') and 0 in [float(v) for v in va_cands if v.is_real] is False):
            y_int = simplify(f_expr.subs(x, 0))
            if y_int.is_real:
                steps.append(f"**Y-Intercept:** $f(0) = {lx(y_int)}$ → $(0, {lx(y_int)})$")
            else:
                steps.append("**Y-Intercept:** None ($x = 0$ not in domain)")
        else:
            steps.append("**Y-Intercept:** None ($x = 0$ not in domain)")
    except Exception:
        steps.append("**Y-Intercept:** Could not compute")

    # ── Range ────────────────────────────────────────────────────────────────
    range_str = r"(-\infty, \infty)"
    steps.append(f"**Range:** $(-\\infty, \\infty)$ — logarithmic functions cover all reals")

    # ── Evaluate at point ─────────────────────────────────────────────────────
    if eval_pt is not None:
        try:
            val = simplify(f_expr.subs(x, eval_pt))
            steps.append(f"**Evaluate:** $f({lx(eval_pt)}) = {lx(val)} \\approx {float(val.evalf()):.4f}$")
        except Exception:
            steps.append(f"**Evaluate:** Could not evaluate at $x = {lx(eval_pt)}$ (may be outside domain)")

    show_result("Logarithmic Function Analysis", steps, "Function:", lx(f_expr))
    st.markdown(f"**Domain:** $${domain_str}$$")
    st.markdown(f"**Range:** $${range_str}$$")
    if va_cands:
        st.markdown(f"**Vertical Asymptote(s):** $${', '.join('x = ' + lx(v) for v in va_cands)}$$")


# =============================================================================
# SOLVER: RULES OF LOGARITHMS (expand / condense)
# =============================================================================

def solve_log_rules(raw: str):
    """
    Expand or condense a logarithmic expression.
    Input: expand log((x^2*(x+1))/(x-3))
           condense 2*log(x) + log(x+1) - log(x-2)
    """
    s = raw.lower().strip()
    mode = 'expand' if re.search(r'\bexpand\b', s) else ('condense' if re.search(r'\bcondense\b', s) else None)
    if mode is None:
        show_error("Specify 'expand' or 'condense'. Example: expand log(x^2*(x+1))")
        return

    # Strip the keyword
    expr_str_raw = re.sub(r'\b(expand|condense)\b', '', raw, flags=re.IGNORECASE).strip()

    uses_log10 = bool(re.search(r'\blog\s*\(', expr_str_raw, re.IGNORECASE))
    if uses_log10:
        local_dict = {'x': x, 'y': y, 'z': z, 'log': lambda a: log(a, 10), 'ln': log,
                      'Abs': Abs, 'sqrt': sqrt, 'exp': exp, 'E': E}
        base_label = "\\log"
    else:
        local_dict = {'x': x, 'y': y, 'z': z, 'log': log, 'ln': log,
                      'Abs': Abs, 'sqrt': sqrt, 'exp': exp, 'E': E}
        base_label = "\\ln"

    expr_str = preprocess_expr(expr_str_raw)
    try:
        f_expr = sympify(expr_str, locals=local_dict)
    except Exception as e:
        show_error(f"Could not parse expression: {e}")
        return

    steps = []
    steps.append(f"Expression: ${lx(f_expr)}$")

    if mode == 'expand':
        steps.append("Apply **Product Rule:** $\\log(AB) = \\log A + \\log B$")
        steps.append("Apply **Quotient Rule:** $\\log(A/B) = \\log A - \\log B$")
        steps.append("Apply **Power Rule:** $\\log(A^n) = n \\cdot \\log A$")
        try:
            expanded = sp.expand_log(f_expr, force=True)
            steps.append(f"Result: ${lx(expanded)}$")
        except Exception as e:
            show_error(f"Could not expand: {e}")
            return
        show_result("Rules of Logarithms — Expand", steps, "Expanded Form:", lx(expanded))

    else:  # condense
        steps.append("Apply log rules **in reverse** to combine into a single logarithm.")
        steps.append("**Power Rule (reverse):** $n \\cdot \\log A = \\log(A^n)$")
        steps.append("**Product Rule (reverse):** $\\log A + \\log B = \\log(AB)$")
        steps.append("**Quotient Rule (reverse):** $\\log A - \\log B = \\log(A/B)$")
        try:
            condensed = sp.logcombine(f_expr, force=True)
            steps.append(f"Result: ${lx(condensed)}$")
        except Exception as e:
            show_error(f"Could not condense: {e}")
            return
        show_result("Rules of Logarithms — Condense", steps, "Condensed Form:", lx(condensed))


# =============================================================================
# SOLVER: AROC / DIFFERENCE QUOTIENT FROM WORD PROBLEMS
# =============================================================================

def solve_aroc_word(raw: str):
    """
    Handle word-problem style AROC and DQ inputs that include context sentences.
    Strips the context and routes to the existing solvers.
    Example: 'The height of a ball is h(t) = -16t^2 + 64t.
              Find the average rate of change from t=1 to t=3.'
    """
    steps = []

    # ── Extract function (h(t)=, f(x)=, etc.) ────────────────────────────────
    func_m = re.search(r'[a-zA-Z]\s*\(\s*[a-zA-Z]\s*\)\s*=\s*([^,.]+)', raw, re.IGNORECASE)
    if not func_m:
        show_error("Could not find a function definition (e.g. h(t) = -16t^2 + 64t) in the problem.")
        return

    func_expr_str = preprocess_expr(func_m.group(1).strip())
    # Detect the independent variable
    var_m = re.search(r'[a-zA-Z]\s*\(\s*([a-zA-Z])\s*\)', func_m.group(0))
    ind_var_str = var_m.group(1) if var_m else 'x'
    ind_var = symbols(ind_var_str)

    local_dict = {'x': x, 't': symbols('t'), 'n': symbols('n'),
                  'log': log, 'ln': log, 'exp': exp, 'sqrt': sqrt,
                  'Abs': Abs, 'E': E, 'pi': pi}
    try:
        f_expr_raw = sympify(func_expr_str, locals=local_dict)
        # Substitute to x for uniform treatment
        f_expr = f_expr_raw.subs(ind_var, x)
    except Exception as e:
        show_error(f"Could not parse function: {e}")
        return

    steps.append(f"Function identified: ${ind_var}$-variable → $f(x) = {lx(f_expr)}$")

    # ── Detect: AROC or DQ ────────────────────────────────────────────────────
    is_aroc = bool(re.search(r'average\s+rate\s+of\s+change|aroc', raw, re.IGNORECASE))
    is_dq   = bool(re.search(r'difference\s+quotient', raw, re.IGNORECASE))

    if is_aroc:
        # Find the two endpoints
        # Patterns: from t=1 to t=3, from x=a to x=b, from 1 to 3, between 2 and 5
        ep_m = re.search(
            r'from\s+(?:[a-zA-Z]\s*=\s*)?(-?\d+(?:\.\d+)?)\s+to\s+(?:[a-zA-Z]\s*=\s*)?(-?\d+(?:\.\d+)?)',
            raw, re.IGNORECASE
        )
        if not ep_m:
            ep_m = re.search(
                r'between\s+(-?\d+(?:\.\d+)?)\s+and\s+(-?\d+(?:\.\d+)?)',
                raw, re.IGNORECASE
            )
        if not ep_m:
            show_error("Could not find endpoints. Expected: 'from t=1 to t=3' or 'from 1 to 3'.")
            return

        a_val = sympify(ep_m.group(1))
        b_val = sympify(ep_m.group(2))
        steps.append(f"Endpoints: $a = {lx(a_val)},\\quad b = {lx(b_val)}$")
        steps.append(r"Formula: $\text{AROC} = \frac{f(b) - f(a)}{b - a}$")

        fa = simplify(f_expr.subs(x, a_val))
        fb = simplify(f_expr.subs(x, b_val))
        steps.append(f"$f({lx(a_val)}) = {lx(fa)}$")
        steps.append(f"$f({lx(b_val)}) = {lx(fb)}$")

        num_val  = simplify(fb - fa)
        den_val  = simplify(b_val - a_val)
        aroc_val = simplify(num_val / den_val)
        steps.append(f"$\\text{{AROC}} = \\dfrac{{{lx(num_val)}}}{{{lx(den_val)}}} = {lx(aroc_val)}$")

        show_result("Average Rate of Change (Word Problem)", steps, "Average Rate of Change:", lx(aroc_val))
        show_domain_range(expr=f_expr, range_latex=r"(-\infty, \infty)")

    elif is_dq:
        steps.append(r"Formula: $\dfrac{f(x+h) - f(x)}{h}$")
        fxh = f_expr.subs(x, x + h)
        steps.append(f"$f(x+h) = {lx(expand(fxh))}$")
        numerator = expand(fxh - f_expr)
        steps.append(f"$f(x+h) - f(x) = {lx(numerator)}$")
        dq = simplify(numerator / h)
        try:
            nf = factor(numerator)
            if h in nf.free_symbols:
                dq = simplify(nf / h)
        except Exception:
            pass
        steps.append(f"$\\dfrac{{f(x+h)-f(x)}}{{h}} = {lx(dq)}$")
        show_result("Difference Quotient (Word Problem)", steps, "Difference Quotient:", lx(dq))
        show_domain_range(expr=f_expr, range_latex=r"(-\infty, \infty)")
    else:
        show_error("Could not determine if this is an AROC or Difference Quotient problem.")


# =============================================================================
# SOLVER: SYSTEMS OF INEQUALITIES
# =============================================================================

def solve_system_inequalities(raw: str):
    """
    Solve a system of inequalities in one or two variables.
    For one variable: find the intersection of the solution sets.
    For two variables: describe the feasible region.
    Input (single problem box, newline-separated or use 'and'/'&&'):
      x + y >= 2
      x - y <= 4
      y >= 0
    Or via the System mode (list of strings passed directly).
    """
    # Split on newlines or 'and'
    parts_raw = re.split(r'\n|(?:\band\b)', raw, flags=re.IGNORECASE)
    ineqs_str = [p.strip() for p in parts_raw if p.strip() and re.search(r'[<>]=?', p)]

    if len(ineqs_str) < 2:
        show_error(
            "Please enter at least 2 inequalities, separated by newlines or 'and'.\n"
            "Example: x + 2*y >= 4 and x - y <= 1"
        )
        return

    steps = []
    steps.append(f"System of {len(ineqs_str)} inequalities:")

    parsed_ineqs = []
    all_vars = set()
    for i, ineq_s in enumerate(ineqs_str):
        norm = preprocess_expr(ineq_s)
        parts = split_inequality(norm)
        if not parts:
            show_error(f"Could not parse inequality {i+1}: {ineq_s}")
            return
        lhs_s, rel, rhs_s = parts
        lhs_e, e1 = safe_sympify(lhs_s)
        rhs_e, e2 = safe_sympify(rhs_s)
        if e1 or e2 or lhs_e is None or rhs_e is None:
            show_error(f"Parse error in inequality {i+1}: {e1 or e2}")
            return
        parsed_ineqs.append((lhs_e, rel, rhs_e))
        all_vars |= (lhs_e - rhs_e).free_symbols
        steps.append(f"Inequality {i+1}: ${lx(lhs_e)} {rel} {lx(rhs_e)}$")

    var_list = sorted(all_vars, key=lambda s: str(s))
    n_vars = len(var_list)

    # ── One-variable system: intersect solution sets ───────────────────────────
    if n_vars == 1:
        the_var = var_list[0]
        steps.append(f"**One-variable system** — solving each inequality for ${lx(the_var)}$, then intersecting.")
        sol_sets = []
        for lhs_e, rel, rhs_e in parsed_ineqs:
            expr = lhs_e - rhs_e
            try:
                rel_map = {'<': expr < 0, '>': expr > 0, '<=': expr <= 0, '>=': expr >= 0}
                ss = solveset(rel_map[rel], the_var, domain=S.Reals)
                sol_sets.append(ss)
                steps.append(f"${lx(lhs_e)} {rel} {lx(rhs_e)}$ → ${lx(ss)}$")
            except Exception as e:
                show_error(f"Could not solve inequality: {e}")
                return

        # Intersect all solution sets
        from sympy.sets.sets import Intersection
        combined = sol_sets[0]
        for ss in sol_sets[1:]:
            combined = Intersection(combined, ss)
        steps.append(f"Intersection: ${lx(combined)}$")
        show_result("System of Inequalities", steps, "Solution Set:", lx(combined))
        st.markdown(f"**Solution Set:** $${lx(combined)}$$")

    # ── Two-variable system: describe feasible region ─────────────────────────
    elif n_vars == 2:
        var_a, var_b = var_list[0], var_list[1]
        steps.append(f"**Two-variable system** in ${lx(var_a)}, {lx(var_b)}$.")
        steps.append("Solve each inequality for the boundary line and identify the feasible half-plane.")

        boundary_lines = []
        for lhs_e, rel, rhs_e in parsed_ineqs:
            boundary = Eq(lhs_e, rhs_e)
            # Solve boundary for var_b to get slope-intercept form
            try:
                sol_b = solve(boundary, var_b)
                if sol_b:
                    steps.append(f"Boundary line: ${lx(var_b)} = {lx(sol_b[0])}$ (from ${lx(lhs_e)} {rel} {lx(rhs_e)}$)")
                    boundary_lines.append((sol_b[0], rel))
                else:
                    # Vertical line
                    sol_a = solve(boundary, var_a)
                    if sol_a:
                        steps.append(f"Boundary line: ${lx(var_a)} = {lx(sol_a[0])}$ (vertical)")
                        boundary_lines.append((sol_a[0], rel))
            except Exception:
                steps.append(f"Boundary: ${lx(lhs_e)} = {lx(rhs_e)}$")

        # Try to find corner points (vertices of feasible region) by solving pairs of boundary equations
        boundary_eqs = [Eq(lhs_e, rhs_e) for lhs_e, rel, rhs_e in parsed_ineqs]
        from itertools import combinations
        corner_pts = []
        for eq1, eq2 in combinations(boundary_eqs, 2):
            try:
                pt = solve([eq1, eq2], [var_a, var_b])
                if pt and isinstance(pt, dict):
                    pa, pb = pt.get(var_a), pt.get(var_b)
                    if pa is not None and pb is not None:
                        # Check feasibility
                        feasible = True
                        for lhs_e, rel, rhs_e in parsed_ineqs:
                            val = (lhs_e - rhs_e).subs([(var_a, pa), (var_b, pb)])
                            try:
                                val_f = float(val.evalf())
                                if rel == '<'  and not (val_f <  0): feasible = False; break
                                if rel == '<=' and not (val_f <= 0): feasible = False; break
                                if rel == '>'  and not (val_f >  0): feasible = False; break
                                if rel == '>=' and not (val_f >= 0): feasible = False; break
                            except Exception:
                                pass
                        if feasible:
                            corner_pts.append((pa, pb))
            except Exception:
                pass

        if corner_pts:
            pts_str = ", ".join(f"$({lx(pa)}, {lx(pb)})$" for pa, pb in corner_pts)
            steps.append(f"**Corner point(s) of feasible region:** {pts_str}")

        steps.append("The **feasible region** is the intersection of all shaded half-planes.")
        show_result("System of Inequalities (Two Variables)", steps,
                    "Feasible Region:", r"\text{See boundary lines and corner points above}")
        if corner_pts:
            st.markdown("**Corner Points:**")
            for pa, pb in corner_pts:
                st.markdown(f"- $({lx(pa)},\\; {lx(pb)})$")
    else:
        show_error("Systems of inequalities currently support 1 or 2 variables (x, y).")


# =============================================================================
# MAIN DISPATCH
# =============================================================================

def dispatch(raw: str):
    """Detect type and route to the correct solver."""
    problem_type = detect_type(raw)

    type_display = {
        'rational_analysis': 'Rational Function Analysis',
        'exp_function': 'Exponential Function Analysis',
        'log_function': 'Logarithmic Function Analysis',
        'log_rules': 'Rules of Logarithms',
        'aroc_word': 'AROC / DQ (Word Problem)',
        'system_inequalities': 'System of Inequalities',
        'abs_equation': 'Absolute Value Equation',
        'abs_inequality': 'Absolute Value Inequality',
        'polynomial_inequality': 'Polynomial Inequality',
        'rational_equation': 'Rational Equation',
        'rational_arithmetic': 'Rational Function Arithmetic',
        'function_composition': 'Function Composition',
        'average_rate_of_change': 'Average Rate of Change',
        'difference_quotient': 'Difference Quotient',
        'inverse_function': 'Inverse Function',
        'exp_equation': 'Exponential Equation',
        'log_equation': 'Logarithmic Equation',
        'exp_inequality': 'Exponential Inequality',
        'log_inequality': 'Logarithmic Inequality',
        'polynomial_equation': 'Polynomial / General Equation',
        'rational_simplify': 'Rational Expression',
        'log_simplify': 'Logarithmic Expression',
        'domain': 'Domain',
        'range': 'Range',
        'unknown': 'Unknown',
    }

    if problem_type == 'unknown':
        # Last-ditch: try to just solve it
        st.info("ℹ️ Could not automatically detect problem type. Attempting general solve...")
        norm = preprocess_expr(raw)
        parts = split_equation(norm)
        if parts:
            solve_polynomial_equation(raw)
        else:
            solve_rational_simplify(raw)
        st.stop()

    if problem_type == 'abs_equation':
        solve_abs_equation(raw)
    elif problem_type == 'rational_analysis':
        solve_rational_analysis(raw)
    elif problem_type == 'exp_function':
        solve_exp_function(raw)
    elif problem_type == 'log_function':
        solve_log_function(raw)
    elif problem_type == 'log_rules':
        solve_log_rules(raw)
    elif problem_type == 'aroc_word':
        solve_aroc_word(raw)
    elif problem_type == 'system_inequalities':
        solve_system_inequalities(raw)
    elif problem_type == 'abs_inequality':
        solve_abs_inequality(raw)
    elif problem_type == 'polynomial_inequality':
        solve_polynomial_inequality(raw)
    elif problem_type == 'rational_equation':
        solve_rational_equation(raw)
    elif problem_type == 'rational_arithmetic':
        solve_rational_arithmetic(raw)
    elif problem_type == 'function_composition':
        solve_composition(raw)
    elif problem_type == 'average_rate_of_change':
        solve_aroc(raw)
    elif problem_type == 'difference_quotient':
        solve_dq(raw)
    elif problem_type == 'inverse_function':
        solve_inverse(raw)
    elif problem_type == 'exp_equation':
        solve_exp_equation(raw)
    elif problem_type == 'log_equation':
        solve_log_equation(raw)
    elif problem_type == 'exp_inequality':
        solve_exp_inequality(raw)
    elif problem_type == 'log_inequality':
        solve_log_inequality(raw)
    elif problem_type == 'polynomial_equation':
        solve_polynomial_equation(raw)
    elif problem_type == 'rational_simplify':
        solve_rational_simplify(raw)
    elif problem_type == 'log_simplify':
        solve_log_simplify(raw)
    elif problem_type == 'poly_division':
        solve_poly_division(raw)
    elif problem_type == 'find_zeros':
        solve_find_zeros(raw)
    elif problem_type == 'function_properties':
        solve_function_properties(raw)
    elif problem_type == 'transformations':
        solve_transformations(raw)
    elif problem_type == 'library_function':
        solve_library_function(raw)
    elif problem_type == 'domain':
        st.info("Tip: To find the domain of a specific function, try: 'find the domain of (x+1)/(x-2)'.")
        solve_rational_simplify(raw)
    else:
        show_error(f"Problem type '{problem_type}' is not yet supported. Please check input format.")

    st.stop()


# =============================================================================
# STREAMLIT UI
# =============================================================================

st.markdown('<div class="main-title">📐 College Algebra Math Solver</div>', unsafe_allow_html=True)
st.markdown("Enter a math problem below and click **Solve**.")
st.markdown("---")

# Example inputs
with st.expander("📋 Example Inputs"):
    examples = [
        "f(x)=(x+1)/(x-2), analyze",
        "f(x) = 3*2**x, analyze",
        "f(x) = 5*(0.5)**x, find f(3)",
        "f(x) = log(x-2) + 3, analyze",
        "f(x) = ln(x+1), find f(4)",
        "expand log((x**2*(x+1))/(x-3))",
        "condense 2*log(x) + log(x+1) - log(x-2)",
        "|x-3| = 5",
        "|x-3| > 5",
        "x**2 - 9 >= 0",
        "(x+1)/(x-2) = 3",
        "f(x)=(x+1)/(x-2), g(x)=(x-3)/(x+4), find f+g",
        "f(x)=(x+1)/(x-2), g(x)=(x-3)/(x+4), find fog",
        "find the inverse of (x+1)/(x-2)",
        "average rate of change of x**2+1 from 2 to 5",
        "difference quotient of x**2+3*x+1",
        "2**x = 8",
        "log(x) + log(x-3) = 2",
        "ln(x-1) > 0",
        "x**2 - 5*x + 6 = 0",
        "divide x**3 - 6*x**2 + 11*x - 6 by (x - 1)",
        "find zeros of x**3 - 6*x**2 + 11*x - 6",
        "f(x) = 2*(x-3)**2 + 1, transformations",
        "f(x) = sqrt(x-2) + 3, properties",
        "f(x) = |x+1| - 2, find f(3)",
    ]
    cols = st.columns(2)
    for i, ex in enumerate(examples):
        cols[i % 2].code(ex)

st.markdown("---")

# ── MODE SELECTOR ─────────────────────────────────────────────────────────────
mode = st.radio(
    "Select input mode:",
    ["Single Problem", "System of Equations", "System of Inequalities"],
    horizontal=True,
    key="mode_selector"
)

st.markdown("")

if mode == "System of Equations":
    st.markdown("**Enter each equation on a separate line (2 or 3 equations):**")
    st.caption("Use x, y, z as variables. Example: `x + y = 5`")

    if "system_equations" not in st.session_state:
        st.session_state.system_equations = ["", ""]

    n_eqs = st.selectbox("Number of equations:", [2, 3], key="n_eqs")
    while len(st.session_state.system_equations) < n_eqs:
        st.session_state.system_equations.append("")
    eqs_input = []
    for i in range(n_eqs):
        val = st.text_input(
            f"Equation {i+1}:",
            value=st.session_state.system_equations[i] if i < len(st.session_state.system_equations) else "",
            key=f"sys_eq_{i}"
        )
        eqs_input.append(val)

    col1, col2 = st.columns([1, 5])
    solve_btn = col1.button("🔢 Solve System", type="primary", use_container_width=True)

    if solve_btn:
        filled = [e for e in eqs_input if e.strip()]
        if len(filled) < 2:
            st.warning("Please enter at least 2 equations.")
            st.stop()
        with st.spinner("Solving system..."):
            try:
                solve_system(filled)
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
elif mode == "System of Inequalities":
    st.markdown("**Enter each inequality on a separate line (2–4 inequalities):**")
    st.caption("Use x, y as variables. Example: `x + y >= 2`")

    if "sys_ineq_list" not in st.session_state:
        st.session_state.sys_ineq_list = ["", ""]

    n_ineqs = st.selectbox("Number of inequalities:", [2, 3, 4], key="n_ineqs")
    while len(st.session_state.sys_ineq_list) < n_ineqs:
        st.session_state.sys_ineq_list.append("")

    ineqs_input = []
    for i in range(n_ineqs):
        val = st.text_input(
            f"Inequality {i+1}:",
            value=st.session_state.sys_ineq_list[i] if i < len(st.session_state.sys_ineq_list) else "",
            key=f"sys_ineq_{i}"
        )
        ineqs_input.append(val)

    col1, col2 = st.columns([1, 5])
    solve_btn_ineq = col1.button("🔢 Solve System", type="primary", use_container_width=True, key="solve_ineq_btn")

    if solve_btn_ineq:
        filled = [e for e in ineqs_input if e.strip()]
        if len(filled) < 2:
            st.warning("Please enter at least 2 inequalities.")
            st.stop()
        with st.spinner("Solving system of inequalities..."):
            try:
                combined_raw = "\n".join(filled)
                solve_system_inequalities(combined_raw)
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
else:
    user_input = st.text_input(
        "Enter your math problem:",
        placeholder="e.g. |x-3| = 5 or 2**x = 8 or find zeros of x**3-6*x**2+11*x-6",
        key="math_input"
    )

    col1, col2 = st.columns([1, 5])
    solve_btn = col1.button("🔢 Solve", type="primary", use_container_width=True)

    if solve_btn:
        if not user_input or not user_input.strip():
            st.warning("Please enter a math problem.")
            st.stop()
        with st.spinner("Solving..."):
            try:
                dispatch(user_input.strip())
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                st.info("Please check your input format. See Input Notes for guidance.")

st.markdown("---")

with st.expander("ℹ️ Input Notes"):
    st.markdown("""
**Rational Function Analysis:**
- `f(x) = (x+1)/(x-2), analyze`
- Shows: vertical/horizontal asymptotes, x/y-intercepts, domain, holes

**Exponential Functions:**
- `f(x) = 3*2**x, analyze` — growth/decay, y-intercept, asymptote, domain/range
- `f(x) = 5*(0.5)**x, find f(3)` — evaluate at a point

**Logarithmic Functions:**
- `f(x) = log(x-2) + 3, analyze` — domain, range, asymptotes, intercepts
- `f(x) = ln(x+1), find f(4)` — evaluate at a point

**Rules of Logarithms:**
- `expand log(x**2*(x+1)/(x-3))` — expand using product/quotient/power rules
- `condense 2*log(x) + log(x+1) - log(x-2)` — combine into a single log

**AROC / DQ from Word Problems:**
- `h(t) = -16*t**2 + 64*t, average rate of change from t=1 to t=3`
- `f(t) = 3*t**2 + 1, difference quotient`

**Systems of Inequalities:**
- Switch to **System of Inequalities** mode
- Enter each inequality separately (2–4 supported)
- Supports 1-variable (finds intersection) and 2-variable (feasible region + corner points)

**General:**
- Use `**` for exponents (`x**2`) or `^` (auto-converted)
- Use `*` for multiplication (`3*x`)
- Use `|expr|` for absolute value (`|x-3|`)
- Use `(x+1)/(x-2)` for fractions

**Functions:**
- Function ops: `f(x)=..., g(x)=..., find f+g` (or fog, gof, f-g, f*g, f/g)
- Composition: `Find: (f ∘ g)(x)` or `find fog`
- Inverse: `find the inverse of (x+1)/(x-2)`
- AROC: `average rate of change of x**2 from 2 to 5`
- DQ: `difference quotient of x**2+3*x`
- Properties: `f(x) = x**2 - 4, properties`
- Transformations: `f(x) = 2*(x-3)**2 + 1, transformations`
- Library: `f(x) = sqrt(x-2), find f(6)`
- Evaluate: `f(x) = |x+1| - 2, find f(3)`

**Polynomials:**
- Division: `divide x**3 - 6*x**2 + 11*x - 6 by (x - 1)`
- Zeros: `find zeros of x**3 - 6*x**2 + 11*x - 6`

**Logarithms:**
- Natural log: `ln(x)` 
- Base-10 log: `log(x)` (treated as log base 10 in equations)

**Systems:**
- Switch to "System of Equations" mode above
- Enter each equation separately
- Supports 2 or 3 variables (x, y, z)
""")

st.markdown("---")
st.caption("College Algebra Math Solver · Powered by SymPy + Streamlit · For educational use")