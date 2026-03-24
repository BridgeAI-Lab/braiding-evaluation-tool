import os
import json
from datetime import datetime
from io import BytesIO

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_file,
    abort,
)
from flask_login import (
    LoginManager,
    login_user,
    logout_user,
    login_required,
    current_user,
)
import pandas as pd
from scipy import stats
import ast
import re

from config import UPLOAD_FOLDER
from models import db, User, EvalFile, Assignment, Annotation, format_evaluation_display, get_guidelines, save_guidelines, restore_default_guidelines

PER_PAGE = 20

# Placeholder for GPT F, C, H (no GPT values in Excel yet)
GPT_PLACEHOLDER_F, GPT_PLACEHOLDER_C, GPT_PLACEHOLDER_H = 3, 3, 3


def _parse_trust(val):
    """Parse trust value to [a, b]. Handles '[80, 20]' string, list, or numpy types."""
    if val is None or (hasattr(val, "__len__") and len(val) == 0):
        return None
    if isinstance(val, (list, tuple)) and len(val) >= 2:
        try:
            return [int(float(val[0])), int(float(val[1]))]
        except (ValueError, TypeError):
            return None
    if isinstance(val, str):
        m = re.match(r"\[?\s*([\d.]+)\s*[,]\s*([\d.]+)\s*\]?", val.strip())
        if m:
            return [int(float(m.group(1))), int(float(m.group(2)))]
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
                return [int(float(parsed[0])), int(float(parsed[1]))]
        except (ValueError, SyntaxError):
            pass
    return None


def _correlation(x, y, method):
    """Compute correlation. x, y are lists. method: pearson, spearman, kendall. Returns (None, None) if undefined (e.g. constant input)."""
    import numpy as np
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)
    mask = ~(np.isnan(x_arr) | np.isnan(y_arr))
    x_clean = x_arr[mask]
    y_clean = y_arr[mask]
    if len(x_clean) < 2:
        return None, None
    if np.std(x_clean) == 0 or np.std(y_clean) == 0:
        return None, None  # Correlation undefined when one variable is constant
    if method == "pearson":
        r, p = stats.pearsonr(x_clean, y_clean)
    elif method == "spearman":
        r, p = stats.spearmanr(x_clean, y_clean)
    elif method == "kendall":
        r, p = stats.kendalltau(x_clean, y_clean)
    else:
        r, p = stats.pearsonr(x_clean, y_clean)
    if np.isnan(r) or np.isnan(p):
        return None, None
    return float(r), float(p)


def _compute_annotator_agreement(file_id, samples, assignments):
    """
    Compute Cohen's kappa (2 ann), Fleiss' kappa (3+ ann), Krippendorff's alpha (2+ ann)
    for Trust A, Trust B, Fluency, Coherency, Hallucination.
    Returns dict with per-dimension results.
    """
    try:
        from sklearn.metrics import cohen_kappa_score
        from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
        import krippendorff
        import numpy as np
    except ImportError:
        return None

    # Build annotator->{narrative_number: value} for each dimension
    ann_data = {}
    for a in assignments:
        anns = {
            int(ann.narrative_number) if ann.narrative_number is not None else None: json.loads(ann.evaluation_json)
            for ann in Annotation.query.filter_by(file_id=file_id, user_id=a.user_id).all()
        }
        ann_data[a.user_id] = {"user": a.user, "anns": anns}

    n_annotators = len(assignments)
    dimensions = [
        ("Trust A", lambda ev: (ev.get("trust") or [0, 0])[0] if ev else None, "ordinal"),
        ("Trust B", lambda ev: (ev.get("trust") or [0, 0])[1] if ev and (ev.get("trust") or []) and len(ev.get("trust", [])) > 1 else None, "ordinal"),
        ("Fluency", lambda ev: ev.get("fluency") if ev else None, "ordinal"),
        ("Coherency", lambda ev: ev.get("coherency") if ev else None, "ordinal"),
        ("Hallucination", lambda ev: ev.get("hallucination") if ev else None, "ordinal"),
    ]

    results = {}
    for dim_name, get_val, level in dimensions:
        # Build matrix: samples (rows) x annotators (cols), value or None
        # Only include samples with >= 2 annotations
        rows_by_sample = {}
        for s in samples:
            n = s.get("narrative_number")
            try:
                n_key = int(n) if n is not None else None
            except (ValueError, TypeError):
                continue
            if n_key is None:
                continue
            vals = []
            for a in assignments:
                ev = ann_data.get(a.user_id, {}).get("anns", {}).get(n_key)
                v = get_val(ev)
                vals.append(v)
            if sum(1 for x in vals if x is not None) >= 2:
                rows_by_sample[n_key] = vals

        if len(rows_by_sample) < 2:
            results[dim_name] = {"cohen": None, "fleiss": None, "krippendorff": None, "percent_agreement": None, "mae": None}
            continue

        sample_order = sorted(rows_by_sample.keys())
        # reliability_data for Krippendorff: list of lists, each list = one annotator's scores across samples
        # shape: n_annotators x n_samples, use None for missing
        reliability_data = []
        for j in range(n_annotators):
            col = [rows_by_sample[s][j] for s in sample_order]
            reliability_data.append(col)

        # Cohen's kappa: only for exactly 2 annotators
        cohen_val = None
        if n_annotators == 2:
            a1 = [reliability_data[0][i] for i in range(len(sample_order)) if reliability_data[0][i] is not None and reliability_data[1][i] is not None]
            a2 = [reliability_data[1][i] for i in range(len(sample_order)) if reliability_data[0][i] is not None and reliability_data[1][i] is not None]
            if len(a1) >= 2:
                try:
                    cohen_val = cohen_kappa_score(a1, a2, weights="quadratic")
                except Exception:
                    cohen_val = None

        # Fleiss' kappa: 3+ annotators. Need (n_samples x n_categories) count matrix.
        # Only use samples where all annotators rated (complete rows) - aggregate_raters needs integers.
        fleiss_val = None
        if n_annotators >= 3:
            try:
                if "Trust" in dim_name:
                    def to_cat(v):
                        if v is None: return None
                        return min(int(float(v) / 20.01), 4)
                else:
                    def to_cat(v):
                        if v is None: return None
                        return max(0, min(int(float(v)) - 1, 4))

                raw = []
                for s in sample_order:
                    row_vals = [rows_by_sample[s][j] for j in range(n_annotators)]
                    cats = [to_cat(v) for v in row_vals]
                    if None in cats or len([c for c in cats if c is not None]) < 3:
                        continue
                    raw.append([c for c in cats])

                if len(raw) >= 2:
                    arr = np.array(raw, dtype=int)
                    table, _ = aggregate_raters(arr, n_cat=5)
                    fleiss_val = float(fleiss_kappa(table))
            except Exception:
                fleiss_val = None

        # Krippendorff's alpha: 2+ annotators, handles missing
        krip_val = None
        try:
            krip_val = krippendorff.alpha(
                reliability_data=reliability_data,
                level_of_measurement=level
            )
        except Exception:
            krip_val = None

        # Percent agreement: raw proportion of samples where all annotators gave the exact same score
        n_total = len(rows_by_sample)
        n_exact_match = 0
        for s in sample_order:
            raw_vals = [v for v in rows_by_sample[s] if v is not None]
            if len(raw_vals) < 2:
                continue
            # Normalize so 70.0 and 70 count as same (int/float from JSON)
            vals = [int(float(v)) if isinstance(v, (int, float)) else v for v in raw_vals]
            if len(set(vals)) == 1:
                n_exact_match += 1
        pct_agree = (n_exact_match / n_total * 100) if n_total > 0 else None

        # MAE (mean absolute error): only for Trust (0-100 continuous scale)
        mae_val = None
        if "Trust" in dim_name:
            mae_diffs = []
            for s in sample_order:
                raw_vals = [float(v) for v in rows_by_sample[s] if v is not None and isinstance(v, (int, float))]
                if len(raw_vals) < 2:
                    continue
                pair_diffs = []
                for i in range(len(raw_vals)):
                    for j in range(i + 1, len(raw_vals)):
                        pair_diffs.append(abs(raw_vals[i] - raw_vals[j]))
                if pair_diffs:
                    mae_diffs.append(sum(pair_diffs) / len(pair_diffs))
            mae_val = round(sum(mae_diffs) / len(mae_diffs), 2) if mae_diffs else None

        results[dim_name] = {
            "cohen": cohen_val,
            "fleiss": fleiss_val,
            "krippendorff": krip_val,
            "percent_agreement": round(pct_agree, 1) if pct_agree is not None else None,
            "mae": mae_val,
        }

    return results


def update_annotator_activity():
    """Update last_activity_at for non-admin users."""
    if current_user.is_authenticated and not current_user.is_admin:
        current_user.last_activity_at = datetime.utcnow()
        db.session.commit()


def display_filename(filename):
    """Remove .xlsx/.xls extension for display."""
    if not filename:
        return ""
    return filename.rsplit(".", 1)[0] if "." in filename else filename


def _to_int_safe(val):
    """Convert value to int, handling None/NaN/numpy."""
    if val is None or (hasattr(val, "__float__") and pd.isna(val)):
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def _get_gpt_fch(sample):
    """Get GPT Fluency, Coherency, Hallucination from sample. Returns (f, c, h). Checks multiple column names."""
    def get_val(*keys):
        for k in keys:
            v = sample.get(k)
            if v is not None and not (hasattr(v, "__float__") and pd.isna(v)):
                return _to_int_safe(v)
        # Fallback: match by stripped key (Excel may have "Hallucination " with trailing space)
        for col, v in sample.items():
            if isinstance(col, str) and col.strip() and v is not None and not (hasattr(v, "__float__") and pd.isna(v)):
                if any(col.strip().lower() == k.lower() for k in keys):
                    return _to_int_safe(v)
        return None
    f = get_val("Fluency", "fluency", "eval_fluency", "eval_Fluency")
    c = get_val("Coherency", "coherency", "eval_coherency", "eval_Coherency")
    h = get_val("Hallucination", "hallucination", "eval_hallucination", "eval_Hallucination")
    return (f, c, h)


def format_gpt_sample_display(sample):
    """Format GPT values from Excel sample as [trust] | F:x | C:x | H:x (same as annotators)."""
    t = _parse_trust(sample.get("gpt_trust"))
    if t is None:
        t = [0, 0]
    f, c, h = _get_gpt_fch(sample)
    f = f if f is not None else 0
    c = c if c is not None else 0
    h = h if h is not None else 0
    return f"[{t[0]},{t[1]}]|F:{f}|C:{c}|H:{h}"


app = Flask(__name__)
def _display_name(x):
    if isinstance(x, str):
        return display_filename(x)
    return display_filename(getattr(x, "filename", "") or "")


def capitalize_name(name):
    """Capitalize first letter of name (e.g. alice -> Alice)."""
    if not name:
        return ""
    s = str(name).strip()
    return s[0].upper() + s[1:].lower() if len(s) > 1 else s.upper()


app.jinja_env.filters["display_name"] = _display_name
app.jinja_env.filters["capitalize"] = capitalize_name
app.jinja_env.filters["format_gpt"] = format_gpt_sample_display
app.config.from_object("config")

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"
login_manager.login_message = "Please log in to access this page."


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


@app.context_processor
def inject_guidelines():
    """Make guidelines available in all templates."""
    return {"guidelines_html": get_guidelines()}


def format_time_ago(dt):
    """Format datetime as 'X min ago', '2 hours ago', etc."""
    if not dt:
        return "—"
    now = datetime.utcnow()
    diff = now - dt
    secs = diff.total_seconds()
    if secs < 60:
        return "just now"
    if secs < 3600:
        m = int(secs / 60)
        return f"{m} min ago"
    if secs < 86400:
        h = int(secs / 3600)
        return f"{h} hour{'s' if h != 1 else ''} ago"
    if secs < 604800:
        d = int(secs / 86400)
        return f"{d} day{'s' if d != 1 else ''} ago"
    return dt.strftime("%Y-%m-%d %H:%M")


# ---------- Auth routes ----------


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if not username or not password:
            flash("Username and password are required.", "error")
            return render_template("register.html")
        if User.query.filter_by(username=username).first():
            flash("Username already exists.", "error")
            return render_template("register.html")
        user = User(username=username)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        flash("Registration successful. Please log in.", "success")
        return redirect(url_for("login"))
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            user.last_login_at = datetime.utcnow()
            user.last_activity_at = datetime.utcnow()
            db.session.commit()
            next_page = request.args.get("next") or url_for("index")
            return redirect(next_page)
        flash("Invalid username or password.", "error")
    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


# ---------- User index (annotator tasks) ----------


@app.route("/")
def index():
    if not current_user.is_authenticated:
        return redirect(url_for("login"))
    if current_user.is_admin:
        return redirect(url_for("admin_dashboard"))
    # Annotator: list assigned files
    update_annotator_activity()
    assignments = Assignment.query.filter_by(user_id=current_user.id).all()
    files = []
    for a in assignments:
        f = EvalFile.query.get(a.file_id)
        if f and not f.archived:
            samples = json.loads(f.samples_json)
            anns = Annotation.query.filter_by(
                file_id=f.id, user_id=current_user.id
            ).all()
            done = len(anns)
            total = len(samples)
            files.append(
                {
                    "id": f.id,
                    "filename": f.filename,
                    "display_name": display_filename(f.filename),
                    "done": done,
                    "total": total,
                    "progress_pct": round(100 * done / total, 1) if total else 0,
                }
            )
    return render_template("user_index.html", files=files)


# ---------- Annotator: annotate samples ----------


@app.route("/annotate/<int:file_id>")
@login_required
def annotate_file(file_id):
    if current_user.is_admin:
        return redirect(url_for("admin_dashboard"))
    f = EvalFile.query.get_or_404(file_id)
    if f.archived:
        abort(404)
    if not Assignment.query.filter_by(file_id=file_id, user_id=current_user.id).first():
        abort(403)
    update_annotator_activity()
    samples = json.loads(f.samples_json)
    annotations = Annotation.query.filter_by(
        file_id=file_id, user_id=current_user.id
    ).all()
    anns = {a.narrative_number: json.loads(a.evaluation_json) for a in annotations}
    flagged_nums = {int(a.narrative_number) for a in annotations if getattr(a, "flagged", False)}
    done = len(anns)
    total = len(samples)
    progress_pct = round(100 * done / total, 1) if total else 0
    completed_nums = {int(k) for k in anns.keys()}
    page = request.args.get("page", 1, type=int)
    total_pages = (total + PER_PAGE - 1) // PER_PAGE if total else 1
    page = max(1, min(page, total_pages))
    start = (page - 1) * PER_PAGE
    end = start + PER_PAGE
    page_samples = samples[start:end]
    start_index = (page - 1) * PER_PAGE
    return render_template(
        "annotate.html",
        file=f,
        samples=page_samples,
        annotations=anns,
        done=done,
        total=total,
        progress_pct=progress_pct,
        completed_nums=completed_nums,
        flagged_nums=flagged_nums,
        page=page,
        total_pages=total_pages,
        start_index=start_index,
    )


@app.route("/annotate/<int:file_id>/sample/<int:narrative_number>", methods=["GET", "POST"])
@login_required
def annotate_sample(file_id, narrative_number):
    if current_user.is_admin:
        return redirect(url_for("admin_dashboard"))
    f = EvalFile.query.get_or_404(file_id)
    if f.archived:
        abort(404)
    if not Assignment.query.filter_by(file_id=file_id, user_id=current_user.id).first():
        abort(403)
    update_annotator_activity()
    samples = json.loads(f.samples_json)
    sample = next((s for s in samples if s.get("narrative_number") == narrative_number), None)
    if not sample:
        abort(404)
    ann = Annotation.query.filter_by(
        file_id=file_id,
        user_id=current_user.id,
        narrative_number=narrative_number,
    ).first()
    existing = json.loads(ann.evaluation_json) if ann else None

    if request.method == "POST":
        try:
            trust_n1 = int(request.form.get("trust_n1", 0))
            trust_n2 = int(request.form.get("trust_n2", 0))
            fluency = int(request.form.get("fluency", 0))
            coherency = int(request.form.get("coherency", 0))
            hallucination = int(request.form.get("hallucination", 0))
            if fluency < 1 or fluency > 5 or coherency < 1 or coherency > 5 or hallucination < 1 or hallucination > 5:
                flash("Fluency, Coherency, and Hallucination must be 1-5.", "error")
                return redirect(request.url)
            eval_dict = {
                "trust": [trust_n1, trust_n2],
                "fluency": fluency,
                "coherency": coherency,
                "hallucination": hallucination,
            }
            if ann:
                ann.evaluation_json = json.dumps(eval_dict)
            else:
                ann = Annotation(
                    file_id=file_id,
                    user_id=current_user.id,
                    narrative_number=narrative_number,
                    evaluation_json=json.dumps(eval_dict),
                )
                db.session.add(ann)
            db.session.commit()
            flash("Annotation saved.", "success")
        except (ValueError, TypeError) as e:
            flash("Invalid input.", "error")
        next_sample = request.form.get("next")
        if next_sample:
            return redirect(url_for("annotate_sample", file_id=file_id, narrative_number=int(next_sample)))
        return redirect(url_for("annotate_file", file_id=file_id))

    # Find prev/next for navigation
    idx = next(i for i, s in enumerate(samples) if s.get("narrative_number") == narrative_number)
    prev_num = samples[idx - 1]["narrative_number"] if idx > 0 else None
    next_num = samples[idx + 1]["narrative_number"] if idx < len(samples) - 1 else None

    is_flagged = ann.flagged if ann else False
    return render_template(
        "annotate_sample.html",
        file=f,
        sample=sample,
        existing=existing,
        prev_num=prev_num,
        next_num=next_num,
        total=len(samples),
        current_idx=idx + 1,
        is_flagged=is_flagged,
    )


# ---------- Admin routes ----------


@app.route("/admin")
@login_required
def admin_dashboard():
    if not current_user.is_admin:
        abort(403)
    files = EvalFile.query.filter_by(archived=False).order_by(EvalFile.uploaded_at.desc()).all()
    file_progress = []
    for f in files:
        samples = json.loads(f.samples_json)
        total_samples = len(samples)
        assignments = Assignment.query.filter_by(file_id=f.id).all()
        annotator_progress = []
        for a in assignments:
            done = Annotation.query.filter_by(file_id=f.id, user_id=a.user_id).count()
            annotator_progress.append({"username": a.user.username, "done": done, "total": total_samples})
        file_progress.append({"file": f, "total_samples": total_samples, "annotators": annotator_progress})
    archived_count = EvalFile.query.filter_by(archived=True).count()
    return render_template("admin_dashboard.html", file_progress=file_progress, archived_count=archived_count)


@app.route("/admin/archived")
@login_required
def admin_archived():
    if not current_user.is_admin:
        abort(403)
    files = EvalFile.query.filter_by(archived=True).order_by(EvalFile.uploaded_at.desc()).all()
    file_progress = []
    for f in files:
        samples = json.loads(f.samples_json)
        total_samples = len(samples)
        assignments = Assignment.query.filter_by(file_id=f.id).all()
        annotator_progress = []
        for a in assignments:
            done = Annotation.query.filter_by(file_id=f.id, user_id=a.user_id).count()
            annotator_progress.append({"username": a.user.username, "done": done, "total": total_samples})
        file_progress.append({"file": f, "total_samples": total_samples, "annotators": annotator_progress})
    return render_template("admin_archived.html", file_progress=file_progress)


@app.route("/admin/activity")
@login_required
def admin_activity():
    if not current_user.is_admin:
        abort(403)
    annotators = User.query.filter_by(is_admin=False).order_by(User.username).all()
    activity_data = []
    for u in annotators:
        total_anns = Annotation.query.filter_by(user_id=u.id).count()
        activity_data.append({
            "user": u,
            "total_annotations": total_anns,
            "last_login": format_time_ago(u.last_login_at),
            "last_activity": format_time_ago(u.last_activity_at),
            "is_active": u.last_activity_at and (datetime.utcnow() - u.last_activity_at).total_seconds() < 900,
        })
    return render_template("admin_activity.html", activity_data=activity_data)


def _compute_correlation_for_file(file_id, method="spearman"):
    """Compute correlation between GPT and annotator evaluations for a file. Returns list of annotator results."""
    if method not in ("pearson", "spearman", "kendall"):
        method = "spearman"
    f = EvalFile.query.get(file_id)
    if not f:
        return None, method
    samples = json.loads(f.samples_json)
    # Use users who have annotations (not just assigned) so we show results when annotations exist
    annotations = Annotation.query.filter_by(file_id=file_id).all()
    user_ids = sorted({a.user_id for a in annotations})
    if not user_ids:
        return [], method
    annotator_results = []
    for user_id in user_ids:
        user = User.query.get(user_id)
        if not user or user.is_admin:
            continue
        anns = {
            int(ann.narrative_number): json.loads(ann.evaluation_json)
            for ann in annotations if ann.user_id == user_id
        }
        gpt_trust_a, gpt_trust_b = [], []
        ann_trust_a, ann_trust_b = [], []
        ann_f, ann_c, ann_h = [], [], []
        gpt_f, gpt_c, gpt_h = [], [], []
        for s in samples:
            n = s.get("narrative_number")
            n_key = int(n) if n is not None else None
            gpt = _parse_trust(s.get("gpt_trust"))
            ev = anns.get(n_key) if anns and n_key is not None else None
            if gpt is None or not ev or not ev.get("trust"):
                continue
            t = ev["trust"]
            gpt_fch = _get_gpt_fch(s)
            gpt_trust_a.append(gpt[0])
            gpt_trust_b.append(gpt[1])
            ann_trust_a.append(t[0] if len(t) > 0 else 0)
            ann_trust_b.append(t[1] if len(t) > 1 else 0)
            ann_f.append(ev.get("fluency", 0))
            ann_c.append(ev.get("coherency", 0))
            ann_h.append(ev.get("hallucination", 0))
            gpt_f.append(gpt_fch[0] if gpt_fch[0] is not None else GPT_PLACEHOLDER_F)
            gpt_c.append(gpt_fch[1] if gpt_fch[1] is not None else GPT_PLACEHOLDER_C)
            gpt_h.append(gpt_fch[2] if gpt_fch[2] is not None else GPT_PLACEHOLDER_H)

        corr_trust_a = _correlation(gpt_trust_a, ann_trust_a, method)
        corr_trust_b = _correlation(gpt_trust_b, ann_trust_b, method)
        corr_f = _correlation(gpt_f, ann_f, method)
        corr_c = _correlation(gpt_c, ann_c, method)
        corr_h = _correlation(gpt_h, ann_h, method)

        annotator_results.append({
            "username": user.username,
            "n_annotated": len(ann_trust_a),
            "trust_a": corr_trust_a,
            "trust_b": corr_trust_b,
            "fluency": corr_f,
            "coherency": corr_c,
            "hallucination": corr_h,
        })
    return annotator_results, method


@app.route("/admin/correlation")
@login_required
def admin_correlation():
    if not current_user.is_admin:
        abort(403)
    files = EvalFile.query.filter_by(archived=False).order_by(EvalFile.uploaded_at.desc()).all()
    file_id = request.args.get("file_id", type=int)
    method = request.args.get("method", "spearman")
    results = None
    selected_file = None
    if file_id:
        selected_file = EvalFile.query.get(file_id)
        if selected_file:
            results, method = _compute_correlation_for_file(file_id, method)

    return render_template(
        "admin_correlation.html",
        files=files,
        selected_file=selected_file,
        results=results,
        method=method,
    )


@app.route("/admin/upload", methods=["GET", "POST"])
@login_required
def admin_upload():
    if not current_user.is_admin:
        abort(403)
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file selected.", "error")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("No file selected.", "error")
            return redirect(request.url)
        if not file.filename.lower().endswith((".xlsx", ".xls")):
            flash("Only Excel files (.xlsx, .xls) are allowed.", "error")
            return redirect(request.url)
        try:
            df = pd.read_excel(file)
            required = ["narrative_number", "dataset", "Review1", "Review2", "braided_narrative", "gt_trust", "gpt_trust"]
            for col in required:
                if col not in df.columns:
                    flash(f"Missing required column: {col}", "error")
                    return redirect(request.url)
            def clean_val(v, key):
                if pd.isna(v):
                    return None
                if hasattr(v, "item"):
                    v = v.item()
                if key == "narrative_number" and v is not None:
                    return int(v)
                return v

            samples = []
            for _, row in df.iterrows():
                s = {k: clean_val(row[k], k) for k in df.columns}
                samples.append(s)
            eval_file = EvalFile(
                filename=file.filename,
                samples_json=json.dumps(samples),
            )
            db.session.add(eval_file)
            db.session.commit()
            flash(f"File '{display_filename(file.filename).upper()}' uploaded successfully.", "success")
            return redirect(url_for("admin_dashboard"))
        except Exception as e:
            flash(f"Error processing file: {str(e)}", "error")
    return render_template("admin_upload.html")


@app.route("/admin/guidelines", methods=["GET", "POST"])
@login_required
def admin_guidelines():
    if not current_user.is_admin:
        abort(403)
    if request.method == "POST":
        if request.form.get("restore_default"):
            restore_default_guidelines()
            flash("Guidelines restored to default.", "success")
        else:
            content = request.form.get("guidelines", "")
            save_guidelines(content)
            flash("Guidelines updated.", "success")
        return redirect(url_for("admin_guidelines"))
    return render_template("admin_guidelines.html", guidelines=get_guidelines())


@app.route("/admin/file/<int:file_id>/archive", methods=["POST"])
@login_required
def admin_archive_file(file_id):
    if not current_user.is_admin:
        abort(403)
    f = EvalFile.query.get_or_404(file_id)
    f.archived = True
    db.session.commit()
    flash(f"File '{display_filename(f.filename).upper()}' archived.", "success")
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/file/<int:file_id>/unarchive", methods=["POST"])
@login_required
def admin_unarchive_file(file_id):
    if not current_user.is_admin:
        abort(403)
    f = EvalFile.query.get_or_404(file_id)
    f.archived = False
    db.session.commit()
    flash(f"File '{display_filename(f.filename).upper()}' restored.", "success")
    return redirect(request.referrer or url_for("admin_dashboard"))


@app.route("/admin/file/<int:file_id>/delete", methods=["POST"])
@login_required
def admin_delete_file(file_id):
    if not current_user.is_admin:
        abort(403)
    f = EvalFile.query.get_or_404(file_id)
    name = display_filename(f.filename).upper()
    db.session.delete(f)
    db.session.commit()
    flash(f"File '{name}' deleted permanently.", "success")
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/file/<int:file_id>")
@login_required
def admin_file_view(file_id):
    if not current_user.is_admin:
        abort(403)
    f = EvalFile.query.get_or_404(file_id)
    samples = json.loads(f.samples_json)
    assignments = Assignment.query.filter_by(file_id=file_id).all()
    annotators = [a.user for a in assignments]
    anns_by_sample = {}
    for ann in Annotation.query.filter_by(file_id=file_id).all():
        key = (int(ann.narrative_number), ann.user_id)
        eval_dict = json.loads(ann.evaluation_json)
        anns_by_sample[key] = {
            "formatted": format_evaluation_display(eval_dict),
            "flagged": getattr(ann, "flagged", False),
            "ann_id": ann.id,
        }
    page = request.args.get("page", 1, type=int)
    total = len(samples)
    total_pages = (total + PER_PAGE - 1) // PER_PAGE if total else 1
    page = max(1, min(page, total_pages))
    start = (page - 1) * PER_PAGE
    end = start + PER_PAGE
    page_samples = samples[start:end]

    # Correlation for this file - use same annotators and annotations as Samples tab
    corr_method = request.args.get("method", "spearman")
    if corr_method not in ("pearson", "spearman", "kendall"):
        corr_method = "spearman"
    corr_results = []
    for a in assignments:
        anns = {
            int(ann.narrative_number): json.loads(ann.evaluation_json)
            for ann in Annotation.query.filter_by(file_id=file_id, user_id=a.user_id).all()
        }
        gpt_trust_a, gpt_trust_b = [], []
        ann_trust_a, ann_trust_b = [], []
        ann_f, ann_c, ann_h = [], [], []
        gpt_f, gpt_c, gpt_h = [], [], []
        for s in samples:
            n = s.get("narrative_number")
            n_key = int(n) if n is not None else None
            gpt = _parse_trust(s.get("gpt_trust"))
            ev = anns.get(n_key) if n_key is not None else None
            if gpt is None or not ev or not ev.get("trust"):
                continue
            t = ev["trust"]
            gpt_fch = _get_gpt_fch(s)
            gpt_trust_a.append(gpt[0])
            gpt_trust_b.append(gpt[1])
            ann_trust_a.append(t[0] if len(t) > 0 else 0)
            ann_trust_b.append(t[1] if len(t) > 1 else 0)
            ann_f.append(ev.get("fluency", 0))
            ann_c.append(ev.get("coherency", 0))
            ann_h.append(ev.get("hallucination", 0))
            gpt_f.append(gpt_fch[0] if gpt_fch[0] is not None else GPT_PLACEHOLDER_F)
            gpt_c.append(gpt_fch[1] if gpt_fch[1] is not None else GPT_PLACEHOLDER_C)
            gpt_h.append(gpt_fch[2] if gpt_fch[2] is not None else GPT_PLACEHOLDER_H)

        corr_trust_a = _correlation(gpt_trust_a, ann_trust_a, corr_method)
        corr_trust_b = _correlation(gpt_trust_b, ann_trust_b, corr_method)
        corr_f = _correlation(gpt_f, ann_f, corr_method)
        corr_c = _correlation(gpt_c, ann_c, corr_method)
        corr_h = _correlation(gpt_h, ann_h, corr_method)

        corr_results.append({
            "username": a.user.username,
            "n_annotated": len(ann_trust_a),
            "trust_a": corr_trust_a,
            "trust_b": corr_trust_b,
            "fluency": corr_f,
            "coherency": corr_c,
            "hallucination": corr_h,
        })

    # Annotator agreement (Cohen, Fleiss, Krippendorff)
    agreement_results = _compute_annotator_agreement(file_id, samples, assignments) if len(assignments) >= 2 else None

    return render_template(
        "admin_file_view.html",
        file=f,
        samples=page_samples,
        all_samples=samples,
        annotators=annotators,
        anns_by_sample=anns_by_sample,
        page=page,
        total_pages=total_pages,
        total=total,
        correlation_results=corr_results,
        correlation_method=corr_method,
        agreement_results=agreement_results,
    )


@app.route("/admin/file/<int:file_id>/unassign/<int:user_id>", methods=["POST"])
@login_required
def admin_unassign(file_id, user_id):
    if not current_user.is_admin:
        abort(403)
    a = Assignment.query.filter_by(file_id=file_id, user_id=user_id).first()
    if a:
        user = a.user
        db.session.delete(a)
        db.session.commit()
        flash(f"Removed {user.username} from assignment.", "success")
    return redirect(url_for("admin_assign", file_id=file_id))


@app.route("/admin/file/<int:file_id>/flag", methods=["POST"])
@login_required
def admin_toggle_flag(file_id):
    if not current_user.is_admin:
        abort(403)
    user_id = request.form.get("user_id", type=int)
    narrative_number = request.form.get("narrative_number", type=int)
    if not user_id or narrative_number is None:
        flash("Missing parameters.", "error")
        page = request.args.get("page", 1, type=int)
        return redirect(url_for("admin_file_view", file_id=file_id, page=page))
    ann = Annotation.query.filter_by(
        file_id=file_id, user_id=user_id, narrative_number=narrative_number
    ).first()
    if ann:
        ann.flagged = not ann.flagged
        db.session.commit()
        status = "flagged for revision" if ann.flagged else "unflagged"
        flash(f"Sample {narrative_number} {status}.", "success")
    page = request.args.get("page", 1, type=int)
    return redirect(url_for("admin_file_view", file_id=file_id, page=page))


@app.route("/admin/file/<int:file_id>/assign", methods=["GET", "POST"])
@login_required
def admin_assign(file_id):
    if not current_user.is_admin:
        abort(403)
    f = EvalFile.query.get_or_404(file_id)
    if request.method == "POST":
        user_id = request.form.get("user_id", type=int)
        if user_id:
            user = User.query.get(user_id)
            if user and not user.is_admin:
                existing = Assignment.query.filter_by(
                    file_id=file_id, user_id=user_id
                ).first()
                if not existing:
                    a = Assignment(file_id=file_id, user_id=user_id)
                    db.session.add(a)
                    db.session.commit()
                    flash(f"Assigned to {user.username}.", "success")
                else:
                    flash("Already assigned.", "info")
        return redirect(url_for("admin_assign", file_id=file_id))
    users = User.query.filter_by(is_admin=False).all()
    assigned = Assignment.query.filter_by(file_id=file_id).all()
    assigned_ids = {a.user_id for a in assigned}
    return render_template(
        "admin_assign.html",
        file=f,
        users=users,
        assigned_ids=assigned_ids,
        assigned=assigned,
    )


@app.route("/admin/file/<int:file_id>/edit/<int:user_id>/<int:narrative_number>", methods=["GET", "POST"])
@login_required
def admin_edit_annotation_page(file_id, user_id, narrative_number):
    if not current_user.is_admin:
        abort(403)
    f = EvalFile.query.get_or_404(file_id)
    user = User.query.get_or_404(user_id)
    samples = json.loads(f.samples_json)
    sample = next(
        (s for s in samples if int(s.get("narrative_number") or 0) == narrative_number),
        None,
    )
    if not sample:
        abort(404)
    ann = Annotation.query.filter_by(
        file_id=file_id,
        user_id=user_id,
        narrative_number=narrative_number,
    ).first()
    existing = json.loads(ann.evaluation_json) if ann else None

    if request.method == "POST":
        try:
            trust_n1 = int(request.form.get("trust_n1", 0))
            trust_n2 = int(request.form.get("trust_n2", 0))
            fluency = int(request.form.get("fluency", 1))
            coherency = int(request.form.get("coherency", 1))
            hallucination = int(request.form.get("hallucination", 1))
            if fluency < 1 or fluency > 5 or coherency < 1 or coherency > 5 or hallucination < 1 or hallucination > 5:
                flash("Fluency, Coherency, and Hallucination must be 1-5.", "error")
                return redirect(request.url)
            eval_dict = {"trust": [trust_n1, trust_n2], "fluency": fluency, "coherency": coherency, "hallucination": hallucination}
            if ann:
                ann.evaluation_json = json.dumps(eval_dict)
            else:
                ann = Annotation(
                    file_id=file_id,
                    user_id=user_id,
                    narrative_number=narrative_number,
                    evaluation_json=json.dumps(eval_dict),
                )
                db.session.add(ann)
            db.session.commit()
            flash("Annotation updated.", "success")
        except (ValueError, TypeError):
            flash("Invalid input.", "error")
        return redirect(url_for("admin_file_view", file_id=file_id))

    return render_template(
        "admin_edit_annotation.html",
        file=f,
        user=user,
        sample=sample,
        existing=existing,
    )


@app.route("/admin/file/<int:file_id>/edit_annotation", methods=["POST"])
@login_required
def admin_edit_annotation(file_id):
    if not current_user.is_admin:
        abort(403)
    user_id = request.form.get("user_id", type=int)
    narrative_number = request.form.get("narrative_number", type=int)
    trust_n1 = request.form.get("trust_n1", type=int)
    trust_n2 = request.form.get("trust_n2", type=int)
    fluency = request.form.get("fluency", type=int)
    coherency = request.form.get("coherency", type=int)
    if user_id is None or narrative_number is None:
        flash("Missing parameters.", "error")
        return redirect(url_for("admin_file_view", file_id=file_id))
    if fluency and (fluency < 1 or fluency > 5) or coherency and (coherency < 1 or coherency > 5):
        flash("Fluency and Coherency must be 1-5.", "error")
        return redirect(url_for("admin_file_view", file_id=file_id))
    eval_dict = {
        "trust": [trust_n1 or 0, trust_n2 or 0],
        "fluency": fluency or 1,
        "coherency": coherency or 1,
    }
    ann = Annotation.query.filter_by(
        file_id=file_id,
        user_id=user_id,
        narrative_number=narrative_number,
    ).first()
    if ann:
        ann.evaluation_json = json.dumps(eval_dict)
    else:
        ann = Annotation(
            file_id=file_id,
            user_id=user_id,
            narrative_number=narrative_number,
            evaluation_json=json.dumps(eval_dict),
        )
        db.session.add(ann)
    db.session.commit()
    flash("Annotation updated.", "success")
    return redirect(url_for("admin_file_view", file_id=file_id))


@app.route("/admin/file/<int:file_id>/download")
@login_required
def admin_download(file_id):
    if not current_user.is_admin:
        abort(403)
    f = EvalFile.query.get_or_404(file_id)
    samples = json.loads(f.samples_json)
    assignments = Assignment.query.filter_by(file_id=file_id).all()
    # Preserve all original columns from upload; add annotator columns at the end
    base_cols = list(samples[0].keys()) if samples else []
    annotator_names = [a.user.username for a in assignments]
    rows = []
    for s in samples:
        row = {c: s.get(c) for c in base_cols}
        for a in assignments:
            ann = Annotation.query.filter_by(
                file_id=file_id,
                user_id=a.user_id,
                narrative_number=s.get("narrative_number"),
            ).first()
            if ann:
                eval_dict = json.loads(ann.evaluation_json)
                row[a.user.username] = format_evaluation_display(eval_dict)
            else:
                row[a.user.username] = ""
        rows.append(row)
    df = pd.DataFrame(rows)
    output = BytesIO()
    df.to_excel(output, index=False, engine="openpyxl")
    output.seek(0)
    base_name = os.path.splitext(f.filename)[0]
    out_name = f"{base_name}_annotations.xlsx"
    return send_file(
        output,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=out_name,
    )


# ---------- Init DB and create admin ----------


@app.cli.command("init-db")
def init_db():
    db.create_all()
    if not User.query.filter_by(username="admin").first():
        admin = User(username="admin", is_admin=True)
        admin.set_password("admin")
        db.session.add(admin)
        db.session.commit()
        print("Created admin user: admin / admin")
    print("Database initialized.")


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(username="admin").first():
            admin = User(username="admin", is_admin=True)
            admin.set_password("admin")
            db.session.add(admin)
            db.session.commit()
            print("Created admin user: admin / admin")
    app.run(debug=True, port=5000)
