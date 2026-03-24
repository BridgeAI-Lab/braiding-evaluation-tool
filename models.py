from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import json

db = SQLAlchemy()


class User(UserMixin, db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    last_login_at = db.Column(db.DateTime, nullable=True)
    last_activity_at = db.Column(db.DateTime, nullable=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class EvalFile(db.Model):
    __tablename__ = "eval_files"
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(256), nullable=False)
    uploaded_at = db.Column(db.DateTime, server_default=db.func.now())
    archived = db.Column(db.Boolean, default=False, nullable=False)
    # JSON: list of dicts with keys narrative_number, dataset, Review1, Review2, braided_narrative, gt_trust, gpt_trust
    samples_json = db.Column(db.Text, nullable=False)
    assignments = db.relationship("Assignment", backref="eval_file", cascade="all, delete-orphan")
    annotations = db.relationship("Annotation", backref="eval_file", cascade="all, delete-orphan")


class Assignment(db.Model):
    __tablename__ = "assignments"
    __table_args__ = (db.UniqueConstraint("file_id", "user_id", name="uq_file_user"),)
    id = db.Column(db.Integer, primary_key=True)
    file_id = db.Column(db.Integer, db.ForeignKey("eval_files.id"), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    user = db.relationship("User", backref="assignments")


class Annotation(db.Model):
    __tablename__ = "annotations"
    __table_args__ = (
        db.UniqueConstraint("file_id", "user_id", "narrative_number", name="uq_file_user_narrative"),
    )
    id = db.Column(db.Integer, primary_key=True)
    file_id = db.Column(db.Integer, db.ForeignKey("eval_files.id"), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    user = db.relationship("User", backref="annotations")
    narrative_number = db.Column(db.Integer, nullable=False)  # sample index
    # JSON: {"trust": [70, 30], "fluency": 4, "coherency": 5, "hallucination": 4}
    evaluation_json = db.Column(db.Text, nullable=False)
    flagged = db.Column(db.Boolean, default=False, nullable=False)  # admin flag for revision


class AppSetting(db.Model):
    __tablename__ = "app_settings"
    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(64), unique=True, nullable=False)
    value = db.Column(db.Text, nullable=True)


def get_guidelines():
    """Get guidelines HTML from DB, or return default."""
    s = AppSetting.query.filter_by(key="guidelines").first()
    if s and s.value:
        return s.value
    return _default_guidelines()


def restore_default_guidelines():
    """Reset guidelines to built-in default."""
    save_guidelines(_default_guidelines())


def save_guidelines(html_content):
    """Save guidelines HTML to DB."""
    s = AppSetting.query.filter_by(key="guidelines").first()
    if s:
        s.value = html_content
    else:
        s = AppSetting(key="guidelines", value=html_content)
        db.session.add(s)
    db.session.commit()


def _default_guidelines():
    return """<h6 class="fw-bold mt-2">Human Evaluation Task: Trust Allocation in a Braided Narrative</h6>
<h6 class="fw-semibold mt-3">Overview</h6>
<p>You will be given:</p>
<ul>
<li><strong>Narrative A (N_A)</strong></li>
<li><strong>Narrative B (N_B)</strong></li>
<li>A <strong>Braided Narrative</strong> synthesized from A and B</li>
</ul>
<p>Your task is to determine how much trust the braided narrative places in each original narrative.</p>
<p><strong>Trust</strong> means: How much the braided narrative appears to rely on, validate, or favor one narrative over the other.</p>
<ul>
<li>You are <em>not</em> judging which narrative is correct.</li>
<li>You are only inferring perceived trust based on how the braided narrative uses them.</li>
</ul>

<h6 class="fw-semibold mt-4">🔍 How to Infer Trust</h6>
<p>You must consider both content selection and interpretation.</p>

<p class="mb-1"><strong>1️⃣ Content Selection (What is included?)</strong></p>
<p>Examine:</p>
<ul>
<li>Which narrative's unique information appears in the braided version</li>
<li>Which narrative's conflicting claims are included</li>
<li>How much UNIQUE + CONFLICT content is taken from each narrative</li>
<li>Whether one narrative's distinctive content is mostly excluded</li>
</ul>
<p class="text-warning"><strong>⚠️ Important:</strong> Overlap (shared information) does NOT count toward trust. Only UNIQUE and CONFLICT clauses count.</p>

<p class="mb-1"><strong>2️⃣ Interpretation & Resolution (How is it framed?)</strong></p>
<p>Examine:</p>
<ul>
<li>Which narrative's claims are emphasized or validated</li>
<li>Which narrative's critiques are strengthened or softened</li>
<li>When there is a conflict, which narrative is favored</li>
<li>Whether the conclusion aligns more closely with one narrative</li>
</ul>
<p>Tone and final positioning matter.</p>

<h6 class="fw-semibold mt-4">📏 Content–Trust Consistency Rule</h6>
<p>Your trust scores must reflect the proportion of UNIQUE + CONFLICT content used.</p>
<p><strong>Example:</strong></p>
<ul>
<li>If about 70% of selected unique/conflict content comes from Narrative A → Trust in A ≈ 70</li>
<li>If about 30% comes from Narrative B → Trust in B ≈ 30</li>
</ul>
<p>The scores should approximately match the content distribution.</p>

<p class="mb-1"><strong>50–50 Case</strong></p>
<p>Assign 50–50 when:</p>
<ul>
<li>UNIQUE + CONFLICT content from both narratives is roughly equal</li>
<li>Conflicts are treated symmetrically</li>
<li>Neither narrative is clearly favored in interpretation or conclusion</li>
</ul>
<p>Balanced inclusion + balanced framing = 50 / 50 trust.</p>

<h6 class="fw-semibold mt-4">🔢 Trust Scoring Instructions</h6>
<ul>
<li>Assign an integer score between 0 and 100 to each narrative</li>
<li>The two scores must sum to exactly 100</li>
<li>Interpret scores as relative confidence</li>
<li>Format: Trust in Narrative A: ___ | Trust in Narrative B: ___</li>
</ul>

<h6 class="fw-semibold mt-4">✍️ Additional Quality Evaluation (Likert Scale)</h6>
<p>In addition to trust allocation, evaluate the braided narrative on the following dimensions using a 5-point Likert scale.</p>

<p class="mb-1"><strong>1️⃣ Fluency</strong></p>
<p>How grammatically correct and linguistically smooth is the braided narrative?</p>
<ul>
<li>1 – Very poor (frequent grammatical errors, difficult to read)</li>
<li>2 – Poor</li>
<li>3 – Acceptable</li>
<li>4 – Good</li>
<li>5 – Excellent (natural and fluent)</li>
</ul>

<p class="mb-1"><strong>2️⃣ Coherence</strong></p>
<p>How logically organized and internally consistent is the braided narrative?</p>
<ul>
<li>1 – Very incoherent (disorganized, contradictory)</li>
<li>2 – Weak coherence</li>
<li>3 – Moderately coherent</li>
<li>4 – Mostly coherent</li>
<li>5 – Highly coherent and well-structured</li>
</ul>

<p class="mb-1"><strong>3️⃣ Hallucination</strong></p>
<p>Does the braided narrative introduce information not supported by either Narrative A or Narrative B?</p>
<ul>
<li>1 – Severe hallucination (many unsupported claims)</li>
<li>2 – Noticeable hallucination</li>
<li>3 – Minor unsupported additions</li>
<li>4 – Mostly faithful to source narratives</li>
<li>5 – Fully faithful (no hallucinated content)</li>
</ul>"""


def format_evaluation_display(eval_dict):
    """Convert stored dict to display format: [70, 30] | F:4 | C:5 | H:4"""
    if not eval_dict:
        return ""
    t = eval_dict.get("trust", [0, 0])
    f = eval_dict.get("fluency", 0)
    c = eval_dict.get("coherency", 0)
    h = eval_dict.get("hallucination", 0)
    return f"[{t[0]},{t[1]}]|F:{f}|C:{c}|H:{h}"
