Yesss, we can absolutely sprinkle in some NLP / ML so it doesn’t look “only prompts” 😎

Below is a single extended Python file that:

Keeps the AURIX-mini pipeline (memory → rewriter → planner → retrieval → reconcile → compliance → explainer).

Adds an NLPEngine that does:

Lightweight ML-ish intent detection (TF-IDF + LogisticRegression) if scikit-learn is available.

Fallback keyword-based intent detection if not.

Simple emotion / sentiment scoring using a tiny lexicon.


Feeds these nlp_signals into the Memory Manager and Query Rewriter prompts, so LLM sees both:

“User seems frustrated / urgent / neutral”

“ML suggests intent = audit_invoice_ledger / policy_explanation / generic”



You still only need to change call_llm() to make it work with Copilot/whatever.


---

"""
AURIX-mini-X: Prompt-Layered Audit & Compliance Engine
with lightweight NLP/ML (intent + emotion) on top.

- Works with mock data only.
- All reasoning steps are LLM prompts (suitable for Copilot).
- Adds NLPEngine for:
    * Emotion / sentiment scoring
    * ML-ish intent detection (TF-IDF + LogisticRegression if available)
- Uses these NLP signals to enrich prompts at Query Rewriter & Memory Manager.

NOTE:
- call_llm() is a stub: replace with actual LLM integration or mock.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json
import textwrap
import uuid
import datetime

# ---------------------------------------------------------------------------
# Optional ML dependencies (we gracefully fall back if not installed)
# ---------------------------------------------------------------------------
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
except ImportError:
    TfidfVectorizer = None
    LogisticRegression = None


# ---------------------------------------------------------------------------
# 0. Mock Data (acts like in-memory DB)
# ---------------------------------------------------------------------------

MOCK_DATA: Dict[str, Any] = {
    "invoices": [
        {
            "id": "INV-1032",
            "vendor": "Apex Technologies Pvt Ltd",
            "amount": 2450.0,
            "currency": "USD",
            "description": "Annual cloud service subscription renewal for production servers",
            "date": "2025-03-01",
            "has_approval_note": False
        },
        {
            "id": "INV-2001",
            "vendor": "Delta Infotech Solutions",
            "amount": 1200.0,
            "currency": "USD",
            "description": "Software maintenance and license renewal",
            "date": "2025-02-20",
            "has_approval_note": True
        }
    ],
    "ledgers": [
        {
            "id": "LEDG-231",
            "vendor": "Apex Tech Private Limited",
            "amount": 2435.0,
            "currency": "USD",
            "description": "Cloud infrastructure cost for annual subscription",
            "date": "2025-02-27"
        },
        {
            "id": "LEDG-233",
            "vendor": "Apex Technologies Pvt Ltd",
            "amount": 5000.0,
            "currency": "USD",
            "description": "Hardware procurement for data center expansion",
            "date": "2025-02-10"
        },
        {
            "id": "LEDG-400",
            "vendor": "Delta Infotech Solutions",
            "amount": 1200.0,
            "currency": "USD",
            "description": "Software maintenance and license renewal",
            "date": "2025-02-19"
        }
    ],
    "policies": [
        {
            "id": "POL-01",
            "rule": "Cloud service renewals above $2000 must include a documented IT approval note.",
            "category": "Cloud",
            "severity": "HIGH"
        },
        {
            "id": "POL-02",
            "rule": "Hardware procurement above $4000 must be split into multiple POs for risk control.",
            "category": "Hardware",
            "severity": "MEDIUM"
        },
        {
            "id": "POL-03",
            "rule": "All vendor payments must match within ±3% of approved ledger amounts.",
            "category": "Finance",
            "severity": "HIGH"
        }
    ]
}


# ---------------------------------------------------------------------------
# 1. Conversation State / Memory
# ---------------------------------------------------------------------------

@dataclass
class Message:
    role: str  # "user" or "system"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())


@dataclass
class ConversationState:
    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    history: List[Message] = field(default_factory=list)
    summary: str = ""
    topic_threshold: float = 0.3  # threshold for topic similarity

    def add_user_message(self, text: str) -> None:
        self.history.append(Message(role="user", content=text))

    def add_system_message(self, text: str) -> None:
        self.history.append(Message(role="system", content=text))

    def last_k_messages(self, k: int = 5) -> List[Dict[str, Any]]:
        return [
            {"role": m.role, "content": m.content, "timestamp": m.timestamp}
            for m in self.history[-k:]
        ]


# ---------------------------------------------------------------------------
# 2. LLM Call Stub (replace with Copilot / Azure / etc.)
# ---------------------------------------------------------------------------

def call_llm(prompt: str, expect_json: bool = True) -> Any:
    """
    Core abstraction: all nodes send prompts through this function.

    For the codeathon:
    - You can:
        - Print the prompt
        - Paste into Copilot
        - Paste JSON/text back into a mocked version of this function
          OR wire an actual LLM API.

    CURRENT: stub that raises.
    """
    print("\n" + "=" * 80)
    print("LLM PROMPT:")
    print("=" * 80)
    print(prompt)
    print("=" * 80)
    raise NotImplementedError(
        "call_llm() is a stub. Replace with actual LLM integration or mock outputs."
    )


# ---------------------------------------------------------------------------
# 3. NLPEngine: lightweight intent + emotion layer
# ---------------------------------------------------------------------------

class NLPEngine:
    """
    Adds some ML / NLP flavor:
    - Emotion / sentiment detection via tiny lexicon.
    - Intent detection via:
        * small TF-IDF + LogisticRegression classifier if sklearn is available
        * otherwise, keyword-based classifier.

    This runs locally, no LLM needed.
    """

    def __init__(self) -> None:
        self.available_ml = TfidfVectorizer is not None and LogisticRegression is not None

        # Very tiny lexicon for emotion/sentiment
        self.positive_words = {
            "thanks", "thank you", "great", "good", "nice", "clear", "helpful"
        }
        self.negative_words = {
            "confused", "bad", "issue", "problem", "error", "frustrated", "angry", "upset"
        }
        self.urgent_words = {
            "urgent", "asap", "immediately", "now", "right away"
        }

        # Optional ML model
        if self.available_ml:
            self._init_ml_intent_model()
        else:
            self.vectorizer = None
            self.intent_clf = None

    def _init_ml_intent_model(self) -> None:
        """
        Train a tiny toy classifier just to show ML integration.
        Real system would load a pre-trained model instead.
        """
        examples = [
            "match this invoice with ledger and check any mismatch",
            "reconcile vendor payments against the ledger",
            "find anomalies in invoice and ledger amounts",
            "explain which policy is being violated here",
            "what policies apply to this transaction",
            "tell me if this is compliant with policy",
            "hi", "hello", "can you help me understand this",
            "what does aurix do", "explain the system"
        ]
        labels = [
            "audit_invoice_ledger",  # 0
            "audit_invoice_ledger",
            "anomaly_investigation",
            "policy_explanation",
            "policy_explanation",
            "policy_explanation",
            "generic_query",
            "generic_query",
            "generic_query",
            "generic_query",
            "generic_query",
        ]

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        X = self.vectorizer.fit_transform(examples)
        self.intent_clf = LogisticRegression(max_iter=1000)
        self.intent_clf.fit(X, labels)

    def detect_intent(self, text: str) -> Dict[str, Any]:
        """
        Returns:
        {
          "intent_ml": "...",
          "confidence_ml": float (0-1, rough),
          "intent_rule": "...",
        }
        """
        text_lower = text.lower()

        # Rule-based quick label
        if any(word in text_lower for word in ["match", "reconcile", "ledger", "invoice"]):
            intent_rule = "audit_invoice_ledger"
        elif any(word in text_lower for word in ["policy", "compliant", "violate", "violation"]):
            intent_rule = "policy_explanation"
        elif any(word in text_lower for word in ["anomaly", "suspicious", "weird", "strange"]):
            intent_rule = "anomaly_investigation"
        else:
            intent_rule = "generic_query"

        # ML-based label if available
        if self.available_ml and self.vectorizer is not None and self.intent_clf is not None:
            X = self.vectorizer.transform([text])
            proba = self.intent_clf.predict_proba(X)[0]
            classes = list(self.intent_clf.classes_)
            best_idx = int(proba.argmax())
            intent_ml = classes[best_idx]
            confidence_ml = float(proba[best_idx])
        else:
            intent_ml = intent_rule
            confidence_ml = 0.5  # neutral confidence when we don't have ML

        return {
            "intent_ml": intent_ml,
            "confidence_ml": confidence_ml,
            "intent_rule": intent_rule,
        }

    def detect_emotion(self, text: str) -> Dict[str, Any]:
        """
        Very simple lexicon-based: counts positive/negative/urgent keywords.
        """
        t = text.lower()
        pos = sum(1 for w in self.positive_words if w in t)
        neg = sum(1 for w in self.negative_words if w in t)
        urg = sum(1 for w in self.urgent_words if w in t)

        if neg > pos and neg > 0:
            emotion = "negative"
        elif pos > neg and pos > 0:
            emotion = "positive"
        else:
            emotion = "neutral"

        urgency = "high" if urg > 0 else "normal"

        # crude sentiment score
        sentiment_score = (pos - neg) / max(pos + neg, 1)

        return {
            "emotion": emotion,
            "urgency": urgency,
            "sentiment_score": sentiment_score,
            "positive_hits": pos,
            "negative_hits": neg,
            "urgent_hits": urg,
        }

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Combined NLP output, ready to feed into prompts.
        """
        intent_info = self.detect_intent(text)
        emotion_info = self.detect_emotion(text)
        return {
            "intent_ml": intent_info["intent_ml"],
            "intent_rule": intent_info["intent_rule"],
            "intent_confidence": intent_info["confidence_ml"],
            "emotion": emotion_info["emotion"],
            "urgency": emotion_info["urgency"],
            "sentiment_score": emotion_info["sentiment_score"],
        }


# ---------------------------------------------------------------------------
# 4. Memory Manager Node (topic shift + folding, now with NLP signals)
# ---------------------------------------------------------------------------

MEMORY_MANAGER_PROMPT = """
You are the Memory Manager node inside the AURIX conversation engine.

Your job:
1. Look at recent messages (history), the new user message, and NLP signals about that message.
2. Decide if the new message belongs to the SAME topic as the previous ones,
   or if it starts a NEW topic.
3. If it is a new topic, we should "reset" short-term history for reasoning.
4. Otherwise, we "keep" history.
5. Maintain a short natural-language summary of the important audit context so far.
6. Take into account the user's emotion/urgency to keep relevant context if needed.

Inputs JSON:
{input_json}

You MUST respond with STRICT JSON:
{
  "topic_similarity": 0.0 to 1.0,
  "memory_action": "keep" or "reset",
  "updated_summary": "1-3 line textual summary capturing key audit context"
}
"""

def memory_manager_node(
    state: ConversationState,
    new_user_message: str,
    nlp_signals: Dict[str, Any]
) -> Dict[str, Any]:
    payload = {
        "history_messages": state.last_k_messages(k=6),
        "previous_summary": state.summary,
        "new_message": new_user_message,
        "nlp_signals": nlp_signals  # includes emotion, urgency, ml intent, etc.
    }
    prompt = MEMORY_MANAGER_PROMPT.format(
        input_json=json.dumps(payload, indent=2)
    )
    raw = call_llm(prompt, expect_json=True)
    result = json.loads(raw)

    state.summary = result.get("updated_summary", state.summary)
    return result


# ---------------------------------------------------------------------------
# 5. Query Rewriter Node (now seeing NLP signals)
# ---------------------------------------------------------------------------

QUERY_REWRITER_PROMPT = """
You are the Query Rewriter node inside the AURIX audit engine.

Your job:
- Normalize/clarify the user's query.
- Use memory summary if it is relevant and memory_action = "keep".
- Use NLP signals (ML intent suggestion, emotion, urgency) to refine the intent.
- Extract key entities.
- Decide if this is a follow-up or standalone question.

Inputs JSON:
{input_json}

You MUST respond with STRICT JSON using exactly this schema:
{
  "clarified_query": "natural language restatement of the user's question",
  "intent": "one of: audit_invoice_ledger, policy_explanation, anomaly_investigation, generic_query",
  "entities": {
    "invoice_id": "string or null",
    "ledger_id": "string or null",
    "vendor": "string or null",
    "date_range": "string or null",
    "amount_range": "string or null"
  },
  "conversation": {
    "is_followup": true or false,
    "reason": "short explanation",
    "memory_action": "keep" or "reset"
  },
  "emotion": {
    "label": "positive|neutral|negative",
    "urgency": "high|normal",
    "comment": "how this might affect explanation style"
  }
}
"""

def query_rewriter_node(
    user_query: str,
    memory_info: Dict[str, Any],
    state: ConversationState,
    nlp_signals: Dict[str, Any]
) -> Dict[str, Any]:
    input_payload = {
        "user_query": user_query,
        "memory_info": memory_info,
        "memory_summary": state.summary,
        "nlp_signals": nlp_signals
    }
    prompt = QUERY_REWRITER_PROMPT.format(
        input_json=json.dumps(input_payload, indent=2)
    )
    raw = call_llm(prompt, expect_json=True)
    result = json.loads(raw)
    return result


# ---------------------------------------------------------------------------
# 6. Query Planner Node
# ---------------------------------------------------------------------------

QUERY_PLANNER_PROMPT = """
You are the Query Planner node for AURIX.

You receive the rewritten query and must decide which stages to run.

Possible step types:
- "retrieve"      : fetch relevant invoices, ledgers, policies
- "reconcile"     : match invoices <-> ledgers
- "policy_check"  : evaluate reconciled items against policies
- "explain"       : generate human-readable explanation

Input (rewriter output):
{rewriter_json}

You MUST respond with STRICT JSON like:
{
  "steps": [
    {"id": "step1", "type": "retrieve"},
    {"id": "step2", "type": "reconcile"},
    {"id": "step3", "type": "policy_check"},
    {"id": "step4", "type": "explain"}
  ]
}
"""

def query_planner_node(rewriter_output: Dict[str, Any]) -> Dict[str, Any]:
    prompt = QUERY_PLANNER_PROMPT.format(
        rewriter_json=json.dumps(rewriter_output, indent=2)
    )
    raw = call_llm(prompt, expect_json=True)
    result = json.loads(raw)
    return result


# ---------------------------------------------------------------------------
# 7. Retrieval Agent Node (with lexical + semantic similarity math)
# ---------------------------------------------------------------------------

RETRIEVAL_PROMPT = r"""
You are the Retrieval & Similarity Agent inside AURIX.

Goal:
- From the following invoices, ledgers, and policies,
  select the most relevant records for the clarified query.
- You must use a combination of lexical and semantic similarity.

Mathematical similarity (for explanation to the user, you don't need to compute exact numbers):
1. Let tokens(x) be the set of important words in text x.
2. Define lexical Jaccard similarity:
   S_lex(x, q') = |tokens(x) ∩ tokens(q')| / |tokens(x) ∪ tokens(q')|.
3. Let S_sem(x, q') ∈ [0,1] be your semantic similarity judgment based on meaning.
4. Overall similarity is:
   S_total(x, q') = α * S_lex(x, q') + β * S_sem(x, q'),
   where α and β are positive and α + β = 1 (for example, α=0.4, β=0.6).

You must conceptually follow this formula when you score records.

Inputs:
- Clarified query:
{rewriter_json}

- Invoices:
{invoices_json}

- Ledgers:
{ledgers_json}

- Policies:
{policies_json}

Your tasks:
1. Compute S_total for each invoice and each ledger against the clarified query.
2. Pick the top 3 invoices and the top 3 ledgers by S_total.
3. Select the policies that are most relevant for this query (explain why).
4. For each selected record, provide a score and a short reason.

Respond with STRICT JSON:
{
  "selected_invoices": [
    {
      "id": "INV-1032",
      "similarity": 0.93,
      "reason": "short natural language justification"
    }
  ],
  "selected_ledgers": [
    {
      "id": "LEDG-231",
      "similarity": 0.89,
      "reason": "short natural language justification"
    }
  ],
  "selected_policies": [
    {
      "id": "POL-01",
      "relevance_reason": "short reason"
    }
  ]
}
"""

def retrieval_agent_node(
    rewriter_output: Dict[str, Any],
    data: Dict[str, Any]
) -> Dict[str, Any]:
    prompt = RETRIEVAL_PROMPT.format(
        rewriter_json=json.dumps(rewriter_output, indent=2),
        invoices_json=json.dumps(data["invoices"], indent=2),
        ledgers_json=json.dumps(data["ledgers"], indent=2),
        policies_json=json.dumps(data["policies"], indent=2),
    )
    raw = call_llm(prompt, expect_json=True)
    result = json.loads(raw)
    return result


# ---------------------------------------------------------------------------
# 8. Reconciliation Agent Node
# ---------------------------------------------------------------------------

RECONCILIATION_PROMPT = """
You are the Reconciliation Agent inside AURIX.

Your job:
- For each selected invoice, try to find the best matching ledger.
- Decide status:
    - "match"            : vendor is compatible and amount difference ≤ 3%
    - "mismatch_amount"  : vendor is compatible but amount difference > 3%
    - "mismatch_vendor"  : vendor appears to be a different entity
    - "no_match"         : no reasonable ledger found
- Use amount_diff_percent = |A_invoice - A_ledger| / A_invoice * 100.

Inputs:
- Selected invoices:
{selected_invoices}

- Full invoice records:
{invoice_records}

- Selected ledgers:
{selected_ledgers}

- Full ledger records:
{ledger_records}

Respond with STRICT JSON:
{
  "reconciliations": [
    {
      "invoice_id": "INV-1032",
      "ledger_id": "LEDG-231",
      "status": "match",
      "amount_diff_percent": 0.6,
      "vendor_comment": "short note on vendor name similarity or difference",
      "notes": "short explanation of why this status was chosen"
    }
  ],
  "unmatched_invoices": ["..."],
  "unmatched_ledgers": ["..."]
}
"""

def reconciliation_agent_node(
    retrieval_output: Dict[str, Any],
    data: Dict[str, Any]
) -> Dict[str, Any]:
    prompt = RECONCILIATION_PROMPT.format(
        selected_invoices=json.dumps(retrieval_output.get("selected_invoices", []), indent=2),
        invoice_records=json.dumps(data["invoices"], indent=2),
        selected_ledgers=json.dumps(retrieval_output.get("selected_ledgers", []), indent=2),
        ledger_records=json.dumps(data["ledgers"], indent=2),
    )
    raw = call_llm(prompt, expect_json=True)
    result = json.loads(raw)
    return result


# ---------------------------------------------------------------------------
# 9. Compliance Agent Node
# ---------------------------------------------------------------------------

COMPLIANCE_PROMPT = """
You are the Compliance Agent inside AURIX.

Your job:
- Evaluate the reconciled invoice/ledger pairs against the selected policies.
- Identify policy violations, with reasoning.
- Derive an overall risk level: "LOW", "MEDIUM", or "HIGH".

Inputs:
- Reconciliations:
{recon_json}

- Selected policies:
{selected_policies}

- Full invoice records (for context like has_approval_note):
{invoice_records}

- Full ledger records:
{ledger_records}

Please:
1. For each policy and each relevant invoice/ledger pair, decide if it is violated.
2. Provide a short reason.
3. Compute an overall risk level based on severity of violated policies and number of issues.

Respond with STRICT JSON:
{
  "policy_evaluations": [
    {
      "policy_id": "POL-01",
      "applies_to": ["INV-1032"],
      "is_violated": true,
      "violation_reason": "short text",
      "severity": "HIGH"
    }
  ],
  "overall_risk_level": "LOW" or "MEDIUM" or "HIGH",
  "summary": "1-3 line overall compliance summary"
}
"""

def compliance_agent_node(
    recon_output: Dict[str, Any],
    retrieval_output: Dict[str, Any],
    data: Dict[str, Any]
) -> Dict[str, Any]:
    selected_policy_ids = {p["id"] for p in retrieval_output.get("selected_policies", [])}
    selected_policies = [p for p in data["policies"] if p["id"] in selected_policy_ids]

    prompt = COMPLIANCE_PROMPT.format(
        recon_json=json.dumps(recon_output, indent=2),
        selected_policies=json.dumps(selected_policies, indent=2),
        invoice_records=json.dumps(data["invoices"], indent=2),
        ledger_records=json.dumps(data["ledgers"], indent=2),
    )
    raw = call_llm(prompt, expect_json=True)
    result = json.loads(raw)
    return result


# ---------------------------------------------------------------------------
# 10. Explainer Agent Node (final user-facing summary)
# ---------------------------------------------------------------------------

EXPLAINER_PROMPT = """
You are the Explainer Agent inside AURIX.

Goal:
- Produce a clear, concise, human-readable audit summary for the user.
- It should feel like an auditor wrote it.
- If emotion.urgency is "high" or emotion.label is "negative", be extra clear and empathetic.

Inputs:
- Original user query:
{user_query}

- Clarified query (rewriter output):
{rewriter_json}

- Reconciliation result:
{recon_json}

- Compliance result:
{compliance_json}

Write a Markdown report with sections:

1. **Normalized Question**
2. **Records Considered** (mention key invoice / ledger IDs)
3. **Reconciliation Summary** (matches, mismatches)
4. **Policy Evaluation** (what was checked, what failed)
5. **Final Risk Assessment** (LOW / MEDIUM / HIGH, with brief justification)

Keep it under 300 words.
"""

def explainer_agent_node(
    user_query: str,
    rewriter_output: Dict[str, Any],
    recon_output: Dict[str, Any],
    compliance_output: Dict[str, Any]
) -> str:
    prompt = EXPLAINER_PROMPT.format(
        user_query=user_query,
        rewriter_json=json.dumps(rewriter_output, indent=2),
        recon_json=json.dumps(recon_output, indent=2),
        compliance_json=json.dumps(compliance_output, indent=2),
    )
    raw = call_llm(prompt, expect_json=False)  # expect Markdown text
    return raw


# ---------------------------------------------------------------------------
# 11. Orchestration: one conversational turn of AURIX-mini-X
# ---------------------------------------------------------------------------

def run_aurix_turn(
    user_query: str,
    state: ConversationState,
    nlp_engine: NLPEngine,
    data: Optional[Dict[str, Any]] = None
) -> str:
    """
    Run a full AURIX-mini-X pipeline for one user query, using:
    - Local NLP engine (intent + emotion)
    - Prompt-layered agents.

    Steps:
    0. Local NLP analysis (no LLM)
    1. Memory manager: topic shift + summary update
    2. Query rewriter (uses NLP signals)
    3. Planner
    4. Retrieval (similarity-based with math)
    5. Reconciliation
    6. Compliance
    7. Explainer

    Returns:
        Markdown string (final user-facing report).
    """
    if data is None:
        data = MOCK_DATA

    # Add user message into history
    state.add_user_message(user_query)

    # 0) Local NLP / ML
    nlp_signals = nlp_engine.analyze(user_query)

    # 1) Memory management
    memory_info = memory_manager_node(state, user_query, nlp_signals)

    # 2) Query rewriting (uses memory + NLP)
    rewriter_output = query_rewriter_node(user_query, memory_info, state, nlp_signals)

    # 3) Planner
    planner_output = query_planner_node(rewriter_output)
    # For now we always execute all stages in sequence.

    # 4) Retrieval
    retrieval_output = retrieval_agent_node(rewriter_output, data)

    # 5) Reconciliation
    recon_output = reconciliation_agent_node(retrieval_output, data)

    # 6) Compliance
    compliance_output = compliance_agent_node(recon_output, retrieval_output, data)

    # 7) Explainer
    final_report = explainer_agent_node(
        user_query=user_query,
        rewriter_output=rewriter_output,
        recon_output=recon_output,
        compliance_output=compliance_output,
    )

    # Add system message (for chat history)
    state.add_system_message(final_report)
    return final_report


# ---------------------------------------------------------------------------
# 12. Example usage (for local testing)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Example usage.

    IMPORTANT:
    - call_llm() is not implemented and will raise.
    - For the codeathon demo, you can:
        * mock call_llm(), OR
        * integrate an actual LLM, OR
        * print prompts and manually paste into Copilot.
    """
    state = ConversationState()
    nlp_engine = NLPEngine()

    example_query = (
        "I'm a bit confused, can you urgently check if the Apex Technologies "
        "cloud renewal invoice is correctly matched in the ledger and whether "
        "any policies are being violated?"
    )

    final_markdown = run_aurix_turn(example_query, state, nlp_engine)
    print("\n=== FINAL AURIX REPORT ===\n")
    print(final_markdown)


---

If you want, I can next:

Write a short “Design doc” paragraph explaining:
“We combine local ML (NLPEngine) with LLM-based prompt layers to capture user emotion & intent before planning the audit pipeline.”

Or give you a standup update / PPT bullet points describing the “NLP + prompt-layer AURIX” story in nice BoFA-style tech language.