import sys
from pathlib import Path
# Add parent directory to path to import from utils
sys.path.append(str(Path(__file__).parent.parent))

import json, re, os
from datetime import datetime
from dateutil import parser as dateparser
from typing import Any, Dict, List, Optional
import streamlit as st
from pydantic import BaseModel, EmailStr, ValidationError
from langchain.schema import HumanMessage

# Import from llm_models_openrouter instead of llm_models to use OpenRouter models
from utils.llm_models_openrouter import get_chat_llm

# Define the llm model to use - see: llm_models_openrouter.py
LLM_MODEL = "google/gemini-2.0-flash-001"

# --------- Config ----------

QUESTIONS_PATH = Path(__file__).parent.parent / "tools" / "questions.json"


# --------- Helpers ----------
class AnswerSchema(BaseModel):
    value: Any
    confidence: float
    reason: str

def load_questions() -> List[Dict[str, Any]]:
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def validate_value(q: Dict[str, Any], value: Any) -> Optional[str]:
    """
    Validate a value against a question definition.
    
    Args:
        q: Question definition
        value: Value to validate
        
    Returns:
        Error message if invalid, None if valid
    """
    qtype = q.get("type", "string")
    print(f"Validating value '{value}' for type '{qtype}'")
    
    # Check for empty values
    if value in (None, ""):
        if q.get("required"):
            print(f"Empty value for required field")
            return "Hiányzó érték."
        else:
            print(f"Empty value for optional field - allowed")
            return None

    try:
        # Email validation
        if qtype == "email":
            try:
                # Use a simple regex for email validation instead of EmailStr.validate
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                if re.match(email_pattern, str(value)):
                    print(f"Valid email: {value}")
                else:
                    print(f"Invalid email format: {value}")
                    return "Érvénytelen email cím formátum."
            except Exception as e:
                print(f"Email validation error: {str(e)}")
                return "Érvénytelen email cím."
                
        # Date validation
        elif qtype == "date":
            if not isinstance(value, str):
                print(f"Date is not a string: {value}")
                return "Érvénytelen dátum formátum."
                
            try:
                parsed_date = dateparser.parse(value, dayfirst=True)
                if not parsed_date:
                    print(f"Could not parse date: {value}")
                    return "Nem sikerült értelmezni a dátumot."
                    
                iso = parsed_date.date().isoformat()
                if not re.match(r"^\d{4}-\d{2}-\d{2}$", iso):
                    print(f"Date not in ISO format: {iso}")
                    return "Érvénytelen dátum formátum."
                    
                print(f"Valid date: {value} -> {iso}")
            except Exception as e:
                print(f"Date parsing error: {str(e)}")
                return "Érvénytelen dátum."
                
        # Number validation
        elif qtype == "number":
            try:
                float(value)
                print(f"Valid number: {value}")
            except ValueError:
                print(f"Invalid number: {value}")
                return "Érvénytelen szám."
                
        # Select validation
        elif qtype == "select":
            opts = q.get("options", [])
            if value not in opts:
                print(f"Invalid option: {value}, valid options: {opts}")
                return f"Válassz az opciók közül: {', '.join(opts)}"
            print(f"Valid option: {value}")
            
        # Custom regex validators
        for v in q.get("validators", []):
            rgx = v.get("regex")
            msg = v.get("message", "Érvénytelen formátum.")
            if rgx and (not re.match(rgx, str(value))):
                print(f"Regex validation failed: {rgx} for value: {value}")
                return msg
                
        print(f"All validations passed for value: {value}")
        return None
        
    except ValidationError as e:
        print(f"Validation error: {str(e)}")
        return "Érvénytelen formátum."
    except Exception as e:
        print(f"Unexpected validation error: {str(e)}")
        return "Nem sikerült értelmezni."

def apply_followups(q: Dict[str, Any], answers: Dict[str, Any]) -> List[str]:
    msgs = []
    for f in q.get("followups", []) or []:
        cond = f.get("when", {})
        ok = all(answers.get(k) == v for k, v in cond.items())
        if ok and "ask" in f:
            msgs.append(f["ask"])
    return msgs

def get_next_open_question(questions: List[Dict[str, Any]], answers: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Get the next open question that needs to be answered.
    
    Args:
        questions: List of question definitions
        answers: Dictionary of current answers
        
    Returns:
        The next question to ask, or None if all questions are answered
    """
    print(f"Finding next open question. Current answers: {answers}")
    
    for q in questions:
        var = q["var"]
        required = q.get("required", False)
        print(f"Checking question '{var}' (required: {required})")
        
        # Check if the question is answered
        if var not in answers or answers[var] in (None, ""):
            if required:
                print(f"Question '{var}' is required and not answered")
                return q
            elif var not in answers:
                print(f"Question '{var}' is optional and not answered")
                return q
            else:
                print(f"Question '{var}' has empty answer but is optional")
        else:
            # If the question is answered, check if it's valid
            value = answers[var]
            print(f"Question '{var}' has answer: {value}")
            
            # Validate the answer
            err = validate_value(q, value)
            if err:
                print(f"Question '{var}' has invalid answer: {err}")
                return q
            else:
                print(f"Question '{var}' has valid answer")
    
    print("All questions are answered and valid")
    return None

def llm_extract(var: str, qtype: str, options: Optional[List[str]], user_text: str) -> AnswerSchema:
    """
    Extract structured information from user text using LLM.
    
    Args:
        var: The variable name to extract
        qtype: The type of the variable (text, email, date, select, etc.)
        options: List of options for select type
        user_text: The user's text to extract information from
        
    Returns:
        AnswerSchema object with extracted value, confidence, and reason
    """
    # Create a more explicit system prompt to ensure proper JSON formatting
    system = (
        "Feladat: A felhasználó válaszából töltsd ki az adott mezőt strukturáltan. "
        "Kizárólag a kért JSON-t add vissza, komment nélkül. "
        "A válasznak érvényes JSON formátumúnak kell lennie."
    )
    
    # Create a more explicit user prompt
    user = f"""Mező:
            - var: {var}
            - type: {qtype}
            - options: {options if options else "n/a"}

            Felhasználói üzenet:
            \"\"\"{user_text}\"\"\"

            Válasz formátum (pontosan ezt a formátumot kövesd):
            {{"value": <érték vagy null>, "confidence": 0..1, "reason": "<rövid magyarázat>"}}

            Szabályok:
            - "date" esetén ISO 8601 (YYYY-MM-DD).
            - "email" érvényes email.
            - "select" csak a megadott opciók egyike.
            - Ha nem egyértelmű, value legyen null és írd le, mi hiányzik.
            - A válasznak érvényes JSON-nak kell lennie, más szöveget ne tartalmazzon.
            """

    # Print debug info
    print(f"Extracting field '{var}' of type '{qtype}' from: {user_text[:50]}...")
    
    try:
        # Get the LLM and invoke it using OpenRouter
        llm = get_chat_llm(LLM_MODEL)
        response = llm.invoke([
            HumanMessage(content=system + "\n\n" + user)
        ])
        
        # Get the content from the response
        if hasattr(response, 'content'):
            content = response.content.strip()
        else:
            # If response doesn't have content attribute, try to get it from the message
            content = response.message.content.strip() if hasattr(response, 'message') else str(response)
        
        # Print the raw response for debugging
        print(f"Raw LLM response: {content[:100]}...")
        
        # Try to extract JSON from the response if it's not already valid JSON
        if not content.startswith('{'):
            # Look for JSON-like content in the response
            json_match = re.search(r'({.*})', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)
                print(f"Extracted JSON: {content[:100]}...")
        
        try:
            data = json.loads(content)
            return AnswerSchema(**data)
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {str(e)}")
            # Try to create a valid JSON from the response
            if "value" in content and "confidence" in content and "reason" in content:
                # Attempt to extract values using regex
                value_match = re.search(r'"value"\s*:\s*(?:"([^"]+)"|null|(\d+))', content)
                confidence_match = re.search(r'"confidence"\s*:\s*([\d.]+)', content)
                reason_match = re.search(r'"reason"\s*:\s*"([^"]+)"', content)
                
                if value_match and confidence_match and reason_match:
                    value = value_match.group(1) or value_match.group(2) or None
                    confidence = float(confidence_match.group(1))
                    reason = reason_match.group(1)
                    return AnswerSchema(value=value, confidence=confidence, reason=reason)
            
            # If all else fails, return a fallback response
            return AnswerSchema(
                value=None,
                confidence=0.0,
                reason=f"Nem sikerült JSON-t kinyerni a válaszból: {str(e)}"
            )
    except Exception as e:
        print(f"LLM error: {str(e)}")
        # Handle any errors with the LLM call
        return AnswerSchema(
            value=None,
            confidence=0.0,
            reason=f"LLM hívási hiba: {str(e)}"
        )

# --------- UI ----------
st.set_page_config(page_title="Kérdőív-chat (LLM)", page_icon="💬")

if "questions" not in st.session_state:
    st.session_state.questions = load_questions()
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "history" not in st.session_state:
    st.session_state.history = []  # [(role, text)]
if "current_q" not in st.session_state:
    st.session_state.current_q = st.session_state.questions[0]

st.title("💬 Kérdőív-chat (LLM)")

# Kezdő üzenet
if not st.session_state.history:
    st.session_state.history.append(("assistant", st.session_state.current_q["text"]))

for role, text in st.session_state.history:
    with st.chat_message(role):
        st.markdown(text)

# Chat input
user_msg = st.chat_input("Írj választ…")
if user_msg:
    st.session_state.history.append(("user", user_msg))
    q = st.session_state.current_q
    var = q["var"]
    qtype = q.get("type", "string")
    options = q.get("options", None)

    # LLM extrakció
    extract = llm_extract(var, qtype, options, user_msg)
    candidate = extract.value

    print(f"LLM extract for var '{var}': {extract}")
    print(f"Current answers: {st.session_state.answers}")

    # Validálás + visszajelzés
    err = validate_value(q, candidate)
    print(f"Validation result for '{var}': {err if err else 'OK'}")
    
    if err:
        # Ha hibás vagy bizonytalan, kérjünk pontosítást
        follow = f"Értem. {err} Kérlek, pontosítsd: **{q['text']}**"
        st.session_state.history.append(("assistant", follow))
    else:
        # Mentsük a választ (normalizálás dátumnál)
        try:
            if qtype == "date" and isinstance(candidate, str):
                iso = dateparser.parse(candidate, dayfirst=True).date().isoformat()
                candidate = iso
                print(f"Normalized date: {candidate}")
        except Exception as e:
            print(f"Date parsing error: {str(e)}")
            # Keep original value if parsing fails
        
        # Save the answer
        st.session_state.answers[var] = candidate
        print(f"Saved answer for '{var}': {candidate}")
        print(f"Updated answers: {st.session_state.answers}")

        # Esetleges followup-kérdések (feltételes)
        fups = apply_followups(q, st.session_state.answers)
        print(f"Followups: {fups}")
        for m in fups:
            st.session_state.history.append(("assistant", m))

        # Következő nyitott kérdés
        try:
            nxt = get_next_open_question(st.session_state.questions, st.session_state.answers)
            print(f"Next question: {nxt['var'] if nxt else 'None'}")
            
            if nxt is None:
                # Kész!
                result_json = json.dumps(st.session_state.answers, ensure_ascii=False, indent=2)
                st.session_state.history.append(("assistant", "Köszönöm! Minden kötelező mező megvan. Íme az összefoglaló:"))
                st.session_state.history.append(("assistant", f"```json\n{result_json}\n```"))
            else:
                st.session_state.current_q = nxt
                st.session_state.history.append(("assistant", nxt["text"]))
        except Exception as e:
            print(f"Error getting next question: {str(e)}")
            st.error(f"Hiba történt a következő kérdés meghatározásakor: {str(e)}")

    # Force Streamlit to rerun with the updated state
    try:
        st.rerun()
    except Exception as e:
        print(f"Error during rerun: {str(e)}")
        st.error("Hiba történt az oldal frissítésekor. Kérjük, próbálja újra.")