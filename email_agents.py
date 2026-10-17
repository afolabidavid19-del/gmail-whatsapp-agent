import os
import base64
import requests
from typing import TypedDict, List, Dict, Literal
from dotenv import load_dotenv
from bs4 import BeautifulSoup

# Gmail API imports
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# LangGraph & LangChain OpenAI imports
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Twilio import
from twilio.rest import Client

load_dotenv()

# Read-only scope ensures the agent can never delete or alter emails
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


# ==========================================
# 1. DEFINE LANGGRAPH STATE
# ==========================================
class AgentState(TypedDict):
    emails: List[Dict[str, str]]
    summary: str
    delivery_status: List[str]
    error: str


# ==========================================
# 2. HELPER: GMAIL AUTH & HTML PARSER
# ==========================================
def get_gmail_service():
    """Authenticates and returns the Gmail API service instance."""
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists("credentials.json"):
                raise FileNotFoundError(
                    "Missing 'credentials.json'. Download your Desktop OAuth 2.0 Client ID "
                    "from Google Cloud Console and place it in this folder."
                )
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


def extract_clean_body(payload: dict) -> str:
    """Recursively extracts plain text from Gmail payload, stripping HTML tags."""
    body_text = ""

    if "parts" in payload:
        for part in payload["parts"]:
            mime_type = part.get("mimeType", "")
            data = part.get("body", {}).get("data")
            if mime_type == "text/plain" and data:
                body_text += base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
            elif mime_type == "text/html" and data and not body_text:
                html = base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
                body_text += BeautifulSoup(html, "html.parser").get_text(separator=" ", strip=True)
            elif "parts" in part:
                body_text += extract_clean_body(part)
    else:
        data = payload.get("body", {}).get("data")
        if data:
            raw = base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
            if payload.get("mimeType") == "text/html":
                body_text = BeautifulSoup(raw, "html.parser").get_text(separator=" ", strip=True)
            else:
                body_text = raw

    # Truncate each email body to ~800 chars to keep OpenAI token costs tiny
    return " ".join(body_text.split())[:2000]


# ==========================================
# 3. DEFINE LANGGRAPH NODES
# ==========================================
def fetch_emails_node(state: AgentState) -> dict:
    """Node 1: Fetches primary emails from the past 24 hours."""
    print("📥 [Node: fetch_emails] Connecting to Gmail...")
    try:
        service = get_gmail_service()
        # Filter: Last 24h, exclude social/promotions tabs to reduce noise
        query = "newer_than:1d -category:promotions -category:social"
        results = service.users().messages().list(userId="me", q=query, maxResults=35).execute()
        messages = results.get("messages", [])

        extracted_emails = []
        for msg in messages:
            msg_detail = service.users().messages().get(userId="me", id=msg["id"], format="full").execute()
            payload = msg_detail.get("payload", {})
            headers = payload.get("headers", [])

            subject = next((h["value"] for h in headers if h["name"].lower() == "subject"), "No Subject")
            sender = next((h["value"] for h in headers if h["name"].lower() == "from"), "Unknown Sender")
            snippet = msg_detail.get("snippet", "")
            body = extract_clean_body(payload) or snippet

            extracted_emails.append({
                "sender": sender,
                "subject": subject,
                "content": body
            })

        print(f"✅ [Node: fetch_emails] Retrieved {len(extracted_emails)} relevant emails.")
        return {"emails": extracted_emails, "error": ""}

    except Exception as e:
        print(f"❌ [Node: fetch_emails] Error: {e}")
        return {"emails": [], "error": str(e)}


def summarize_and_prioritize_node(state: AgentState) -> dict:
    """Node 2: Uses OpenAI to categorize and summarize emails for WhatsApp."""
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    print(f"🧠 [Node: summarize_and_prioritize] Analyzing emails with OpenAI ({model_name})...")
    emails = state.get("emails", [])

    # Format emails into a numbered digest for the prompt
    formatted_input = "\n\n".join([
        f"Email #{i+1}\nFrom: {e['sender']}\nSubject: {e['subject']}\nContent: {e['content']}"
        for i, e in enumerate(emails)
    ])

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        max_retries=2
    )

    system_prompt = (
        "You are an executive assistant creating a daily email briefing for WhatsApp.\n"
        "Analyze the user's emails from today and group them strictly by priority:\n\n"
        "🔴 *HIGH PRIORITY (Action Required / Deadlines / Clients / Finance)*\n"
        "🟡 *IMPORTANT (Direct Updates / Receipts / Project Threads)*\n"
        "🟢 *LOW PRIORITY (FYI / Automated Alerts / Newsletters)*\n\n"
        "Formatting Rules for WhatsApp:\n"
        "- Use single asterisks *like this* for bold text (do NOT use double asterisks **).\n"
        "- For each item, write: • *[Sender Name]*: 1-sentence summary & clear action item.\n"
        "- Skip useless OTP codes or marketing fluff completely.\n"
        "- Keep the total response under 1,400 characters so it sends as a single clean WhatsApp message."
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Here are today's {len(emails)} emails:\n\n{formatted_input}")
    ])

    summary_text = str(response.content).strip()
    print("✅ [Node: summarize_and_prioritize] Summary generated.")
    return {"summary": summary_text}


def send_whatsapp_node(state: AgentState) -> dict:
    """Node 3: Sends the prioritized summary via Twilio or Meta WhatsApp Cloud API."""
    print("📲 [Node: send_whatsapp] Dispatching summary to WhatsApp...")
    summary = state.get("summary", "")
    provider = os.getenv("WHATSAPP_PROVIDER", "twilio").lower()
    statuses = []

    if provider == "twilio":
        sid = os.getenv("TWILIO_ACCOUNT_SID")
        token = os.getenv("TWILIO_AUTH_TOKEN")
        from_num = os.getenv("TWILIO_WHATSAPP_FROM")
        to_numbers = [n.strip() for n in os.getenv("WHATSAPP_TO_NUMBERS", "").split(",") if n.strip()]

        client = Client(sid, token)
        for to_num in to_numbers:
            msg = client.messages.create(
                body=summary,
                from_=from_num,
                to=to_num
            )
            statuses.append(f"Twilio -> {to_num}: {msg.sid}")
            print(f"✅ Sent to {to_num} via Twilio (SID: {msg.sid})")

    elif provider == "meta":
        access_token = os.getenv("META_ACCESS_TOKEN")
        phone_id = os.getenv("META_PHONE_NUMBER_ID")
        to_numbers = [n.strip() for n in os.getenv("META_TO_NUMBERS", "").split(",") if n.strip()]
        url = f"https://graph.facebook.com/v21.0/{phone_id}/messages"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }

        for to_num in to_numbers:
            payload = {
                "messaging_product": "whatsapp",
                "to": to_num,
                "type": "text",
                "text": {"preview_url": False, "body": summary}
            }
            res = requests.post(url, headers=headers, json=payload, timeout=15)
            res.raise_for_status()
            statuses.append(f"Meta -> {to_num}: {res.status_code}")
            print(f"✅ Sent to {to_num} via Meta Cloud API")

    else:
        raise ValueError(f"Unsupported WHATSAPP_PROVIDER: {provider}")

    return {"delivery_status": statuses}


# ==========================================
# 4. CONDITIONAL ROUTING EDGE
# ==========================================
def route_after_fetch(state: AgentState) -> Literal["summarize_and_prioritize", "__end__"]:
    """Stops execution early if there was an error or zero emails today."""
    if state.get("error"):
        print("🛑 Stopping workflow due to Gmail fetch error.")
        return END
    if not state.get("emails"):
        print("📭 No new primary emails found today. Skipping OpenAI & WhatsApp.")
        return END
    return "summarize_and_prioritize"


# ==========================================
# 5. BUILD & COMPILE THE LANGGRAPH
# ==========================================
def build_email_agent():
    workflow = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("fetch_emails", fetch_emails_node)
    workflow.add_node("summarize_and_prioritize", summarize_and_prioritize_node)
    workflow.add_node("send_whatsapp", send_whatsapp_node)

    # Connect edges
    workflow.add_edge(START, "fetch_emails")
    workflow.add_conditional_edges(
        "fetch_emails",
        route_after_fetch,
        {
            "summarize_and_prioritize": "summarize_and_prioritize",
            END: END
        }
    )
    workflow.add_edge("summarize_and_prioritize", "send_whatsapp")
    workflow.add_edge("send_whatsapp", END)

    return workflow.compile()


if __name__ == "__main__":
    agent = build_email_agent()
    
    initial_state: AgentState = {
        "emails": [],
        "summary": "",
        "delivery_status": [],
        "error": ""
    }
    
    final_state = agent.invoke(initial_state)
    
    if final_state.get("summary"):
        print("\n--- FINAL WHATSAPP BRIEFING ---\n")
        print(final_state["summary"])