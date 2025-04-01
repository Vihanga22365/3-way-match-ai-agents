import streamlit as st
from openai import OpenAI
import pypdf
from io import BytesIO
import os
from dotenv import load_dotenv # Use dotenv for local development
import markdown

# --- Langchain Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI
# Langchain message types (ensure compatibility across versions)
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

# --- Configuration ---
# Load environment variables (optional, good practice for local dev)
load_dotenv()

# --- API Key Management ---
# Recommended: Use Environment Variables or Streamlit Secrets
# Example using environment variables:
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')


# Fallback for direct key insertion (NOT RECOMMENDED FOR PRODUCTION)
# Replace with your actual keys ONLY if not using env vars/secrets
if not OPENAI_API_KEY:
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not GOOGLE_API_KEY:
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# --- Model Configuration ---
AVAILABLE_MODELS = {
    "GPT-4o (OpenAI)": "gpt-4o",
    "Gemini 2.5 Pro": "gemini-2.5-pro-exp-03-25",
    # Add other models here if needed, e.g., "Gemini Pro (Google/Langchain)": "gemini-pro"
}
DEFAULT_MODEL_KEY = "GPT-4o (OpenAI)"

# --- Helper Functions ---

def extract_text_from_pdf(uploaded_file):
    """
    Extracts text from an uploaded PDF file.

    Args:
        uploaded_file: A Streamlit UploadedFile object.

    Returns:
        str: Extracted text from the PDF, or None if extraction fails.
    """
    try:
        pdf_bytes = BytesIO(uploaded_file.getvalue())
        reader = pypdf.PdfReader(pdf_bytes)
        text = ""
        if reader.is_encrypted:
             st.warning(f"Skipping encrypted file: '{uploaded_file.name}'")
             return None # Skip encrypted files

        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text: # Ensure text was extracted from the page
                    text += page_text + f"\n--- Page {i+1} ---\n" # Add separator between pages
            except Exception as page_e:
                 st.warning(f"Could not extract text from page {i+1} of '{uploaded_file.name}'. It might be image-based. Error: {page_e}")
                 text += f"\n--- Page {i+1} (Extraction Failed) ---\n" # Mark page extraction failure

        # Basic check if any text was extracted at all
        if not text.strip() or text.count("Extraction Failed") == len(reader.pages):
             st.warning(f"Could not extract meaningful text from '{uploaded_file.name}'. The document might be purely image-based or scanned without OCR.")
             return None

        return text
    except pypdf.errors.PdfReadError as pdf_err:
        st.error(f"Error reading PDF file '{uploaded_file.name}': {pdf_err}. The file might be corrupted or password-protected (without user password).")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while processing PDF '{uploaded_file.name}': {e}")
        return None

def convert_messages_for_langchain(messages):
    """Converts standard message dicts to Langchain BaseMessage objects."""
    lc_messages = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if not role or not content:
            continue # Skip invalid messages
        if role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
        elif role == "system":
            lc_messages.append(SystemMessage(content=content))
    return lc_messages

def call_llm_agent(messages_for_api, selected_model_key):
    """
    Calls the selected LLM API (OpenAI or Gemini via Langchain)
    with the current conversation history.

    Args:
        messages_for_api (list): List of message dictionaries.
        selected_model_key (str): The key from AVAILABLE_MODELS identifying the chosen model.

    Returns:
        str: The response content from the AI agent, or None if an error occurs.
    """
    model_name = AVAILABLE_MODELS.get(selected_model_key)
    if not model_name:
        st.error(f"Invalid model selection key: {selected_model_key}")
        return None

    try:
        # --- OpenAI Call ---
        if "OpenAI" in selected_model_key:
            if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("YOUR_"):
                st.error("⚠️ OpenAI API key is missing or invalid. Please configure it.")
                return None
            client = OpenAI(api_key=OPENAI_API_KEY)
            valid_messages = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in messages_for_api if "role" in msg and "content" in msg
            ]
            if not valid_messages:
                 st.error("Cannot call OpenAI API with empty message list.")
                 return None
            response = client.chat.completions.create(
                model=model_name,
                messages=valid_messages,
                temperature=0.2
            )
            return response.choices[0].message.content

        # --- Google Gemini Call (via Langchain) ---
        elif "Gemini" in selected_model_key:
            if not GOOGLE_API_KEY or GOOGLE_API_KEY.startswith("YOUR_"):
                st.error("⚠️ Google API key is missing or invalid. Please configure it.")
                return None

            # Initialize Langchain Chat Model
            llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=GOOGLE_API_KEY, temperature=0.2, convert_system_message_to_human=True) # convert_system needed for some Gemini versions

            # Convert messages to Langchain format
            lc_messages = convert_messages_for_langchain(messages_for_api)
            if not lc_messages:
                 st.error("Cannot call Gemini API with empty message list.")
                 return None

            # Invoke the model
            response = llm.invoke(lc_messages)

            # Ensure response is in the expected format (Langchain might return AIMessage object)
            if isinstance(response, BaseMessage):
                return response.content
            elif isinstance(response, str):
                 return response
            else:
                 st.error(f"Unexpected response type from Langchain Gemini: {type(response)}")
                 return None

        else:
            st.error(f"Model provider for '{selected_model_key}' not recognized.")
            return None

    except Exception as e:
        st.error(f"Error calling {selected_model_key} API: {e}")
        return None

# --- Streamlit Application ---

st.set_page_config(page_title="3-Way Match Agent", layout="wide")
st.title("3-Way Match AI Agent")

# --- Sidebar for Model Selection & Status ---
with st.sidebar:
    st.subheader("Configuration")
    # Model Selection Dropdown
    selected_model_display = st.selectbox(
        "Select LLM Model:",
        options=list(AVAILABLE_MODELS.keys()),
        index=list(AVAILABLE_MODELS.keys()).index(DEFAULT_MODEL_KEY), # Default selection
        key="selected_model_key" # Store selection in session state under this key
    )
    st.caption(f"Using: {AVAILABLE_MODELS[st.session_state.selected_model_key]}")

    # API Key Status Check (Informational)
    # st.markdown("---")
    # st.subheader("API Key Status")
    # if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("YOUR_"):
    #     st.warning("OpenAI API Key not found or placeholder detected.")
    # else:
    #     st.success("OpenAI API Key loaded.")
    # if not GOOGLE_API_KEY or GOOGLE_API_KEY.startswith("YOUR_"):
    #     st.warning("Google API Key not found or placeholder detected.")
    # else:
    #     st.success("Google API Key loaded.")
    # st.caption("Configure keys via environment variables (e.g., `OPENAI_API_KEY`, `GOOGLE_API_KEY`) or Streamlit secrets for deployment.")

    # Status Display Area (moved here)
    st.markdown("---")
    st.subheader("Process Status")
    status_placeholder = st.empty() # Placeholder to update status dynamically

st.write(f"Select the LLM model in the sidebar. Upload Purchase Order (PO), Invoice, and Goods Receipt (GR) PDFs, along with any relevant email trails or communication evidence.")

# --- Session State Initialization ---
# Initialize session state variables if they don't exist
if "messages" not in st.session_state:
    st.session_state.messages = [] # Display history
if "full_context_messages" not in st.session_state:
     st.session_state.full_context_messages = [] # API history
if "processing_stage" not in st.session_state:
    st.session_state.processing_stage = "initial"
if "uploaded_docs_text" not in st.session_state:
    st.session_state.uploaded_docs_text = {}
if "uploaded_emails_text" not in st.session_state:
    st.session_state.uploaded_emails_text = {}
if "discrepancy_found" not in st.session_state:
    st.session_state.discrepancy_found = False
if "initial_analysis_done" not in st.session_state:
    st.session_state.initial_analysis_done = False

# --- File Uploaders ---
col1, col2 = st.columns(2)
with col1:
    uploaded_core_docs = st.file_uploader(
        "Upload Core Docs:\nPurchase Order, Invoice, Goods Receipt",
        type="pdf",
        accept_multiple_files=True,
        key="core_docs_uploader"
    )
with col2:
    uploaded_email_docs = st.file_uploader(
        "Upload Communication Evidence:\nEmail trails, Contracts, etc.",
        type="pdf",
        accept_multiple_files=True,
        key="email_docs_uploader"
    )

# --- Process Button ---
if uploaded_core_docs and not st.session_state.initial_analysis_done:
    if st.button(f"Process 3 Way Match using {st.session_state.selected_model_key}"):
        # Reset states for a new run
        st.session_state.processing_stage = "analyzing_docs"
        st.session_state.messages = []
        st.session_state.full_context_messages = []
        st.session_state.uploaded_docs_text = {}
        st.session_state.uploaded_emails_text = {}
        st.session_state.discrepancy_found = False
        st.session_state.initial_analysis_done = True # Mark as started

        # Add initial system message
        # Note: Gemini (via Langchain) might handle system prompts differently.
        # Setting convert_system_message_to_human=True in ChatGoogleGenerativeAI helps.
        system_message = {"role": "system", "content": "You are an AI assistant specialized in supply chain 3-way matching. Analyze the provided documents meticulously. Identify PO, Invoice, and GR. Compare line items (description, quantity, price). Report matches or list discrepancies clearly. If discrepancies exist, ask the user if you should analyze emails for reconciliation evidence. Do not show the raw document text in your response to the user."}
        # Don't add system message to display history if using Gemini with conversion? Or keep it for clarity? Let's keep it for now.
        # st.session_state.messages.append(system_message) # Optional: Display system message?
        st.session_state.full_context_messages.append(system_message)

        # Process Core Documents
        with st.spinner("Analyzing Purchase Order, Invoice, and Goods Receipt..."):
            combined_core_text = ""
            st.write("---")
            st.write("### Processing Core Documents:")
            files_processed_count = 0
            for doc in uploaded_core_docs:
                st.write(f"- Reading '{doc.name}'...")
                text = extract_text_from_pdf(doc)
                if text:
                    st.session_state.uploaded_docs_text[doc.name] = text
                    combined_core_text += f"--- Document: {doc.name} ---\n{text}\n\n"
                    files_processed_count += 1

            if files_processed_count == 0:
                st.error("Failed to extract text from any core documents.")
                st.session_state.processing_stage = "error"
                st.session_state.initial_analysis_done = False # Allow reprocessing
                st.stop()
            elif files_processed_count < len(uploaded_core_docs):
                 st.warning("Could not process all core documents.")

            # Prepare the initial analysis prompt
            initial_prompt_content = f"""
            Analyze the following documents for a 3-way match. The documents provided are:
            {list(st.session_state.uploaded_docs_text.keys())}

            Document Content:
            [Content Start]
            {combined_core_text}
            [Content End]

            ##Objective##
            You are an agent with the objective of carrying out a 3 wayyb match comparing a purchase order, invoice and a goods receipt in the manufacturing domain. 
            You will analyze the documents and try to reconcile the 3 way match and will find evidence to do so based on the evidence documents provided. 

            ##Instructions##
            1. Identify which document is the Purchase Order (PO), which is the Invoice, and which is the Goods Receipt (GR). If any are missing or unclear, state that and ask the user.
            2. Compare the PO, Invoice, and GR line item by line item. Focus on:
               - Item descriptions/codes
               - Quantities (Compare PO vs GR, Invoice vs GR)
               - Prices/Total amounts (Compare PO vs Invoice)
            3. If all relevant line items match perfectly across the documents (considering quantities and prices as described above), state clearly: "3-way match successful" to the user. 
            4. If there are any discrepancies, only list the discrepencies. List each discrepancy clearly. For each discrepancy, specify:
               - The line item involved (description or number).
               - The nature of the discrepancy (e.g., 'Quantity Mismatch', 'Price Mismatch').
               - The values found in each relevant document (e.g., 'PO Qty: 10, GR Qty: 8', 'Invoice Price: $100, PO Price: $95').
               - The documents involved in the specific mismatch (e.g., 'between Invoice and PO').
            5. After listing all discrepancies, explicitly ask the user: "Discrepancies found. Do you want me to analyze the provided email trails/communication documents for reconciliation evidence?" Do not proceed further until the user responds.
            6. Bsaed on the email trail if you can find evidence to reconcile the PO, Invoice and GR, mention the findigns clearly with details to the user. 
            7. If all the line items can now be reconciled, mention at the end that "How that I have found evidence to reconcile he Purchase Order, nvoice and Goods Receipr, workflow will proceed to create a new Purchase Order to reflect the agreement as per the correspondence"
            6. If you cannot confidently identify the documents or perform the comparison, state the issue clearly.
            7. Do not include the raw text from the 'Document Content' section above in your response to the user. Only present your findings and the question if discrepancies exist.
            8. At any given point, user may ask additional questions regarding the documents and the process. Provide answers to the user as appropreate but do not go out of context. 
            """

            # Add user prompt (representing the analysis request) to histories
            user_analysis_request_display = f"Analyze the {len(st.session_state.uploaded_docs_text)} uploaded core documents for 3-way match."
            st.session_state.messages.append({"role": "user", "content": user_analysis_request_display})
            st.session_state.full_context_messages.append({"role": "user", "content": initial_prompt_content}) # API gets full context

            # Call the selected LLM agent
            assistant_response = call_llm_agent(st.session_state.full_context_messages, st.session_state.selected_model_key)

            if assistant_response:
                assistant_message = {"role": "assistant", "content": assistant_response}
                st.session_state.messages.append(assistant_message)
                st.session_state.full_context_messages.append(assistant_message)

                response_lower = assistant_response.lower()
                if "discrepancies found" in response_lower and ("analyze the provided email" in response_lower or "analyze the communication" in response_lower):
                    st.session_state.discrepancy_found = True
                    st.session_state.processing_stage = "awaiting_user_decision"
                elif "3-way match successful" in response_lower:
                     st.session_state.processing_stage = "done_success"
                else:
                    st.session_state.processing_stage = "agent_clarification"
            else:
                st.error(f"The {st.session_state.selected_model_key} agent did not return a response for the initial analysis.")
                error_message = {"role": "assistant", "content": "Sorry, I encountered an error during the initial analysis."}
                st.session_state.messages.append(error_message)
                st.session_state.processing_stage = "error"

        # Process Email Documents (extract text now)
        if uploaded_email_docs:
            with st.spinner("Processing communication documents..."):
                st.write("---")
                st.write("### Processing Communication Documents:")
                email_files_processed_count = 0
                for doc in uploaded_email_docs:
                    st.write(f"- Reading '{doc.name}'...")
                    text = extract_text_from_pdf(doc)
                    if text:
                        st.session_state.uploaded_emails_text[doc.name] = text
                        email_files_processed_count += 1

                if email_files_processed_count == 0 and len(uploaded_email_docs) > 0:
                     st.warning("Could not extract text from any communication documents.")
                elif email_files_processed_count < len(uploaded_email_docs):
                     st.warning("Could not process all communication documents.")
                elif email_files_processed_count > 0:
                     st.info(f"Processed {email_files_processed_count} communication document(s). Ready for analysis if needed.")

        st.rerun() # Update display

# --- Display Chat History ---
st.write("---")
st.subheader("Agent Interaction")
chat_container = st.container() # Use a container for chat messages
with chat_container:
    # Display previous messages from the DISPLAY history
    for message in st.session_state.messages:
        # Ensure role is valid before displaying
        if message.get("role") in ["user", "assistant"]:
            with st.chat_message(message["role"]):
                # st.markdown(message["content"])
                message_content_with_newlines = message["content"].replace('\n', '<br>')
                message_html_content = markdown.markdown(message_content_with_newlines)
                st.markdown(f"""<div style="font-size: 17px">{message_html_content}</div>""", unsafe_allow_html=True)

# --- Handle User Input ---
if prompt := st.chat_input("Your response or instructions..."):
    # Add user message to display history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Add raw user message to full context history for now
    st.session_state.full_context_messages.append({"role": "user", "content": prompt})

    # Display user message
    with chat_container: # Display within the same container
         with st.chat_message("user"):
            # st.markdown(prompt)
            message_content_with_newlines = prompt.replace('\n', '<br>')
            message_html_content = markdown.markdown(message_content_with_newlines)
            st.markdown(f"""<div style="font-size: 17px">{message_html_content}</div>""", unsafe_allow_html=True)

    # Agent response logic based on stage
    if st.session_state.processing_stage == "awaiting_user_decision":
        if not st.session_state.uploaded_emails_text:
            st.warning("No communication documents were processed. Cannot analyze emails.")
            warning_message = {"role": "assistant", "content": "You asked to proceed, but no communication documents were found or processed. Please upload relevant documents."}
            st.session_state.messages.append(warning_message)
            st.session_state.full_context_messages.append(warning_message)
            st.session_state.processing_stage = "awaiting_emails"
            st.rerun()
        else:
            # Prepare for email analysis
            st.session_state.processing_stage = "analyzing_emails"
            with st.spinner(f"Analyzing email trails for reconciliation evidence..."):
                combined_email_text = ""
                for filename, text in st.session_state.uploaded_emails_text.items():
                    combined_email_text += f"--- Email Document: {filename} ---\n{text}\n\n"

                # Prepare the detailed email analysis prompt FOR THE API
                email_analysis_prompt_content = f"""
                Based on the user's last message ("{prompt}") and our previous conversation where I identified discrepancies and asked about analyzing emails:

                1. First, determine if the user agreed to proceed with analyzing the email trails. Respond ONLY with "Proceeding with email analysis." or "Not proceeding with email analysis based on user response." If proceeding, continue to step 2. If not, stop here.

                2. (If proceeding) Analyze the email communications provided below to see if they reconcile the discrepancies identified earlier.

                Email Trail / Communication Document Content:
                [Content Start]
                {combined_email_text}
                [Content End]

                Instructions for Analysis (only if proceeding):
                a. Carefully read the email trail/communication content provided above.
                b. Look for specific agreements, change confirmations, clarifications, or any communication between the parties that directly addresses the discrepancies identified earlier in our conversation.
                c. For each discrepancy previously listed, state whether you found evidence in the emails to reconcile it.
                   - If evidence is found: Explain the evidence and how it reconciles the specific discrepancy (e.g., "The email dated DD/MM/YYYY confirms agreement on the price change for item X, reconciling the invoice-PO price mismatch.").
                   - If no evidence is found: State clearly that no evidence was found in the provided communications for that specific discrepancy.
                d. Conclude with a final verdict:
                   - If all discrepancies are reconciled: "Final Verdict: 3-way match successful after reconciliation using email evidence."
                   - If some discrepancies remain unresolved: "Final Verdict: 3-way match partially reconciled. The following discrepancies remain unresolved due to lack of evidence in communications: [List unresolved discrepancies]."
                   - If no discrepancies were reconciled: "Final Verdict: 3-way match unsuccessful. No evidence found in communications to reconcile the identified discrepancies."
                e. **Important:** Do not include the raw text from the 'Email Trail / Communication Document Content' section above in your response to the user. Only present your findings and the final verdict.
                """

                # Replace the last user message in API context with this detailed prompt
                st.session_state.full_context_messages[-1] = {"role": "user", "content": email_analysis_prompt_content}

                # Call the LLM
                final_verdict_response = call_llm_agent(st.session_state.full_context_messages, st.session_state.selected_model_key)

                if final_verdict_response:
                    assistant_message = {"role": "assistant", "content": final_verdict_response}
                    st.session_state.messages.append(assistant_message)
                    st.session_state.full_context_messages.append(assistant_message)

                    response_lower = final_verdict_response.lower()
                    if "not proceeding" in response_lower:
                         st.session_state.processing_stage = "done_unreconciled"
                    elif "final verdict" in response_lower:
                         st.session_state.processing_stage = "done_reconciled"
                    else:
                         st.session_state.processing_stage = "agent_clarification"
                else:
                    st.error(f"The {st.session_state.selected_model_key} agent did not return a response for the email analysis.")
                    error_message = {"role": "assistant", "content": "I encountered an error trying to analyze the communications."}
                    st.session_state.messages.append(error_message)
                    st.session_state.processing_stage = "error"

                st.rerun() # Update display

    # Handle general chat / follow-up questions
    else: # Covers initial, done_*, agent_clarification, error, analyzing_*, etc.
        with st.spinner(f"{st.session_state.selected_model_key} is thinking..."):
            # Use the full context history
            assistant_response = call_llm_agent(st.session_state.full_context_messages, st.session_state.selected_model_key)
            if assistant_response:
                assistant_message = {"role": "assistant", "content": assistant_response}
                st.session_state.messages.append(assistant_message)
                st.session_state.full_context_messages.append(assistant_message)
                 # Keep stage as is unless the agent's response clearly indicates a final state
            else:
                st.error(f"The {st.session_state.selected_model_key} agent did not return a response.")
                error_message = {"role": "assistant", "content": "Sorry, I encountered an error."}
                st.session_state.messages.append(error_message)
                # Optionally set stage to error? st.session_state.processing_stage = "error"
            st.rerun()

# --- Update Final Status Display in Sidebar ---
current_stage = st.session_state.processing_stage
with status_placeholder.container(): # Update the placeholder in the sidebar
    if current_stage == "initial":
        st.info("Ready to process documents.")
    elif current_stage == "analyzing_docs":
        st.warning("Analyzing core documents...")
    elif current_stage == "awaiting_user_decision":
        st.warning("Waiting for user input regarding communication analysis.")
    elif current_stage == "awaiting_emails":
        st.error("Waiting for user input, but no communication documents were processed.")
    elif current_stage == "analyzing_emails":
        st.warning("Analyzing communication documents...")
    elif current_stage == "done_success":
        st.success("✅ 3-Way Match Successful (Initial Analysis).")
    elif current_stage == "done_reconciled":
        last_message_content = st.session_state.messages[-1]['content'].lower() if st.session_state.messages and st.session_state.messages[-1].get('role') == 'assistant' else ""
        if "successful after reconciliation" in last_message_content:
            st.success("✅ 3-Way Match Successful (Reconciled).")
        elif "partially reconciled" in last_message_content:
            st.warning("⚠️ 3-Way Match Partially Reconciled.")
        elif "match unsuccessful" in last_message_content and "no evidence found" in last_message_content:
            st.error("❌ 3-Way Match Unsuccessful (No Reconciliation Evidence).")
        else:
            st.success("✅ Reconciliation Attempt Complete.") # Generic fallback
    elif current_stage == "done_unreconciled":
        st.warning("⚠️ 3-Way Match Process Stopped/Not Pursued. Discrepancies may remain.")
    elif current_stage == "agent_clarification":
        st.info("Agent requires clarification or provided an intermediate response.")
    elif current_stage == "error":
        st.error("❌ An error occurred during processing.")