from pathlib import Path
from typing import Optional, List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from textwrap import dedent
from dotenv import load_dotenv
import os
import time
from agno.agent import Agent, RunResponse
from agno.storage.sqlite import SqliteStorage
from agno.models.openrouter import OpenRouter
from agno.tools.googlesearch import GoogleSearchTools  
from docling.document_converter import DocumentConverter
import chromadb  

# Suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()  # Load OPENROUTER_API_KEY from .env

## to Extract the Text from PDF and converted it to .md
def load_pdf_docling(pdf_path: Path) -> Optional[str]:
    """Extract Markdown text from PDF using Docling."""
    try:
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        if result.document:
            markdown_text = result.document.export_to_markdown()
            return markdown_text
        else:
            print(f"Warning: No content extracted from {pdf_path}")
            return None
    except Exception as e:
        print(f"Error processing PDF with Docling: {e}")
        return None


class BangaloreConcierge:
    def __init__(self, pdf_path: str, db_file: str = "tmp/conversations.db"):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        self.markdown_path = Path("bangalore_guide.md")
        self.db_file = db_file
        self.kannada_phrases = [
            "Namaskara (hello)", "Dhanyavadagalu (thank you)", "Kshamisi (sorry/excuse me)",
            "Meter haaki (turn on the meter)", "Idhu eshtu? (how much is this?)",
            "Kannada gottilla (I don't know Kannada)", "Nēra hōgi (go straight)",
            "Balakke tirigi (turn right)", "Edakke tirigi (turn left)", "Kammi madi (reduce the price)"
        ]
        self.chunks, self.metadata, self.collection, self.embedder = self._setup_knowledge_base()
        if not self.chunks:
            raise RuntimeError("Failed to initialize knowledge base")
        self.agent = self._setup_agent()
        self.storage = SqliteStorage(table_name="agent_sessions", db_file=self.db_file)
        self.persona_cache = {}  # Cache persona per session_id

    def _setup_knowledge_base(self) -> tuple[Optional[List[str]], Optional[List[Dict]], Optional[chromadb.Collection], Optional[SentenceTransformer]]:
        """Set up Chroma collection for querying using content extracted from PDF."""
        try:
            # Extract content from PDF using Docling
            markdown_text = load_pdf_docling(self.pdf_path)
            if not markdown_text:
                raise ValueError(f"Failed to extract content from {self.pdf_path}")
            
            # Save extracted content to Markdown file for reference
            with open(self.markdown_path, "w", encoding="utf-8") as f:
                f.write(markdown_text)
            
            # Chunk the extracted text for embedding
            chunks = [markdown_text[i:i+1000] for i in range(0, len(markdown_text), 1000)]
            metadata = []
            for chunk in chunks:
                is_tourist = any(kw in chunk.lower() for kw in ["sightseeing", "visit", "weekend", "palace", "lalbagh", "cubbon", "iskcon"])
                is_kannada = any(phrase.split(" ")[0].lower() in chunk.lower() for phrase in self.kannada_phrases)
                metadata.append({
                    "persona": "tourist" if is_tourist else "resident",
                    "is_kannada": is_kannada
                })
            embedder = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = embedder.encode(chunks, show_progress_bar=True)
            
            # Set up Chroma collection
            client = chromadb.Client()
            collection = client.create_collection(name="bangalore_guide")
            collection.add(
                documents=chunks,
                embeddings=embeddings.tolist(),
                metadatas=metadata,
                ids=[f"chunk_{i}" for i in range(len(chunks))]
            )
            return chunks, metadata, collection, embedder
        except Exception as e:
            print(f"❌ Knowledge base setup failed: {e}")
            return None, None, None, None

    def _custom_retriever(self, agent: Agent, query: str, num_documents: Optional[int] = 5, **kwargs) -> Optional[List[Dict]]:
        """Custom Chroma retriever for Agentic RAG."""
        try:
            query_lower = query.lower()
            if "kannada" in query_lower:
                # Filter for Kannada-specific metadata
                results = self.collection.query(
                    query_texts=[query],
                    n_results=min(3, num_documents),
                    where={"is_kannada": True}
                )
                if results["documents"]:
                    retrieved = [{"content": doc, "metadata": meta} for doc, meta in zip(results["documents"][0], results["metadatas"][0])]
                    print(f"DEBUG: Retrieved {len(retrieved)} Kannada-specific chunks for query '{query}', persona 'tourist'")
                    print("DEBUG: Kannada chunks content (first 100 chars):")
                    for i, result in enumerate(retrieved):
                        print(f"  Chunk {i+1}: {result['content'][:100]}...")
                    return retrieved
            query_embedding = self.embedder.encode([query]).tolist()
            persona = self.detect_persona(query, kwargs.get("session_id", "default_session"))
            # Query with optional persona filter
            filter_query = {"persona": persona} if persona != "unknown" else None
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=num_documents,
                where=filter_query
            )
            if not results["documents"]:
                # Fallback to top 3 without filter
                results = self.collection.query(
                    query_embeddings=query_embedding,
                    n_results=3
                )
            retrieved = [{"content": doc, "metadata": meta, "distance": dist} for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0])]
            # Sort by keyword relevance
            query_keywords = query_lower.split()
            retrieved = sorted(retrieved, key=lambda x: sum(kw in x["content"].lower() for kw in query_keywords), reverse=True)[:num_documents]
            print(f"DEBUG: Retrieved {len(retrieved)} chunks for query '{query}', persona '{persona}', distances: {results['distances'][0][:2]}")
            print("DEBUG: Retrieved chunks content (first 100 chars):")
            for i, result in enumerate(retrieved):
                print(f"  Chunk {i+1}: {result['content'][:100]}...")
            return retrieved
        except Exception as e:
            print(f"⚠️ Retriever failed: {e}")
            return []

    def _setup_agent(self) -> Agent:
        """Set up Agno Agent with OpenRouter, GoogleSearchTools, and custom retriever."""
        try:
            return Agent(
                model=OpenRouter(id="google/gemini-flash-1.5-8b", max_tokens=4096),
                tools=[GoogleSearchTools()],
                show_tool_calls=True,
                description=dedent("""\
                    You are a Bangalore Smart City Concierge, assisting tourists and new residents with precise, engaging, and practical information from 'The Ultimate Bangalore Guide' and current web data when needed. Detect the user's persona (tourist or new resident) from conversation cues and adapt your tone: enthusiastic and concise for tourists, detailed and practical for new residents. Use Bangalore-specific context like traffic, tech culture, and Kannada phrases for authenticity.
                """),
                instructions=[
                    "Detect user persona from cues (e.g., 'weekend,' 'visit' for tourists; 'moving,' 'work' for residents) and maintain it throughout the session using stored session data.",
                    "For tourists: Use an enthusiastic, concise tone, focusing on top attractions (e.g., Bangalore Palace, Toit), must-eat places (e.g., Vidyarthi Bhavan dosas), and Instagram spots. Example: 'Skip the traffic, boss! Take the Purple Line metro to Cubbon Park (₹40, 20 mins). Hit Vidyarthi Bhavan for legendary masala dosa!'",
                    "For new residents: Use a detailed, practical tone, covering neighborhoods (e.g., Koramangala, HSR Layout), commute tips (e.g., Namma Metro, Namma Yatri), and daily life (e.g., Dunzo, Swiggy). Example: 'For IT pros, Koramangala has great connectivity to tech parks, with 2BHK rentals ₹25-35K. Use Namma Yatri for autos.'",
                    "Use ONLY 'The Ultimate Bangalore Guide' context unless the query involves prices, events, or new places (post-2023). For these, use GoogleSearchTools to fetch current data and blend with 'Current information shows...'.",
                    "Structure responses with bullet points, quoting specific details (e.g., 'Bangalore Palace, inspired by Windsor Castle').",
                    "For Kannada queries, prioritize phrases like 'Namaskara (hello)' or 'Meter haaki (turn on the meter)' from the guide or fallback list.",
                    "If context lacks details, say: 'The guide lacks specific details; try a more specific query or check current information.'",
                    "Incorporate Bangalore slang (e.g., 'Boss, traffic is mental!'), auto-rickshaw tips (e.g., 'Say Meter haaki to avoid overcharging'), and tech culture (e.g., 'Join Bangalore Dev Meetups for networking').",
                    "If PDF and web data conflict, prioritize web data for prices/events and note: 'The guide says X, but current information shows Y.'"
                ],
                storage=SqliteStorage(table_name="agent_sessions", db_file=self.db_file),
                add_history_to_messages=True,
                num_history_runs=3,
                search_knowledge=True,
                retriever=self._custom_retriever,
                markdown=True
            )
        except Exception as e:
            print(f"❌ Agent setup failed: {e}")
            return None

    def detect_persona(self, user_input: str, session_id: str) -> str:
        """Detect and persist persona using keyword matching and storage."""
        if session_id in self.persona_cache:
            return self.persona_cache[session_id]
        tourist_keywords = ["weekend", "visit", "tourist", "sightseeing", "2 days", "palace", "explore"]
        resident_keywords = ["moving", "work", "electronic city", "housing", "live", "settle", "relocate"]
        user_input_lower = user_input.lower()
        tourist_score = sum(kw in user_input_lower for kw in tourist_keywords)
        resident_score = sum(kw in user_input_lower for kw in resident_keywords)
        persona = (
            "resident" if resident_score > tourist_score else
            "tourist" if tourist_score > 0 else
            "unknown"
        )
        self.persona_cache[session_id] = persona
        print(f"DEBUG: Detected persona '{persona}' for session '{session_id}'")
        return persona

    def needs_web_search(self, user_input: str) -> bool:
        """Check if query requires a web search."""
        user_input_lower = user_input.lower()
        triggers = ["price", "cost", "2023", "event", "new", "current", "weather"]
        return any(trigger in user_input_lower for trigger in triggers)

    def _perform_web_search(self, user_input: str) -> List[Dict]:
        """Perform web search using GoogleSearchTools."""
        try:
            tool = GoogleSearchTools()
            # Refine query for better relevance (e.g., add '2024' and focus on official sources)
            refined_query = f"{user_input} 2024 official site or reliable sources"
            web_results = tool.google_search(refined_query, max_results=5, language="en")
            print(f"DEBUG: Web search results for '{refined_query}': {str(web_results)[:100]}...")
            # Normalize to List[Dict] with 'content' key if needed
            if isinstance(web_results, list) and all(isinstance(r, dict) for r in web_results):
                return web_results
            elif isinstance(web_results, list):
                return [{"content": str(r)} for r in web_results]
            else:
                return [{"content": str(web_results)}]
        except Exception as e:
            print(f"⚠️ Web search failed: {e}")
            return []

    def _build_prompt(self, user_input: str, context_text: str, web_results: List[Dict], persona: str) -> str:
        """Build the prompt for the agent based on persona and sources."""
        web_info = f"Current web information: {' '.join([r.get('content', '') + ' ' + r.get('title', '') + ' ' + r.get('href', '') for r in web_results])}" if web_results else ""
        return dedent(f"""\
            You are a Bangalore Smart City Concierge. Provide a response based on this context from 'The Ultimate Bangalore Guide':
            {context_text}
            {web_info}
            Rules:
            - Detected persona: '{persona}'. Use enthusiastic, concise tone for tourists; detailed, practical tone for residents.
            - For tourists, focus on top attractions (e.g., Bangalore Palace, Toit), must-eat places (e.g., Vidyarthi Bhavan), Instagram spots. Example: 'Skip the traffic, boss! Take the Purple Line to Cubbon Park (₹40, 20 mins). Try Vidyarthi Bhavan’s dosa!'
            - For residents, cover neighborhoods, commute, daily life (e.g., Namma Yatri, Dunzo). Example: 'Koramangala has great connectivity, 2BHK ₹25-35K.'
            - Use ONLY guide context unless query involves prices, events, or new places (post-2023), then blend with web results using 'Current information shows...'. Extract specific details like prices from web info and include them explicitly.
            - Quote specific details (e.g., 'Bangalore Palace, inspired by Windsor Castle').
            - If context lacks info, say: 'The guide lacks specific details; try a more specific query or check current information.'
            - For Kannada queries, use phrases like 'Namaskara' or 'Meter haaki'.
            - Structure with bullet points, use Bangalore slang (e.g., 'Boss, traffic is mental!'), and include auto-rickshaw tips.
            - If PDF and web data conflict, prioritize web data for prices/events and note: 'The guide says X, but current information shows Y.' Example: 'The guide mentions ₹200, but current information shows ₹230.'
             User query: {user_input}
        """)

    def generate_response(self, user_input: str, session_id: str = "default_session") -> str:
        """Generate a persona-aware response with web search integration."""
        if not all([self.chunks, self.metadata, self.collection, self.agent]):
            return "Error: Agent or knowledge base not initialized"
        
        user_input_lower = user_input.lower()
        persona = self.detect_persona(user_input, session_id)
        context = self._custom_retriever(self.agent, user_input, num_documents=5, session_id=session_id)
        context_text = "\n".join([item["content"] for item in context]) if context else ""
        web_results = self._perform_web_search(user_input) if self.needs_web_search(user_input) else []
        prompt = self._build_prompt(user_input, context_text, web_results, persona)

        # Retry logic for OpenRouter
        max_retries = 3
        for attempt in range(max_retries):
            try:
                run_response: RunResponse = self.agent.run(prompt, session_id=session_id)
                response = run_response.messages[-1].content if run_response.messages else "Error: No response generated."
                print(f"DEBUG: Raw OpenRouter response (attempt {attempt+1}): {response[:100]}...")
                if len(response) < 50 and "Error" not in response:
                    print(f"⚠️ Response too short on attempt {attempt+1}, retrying...")
                    time.sleep(1)
                    continue
                break
            except Exception as e:
                print(f"⚠️ OpenRouter failed on attempt {attempt+1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    response = "Error: Failed to generate response after retries."

        # Fallback if response is invalid
        if not response or "Error" in response or len(response) < 50:
            if "kannada" in user_input_lower:
                response = (
                    "According to The Ultimate Bangalore Guide:\n"
                    "The guide lacks specific details; using fallback phrases:\n"
                    +"-" +{"\n- ".join(self.kannada_phrases)}
                )
            else:
                response = (
                    "According to The Ultimate Bangalore Guide:\n"
                    "The guide lacks specific details, but here's what I found:\n"
                    + "\n".join([f"- {item['content'][:200]}..." for item in context[:2]]) +
                    "\nTry a more specific query or check current information."
                )

        # Add persona-specific tips and creative bonuses
        if persona == "tourist":
            response += "\n🌟 Pro Tip: Try Masala Dosa at Vidyarthi Bhavan for a legendary experience or say 'Namaskara' to locals! Boss, say 'Meter haaki' to avoid auto overcharging!"
            response += "\n**Bangalore Survival Kit:**\n- Download Namma Metro app for quick travel\n- Use 'Idhu eshtu?' when bargaining\n- Check Instagram for hidden gems!"
            if "weather" in user_input_lower:
                response += "\n☔ Weather Tip: Monsoon season (June-Oct) brings rain—pack an umbrella for outdoor spots!"
        elif persona == "resident":
            response += "\n🏡 Local Hack: Use Namma Yatri for fair auto fares and Swiggy/Dunzo for quick groceries. Join Bangalore Dev Meetups for tech networking!"
            response += "\n**New Resident Kit:**\n- Join 'Bangalore Expats' on Facebook\n- Attend Dev Meetups on Meetup.com for networking\n- Get BESCOM app for utilities!"
        
        if "kannada" in user_input_lower and "Kannada Phrases" not in response:
            response += f"\n📝 Kannada Phrases: {', '.join(self.kannada_phrases)}"
        
        if self.needs_web_search(user_input):
            response += "\n🔍 Note: For up-to-date details (e.g., prices, events), I included current web information."

        return response

    def save_conversation(self, filename: str, query: str, response: str):
        """Save query/response to conversations/ directory."""
        try:
            Path("conversations").mkdir(exist_ok=True)
            with open(f"conversations/{filename}", "w", encoding="utf-8") as f:
                f.write(f"Query: {query}\nResponse: {response}\n")
                print(f"DEBUG: Saved response to conversations/{filename} (length: {len(response)} chars)")
        except Exception as e:
            print(f"⚠️ Failed to save conversation: {e}")


if __name__ == "__main__":
    agent = BangaloreConcierge("The_Ultimate_Bangalore_Guide.pdf")
    
    # Single query test
    test_queries = [
        # "I'm visiting Bangalore for a weekend—what are the top attractions and current entry fees for Bangalore Palace and Lalbagh Garden?",
        # "Moving to Bangalore next month for IT work in Electronic City—what are the best neighborhoods, current rental costs, and commute tips?",

    ]
    for i, query in enumerate(test_queries):
        print(f"\nProcessing query {i+1}: {query}")
        response = agent.generate_response(query, session_id=f"session_{i}")
        print(f"Response: {response}")
        filename = (
            "tourist_session.txt" if i == 0 else
            "resident_session.txt" if i == 1 else
            "persona_switch.txt" if i == 3 else
            f"session_{i}.txt"
        )
        agent.save_conversation(filename, query, response)
    
    # Multi-turn session test for persona switching
    switch_session_id = "switch_session"
    switch_queries = [
        # "What should I see in 2 days?",  # Tourist
        # "Moving here for work—best neighborhoods?",  # Resident switch
    ]
    for j, query in enumerate(switch_queries):
        print(f"\nProcessing query in {switch_session_id}: {query}")
        response = agent.generate_response(query, session_id=switch_session_id)
        print(f"Response: {response}")
        # Append to persona_switch.txt
        agent.save_conversation("persona_switch.txt", query, response)
    
    print("\n✅ All queries processed. Check the 'conversations' directory.")
    # ... (rest of your code remains the same)


   
    print("\n=== Interactive CLI Mode ===")
    print("Enter your query (type 'exit' to quit):")
    session_id = "interactive_session"  # Fixed session for continuity
    while True:
        query = input("> ")
        if query.lower() == "exit":
            break
        response = agent.generate_response(query, session_id=session_id)
        print(f"Response: {response}")
        # Optionally save each interaction
        agent.save_conversation("interactive_session.txt", query, response)
    
    print("\n✅ CLI session ended. Check 'conversations/' for logs.")

