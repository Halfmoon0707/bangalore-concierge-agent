# Bangalore Smart City Concierge Agent

This repo implements an AI agent using Agno for assisting tourists and new residents in Bangalore. It uses the provided PDF ("The Ultimate Bangalore Guide") as the primary knowledge base, detects user personas, performs smart web searches via Google, and adapts responses with local insights.

## How Persona Detection Works
- **Keyword Scoring**: Analyzes queries for tourist keywords (e.g., 'weekend', 'visit', 'sightseeing', '2 days', 'palace', 'explore') vs. resident keywords (e.g., 'moving', 'work', 'electronic city', 'housing', 'live', 'settle', 'relocate').
- **Scoring Logic**: tourist_score = count of matching tourist keywords; resident_score = count of resident keywords. Assign 'resident' if resident_score > tourist_score, 'tourist' if tourist_score > 0, else 'unknown'.
- **Persistence and Switching**: Stores persona in a session cache (self.persona_cache). Maintains throughout the session; switches if new query cues change scores (e.g., from tourist to resident in multi-turn tests). Demonstrated in persona_switch.txt.

## RAG vs. Web Search Decision Tree
- **RAG (PDF-First)**: Uses Chroma for vector search on chunked PDF content (1000-char chunks, embedded with Sentence Transformers). Custom retriever filters by persona/metadata (e.g., Kannada-specific chunks) and sorts by relevance.
- **Web Search Trigger**: Checks for keywords like 'price', 'cost', 'current', 'weather', 'event', 'new', or '2023+'. If triggered, refines query (e.g., adds '2024 official site') and uses GoogleSearchTools for up-to-date info.
- **Decision Flow**:
  1. Retrieve from PDF via Chroma.
  2. If query needs current data (e.g., prices), perform web search.
  3. Blend in prompt: PDF for stable info, web for dynamic (e.g., "Current shows ₹230").
- **Edge Cases**: Fallback if web fails (note in response); prioritize web for conflicts.

## Creative Features Added
- **Kannada Phrases**: Auto-included for relevant queries (e.g., 'Namaskara', 'Meter haaki') with explanations.
- **Auto-Rickshaw Tips**: Integrated in pro tips (e.g., "Say 'Meter haaki' to avoid overcharging").
- **Bangalore Survival Kit**: Persona-specific lists (tourist: Namma Metro app, bargaining tips; resident: Expats groups, Dev Meetups, BESCOM app).
- **Weather Recommendations**: Added for weather-related queries (e.g., "Monsoon (June-Oct)—pack an umbrella!").
- **Tech Meetups**: Suggested in resident responses (e.g., "Join Bangalore Dev Meetups on Meetup.com").

## How to Handle Conflicting Information
- **Prompt Rule**: If PDF and web differ (e.g., on prices), prioritize web and note: "The guide says X, but current information shows Y." Example in prompt: "The guide mentions ₹200, but current shows ₹230."
- **Implementation**: Blending happens in _build_prompt by including web_info string; agent extracts and notes conflicts. Tested with price queries (e.g., palace entry).

## Setup and Running
- Install dependencies: `pip install -r Requirements.txt`.
- Run: `python agent.py` (generates conversation files in conversations/).
- Tests: Includes single queries and multi-turn for switching.

Limitations: Web results may vary; documented as real-world challenge.
