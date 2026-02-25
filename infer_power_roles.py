#!/usr/bin/env python
# coding: utf-8
"""
============================================================
ENRON SNA — DEEP ORGANIZATIONAL BEHAVIOR INFERENCE
============================================================
Author: Mayank
Purpose: Unconstrained LLM analysis of the Enron corpus.
         Maps network metrics and linguistic samples to 
         advanced Organizational Design frameworks to uncover
         the "Shadow Org Chart". Includes real-time 
         checkpointing and resume capabilities.
============================================================
"""

import os
import json
import time
import argparse
from typing import List
from pydantic import BaseModel, Field
from openai import OpenAI
from tqdm import tqdm
import concurrent.futures

# ---------------------------------------------------------------------------
# Advanced Structured Output Schema
# ---------------------------------------------------------------------------
class PowerBases(BaseModel):
    expert_power: int = Field(description="Score 1-10 based on giving technical advice or specialized knowledge.")
    referent_power: int = Field(description="Score 1-10 based on likability, charisma, and persuasion without authority.")
    legitimate_power: int = Field(description="Score 1-10 based on clear formal authority, giving direct orders, or approving resources.")
    coercive_power: int = Field(description="Score 1-10 based on use of threats, reprimands, or strict compliance demands.")

class NodeProfile(BaseModel):
    overall_power_index: int = Field(
        description="A precise integer from 1 to 100 representing the true organizational influence, where 100 is the CEO/Top Executive level."
    )
    french_and_raven_bases: PowerBases
    inferred_informal_role: str = Field(
        description="Classify their true role in the shadow org chart (e.g., 'Information Gatekeeper', 'Hidden Decision Maker', 'Isolated Specialist', 'Toxic Broker')."
    )
    communication_style: str = Field(
        description="Describe their linguistic footprint (e.g., 'Directive and terse', 'Deferential and expansive', 'Highly diplomatic')."
    )
    evidence_quotes: List[str] = Field(
        description="Extract 2 to 3 exact, verbatim quotes from the provided email snippets that best prove their assigned power level and style."
    )
    comprehensive_analysis: str = Field(
        description="A dense, academic paragraph (150-200 words) synthesizing how their network position (betweenness/pagerank) interacts with their communication tone to define their real power in the organization."
    )

# ---------------------------------------------------------------------------
# Main Processing Logic
# ---------------------------------------------------------------------------
def run_deep_inference(input_file: str, output_file: str):
    # Config from environment (fallback to placeholder)
    api_key = os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    if not os.path.exists(input_file):
        print(f"❌ Error: Could not find {input_file}")
        return

    # Load the raw input nodes
    with open(input_file, 'r') as f:
        all_nodes = json.load(f)

    enriched_nodes = []
    processed_emails = set()

    # ── RESUME LOGIC ──────────────────────────────────────────────────────
    if os.path.exists(output_file):
        print(f"🔄 Found existing checkpoint at {output_file}. Loading...")
        try:
            with open(output_file, 'r') as f:
                enriched_nodes = json.load(f)
            # Track who has already been processed to skip them
            processed_emails = {person['email'] for person in enriched_nodes if 'deep_profile' in person}
            print(f"✅ Loaded {len(processed_emails)} previously processed profiles.")
        except json.JSONDecodeError:
            print(f"⚠️ Warning: {output_file} is corrupted or empty. Starting fresh.")
            enriched_nodes = []

    # Filter out nodes that have already been processed
    nodes_to_process = [n for n in all_nodes if n['email'] not in processed_emails]

    if not nodes_to_process:
        print("🎉 All nodes have already been processed! Nothing left to do.")
        return

    print(f"⏳ Nodes remaining in queue: {len(nodes_to_process)}")

    # ── Master System Prompt ──────────────────────────────────────────────
    system_prompt = """
    You are an elite professor of Organizational Behavior and Social Network Analysis.
    You are conducting a forensic analysis of a corporate communication network (the Enron corpus) to uncover the 'Shadow Organization'—the real, informal power structures that operate beneath the formal org chart.
    
    You will evaluate an employee based on two data streams:
    1. STRUCTURAL METRICS: Graph theory metrics like PageRank (global influence), Betweenness Centrality (information brokerage/bottlenecking), and In/Out ratio (direction of communication flow).
    2. BEHAVIORAL ARTIFACTS: A raw sample of their email body snippets.
    
    INSTRUCTIONS:
    - Do not be fooled by high volume. An 'Information Sink' may receive 500 emails but have zero power.
    - Look for asymmetry: True executives often have high out-degrees, low in-degrees, and send extremely terse, directive emails. 
    - Evaluate Information Brokers: Look for mid-level managers with high betweenness centrality who control the flow of information between disconnected clusters. Are they filtering? Are they bridging?
    - Apply French and Raven's Five Bases of Power to their text. Do people defer to their expertise? Or their formal title?
    - Be highly analytical, objective, and academic in your final comprehensive assessment.
    """

    print(f"🧠 Initiating Deep LLM Inference using gpt-4o with concurrency...")

    # Worker count from env or default
    max_workers = int(os.environ.get("INFER_WORKERS", "8"))

    def _make_user_content(person: dict) -> str:
        stats_str = json.dumps(person.get("stats", {}), indent=2)
        samples = person.get("sample_emails", [])
        snippets = "\n---\n".join(
            [f"To: {', '.join(s.get('to', []))}\nSubject: {s.get('subject')}\nBody: {s.get('body_snippet')}" for s in samples]
        )
        return f"""
        TARGET EMPLOYEE ID: {person['email']}
        HEURISTIC BASELINE ROLE: {person['heuristic_role']}
        
        === STRUCTURAL NETWORK METRICS ===
        {stats_str}
        
        === BEHAVIORAL EMAIL ARTIFACTS ===
        {snippets}
        """

    def _process_person(person: dict) -> dict:
        """Call LLM for a single person with retries and return updated person."""
        user_content = _make_user_content(person)

        # Create a fresh client per worker to avoid shared-state issues
        client_worker = OpenAI(api_key=api_key, base_url=base_url)

        max_attempts = 4
        backoff = 1.0
        for attempt in range(1, max_attempts + 1):
            try:
                response = client_worker.chat.completions.parse(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    response_format=NodeProfile,
                    temperature=0.3
                )
                profile = response.choices[0].message.parsed
                person['deep_profile'] = profile.model_dump()
                return person

            except Exception as e:
                # On last attempt, save error in profile
                if attempt == max_attempts:
                    person['deep_profile'] = {"error": str(e)}
                    return person
                # Otherwise, exponential backoff and retry
                sleep_for = backoff * (2 ** (attempt - 1))
                time.sleep(sleep_for)

    # Submit all tasks to the thread pool and iterate as they complete
    total = len(nodes_to_process)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_process_person, p): p for p in nodes_to_process}

        pbar = tqdm(total=total, desc="Processing nodes")
        for fut in concurrent.futures.as_completed(futures):
            person_result = fut.result()

            # Append and atomically save after each completed item
            enriched_nodes.append(person_result)
            processed_emails.add(person_result.get('email'))

            temp_file = f"{output_file}.tmp"
            with open(temp_file, 'w') as f:
                json.dump(enriched_nodes, f, indent=2)
            os.replace(temp_file, output_file)

            pbar.update(1)
        pbar.close()

    print(f"\n✅ Deep analysis completely finished! Saved to {output_file}")

if __name__ == "__main__":
    INPUT_PATH = "sna_output/llm_power_inputs.json"
    OUTPUT_PATH = "sna_output/llm_deep_enriched_nodes.json"
    
    run_deep_inference(INPUT_PATH, OUTPUT_PATH)