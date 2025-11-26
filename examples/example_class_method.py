#!/usr/bin/env python3
"""
Example: Using LLM Expect with Class Methods

This example demonstrates how to use the @llm_expect decorator with class methods.
LLM Expect supports decorating instance methods directly, handling `self` binding automatically.
"""

import os
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
from llm_expect import llm_expect

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


class ReleaseNoteGenerator:
    """Example class that generates release notes from JIRA stories."""
    
    def __init__(self):
        """Initialize the generator with OpenAI client."""
        self.client = OpenAI(api_key=api_key) if api_key else None
        self.model = "gpt-4o-mini"
    
    @llm_expect(
        dataset="examples/datasets/release_notes.jsonl",
        tests=["custom_judge"],
        judge_provider="openai",
        judge_model="gpt-4o-mini"
    )
    def create_release_note_for_story(
        self, 
        title: str, 
        description: str, 
        labels: List[str]
    ) -> str:
        """
        Create a single-sentence release note using OpenAI's ChatGPT.
        
        Args:
            title: The story title
            description: The story description
            labels: List of labels associated with the story
            
        Returns:
            A formatted release note as HTML
        """
        if self.client is None:
            return f"<strong>{title}-</strong> Issue resolved."
        
        try:
            labels_str = ', '.join(labels) if labels else 'No labels'
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant that writes clear, concise release notes."},
                {
                    "role": "user",
                    "content": (
                        f"Create a single-sentence release note for the following issue:\n\n"
                        f"Title: {title}\n"
                        f"Description: {description}\n"
                        f"Labels: {labels_str}\n\n"
                        f"The note should be in plain language and be as short as possible. "
                        f"Assume this is a non-technical audience.\n"
                        f"Format: <strong>Title-</strong> Description of what was fixed/added.\n"
                        f"DO NOT include ``` or html markers in the response.\n"
                        f"ONLY return the formatted release note."
                    )
                }
            ]
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=200,
                timeout=30
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating release note: {e}")
            return f"<strong>{title}-</strong> Issue resolved."


if __name__ == "__main__":
    print("🚀 Running Class Method Evaluation Example")
    
    if not api_key:
        print("   ⚠️  Skipped: OPENAI_API_KEY not set.")
        print("   💡 Set OPENAI_API_KEY in your .env file to run this example.")
    else:
        # Run the evaluation
        print("\n📊 Running LLM Expect evaluation...")
        
        # Create an instance
        generator = ReleaseNoteGenerator()
        
        # Run evaluation on the bound method
        results = generator.create_release_note_for_story.run_eval()
        
        print(f"\n   Result: {'✅ PASSED' if results['passed'] else '❌ FAILED'}")
        print(f"   Success Rate: {results['summary']['success_rate']:.1%}")
        print(f"   Total Tests: {results['summary']['total_tests']}")
        print(f"   Passed: {results['summary']['passed_tests']}")
        print(f"   Failed: {results['summary']['failed_tests']}")
        
        # You can also use the class directly for normal usage
        print("\n💼 Using the class directly (outside of evaluation):")
        note = generator.create_release_note_for_story(
            title="Fix login crash",
            description="Fixed a crash on login screen when user enters empty password.",
            labels=["login", "urgent"]
        )
        print(f"   Generated note: {note}")
