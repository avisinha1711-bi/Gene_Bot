#!/usr/bin/env python3
"""
Genetics Chatbot with Note-Taking System
Main entry point for the genetics chatbot application.
"""

from chatbot import GeneticsChatbot
from note_manager import NoteManager
import os

def main():
    """Main function to run the genetics chatbot."""
    print("=" * 60)
    print("       🧬 GENETICS CHATBOT & NOTE MANAGER 🧬")
    print("=" * 60)
    print("Welcome! I can help you with genetics concepts and save your notes.")
    
    # Initialize components
    chatbot = GeneticsChatbot()
    note_manager = NoteManager()
    
    while True:
        print("\n" + "-" * 40)
        print("What would you like to do?")
        print("1. 💬 Chat about genetics")
        print("2. 📝 Save a new note")
        print("3. 📖 View all notes")
        print("4. 🔍 Search notes")
        print("5. 🗑️ Delete a note")
        print("6. 🚪 Exit")
        print("-" * 40)
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            chat_mode(chatbot, note_manager)
        elif choice == '2':
            save_note_mode(note_manager)
        elif choice == '3':
            view_notes_mode(note_manager)
        elif choice == '4':
            search_notes_mode(note_manager)
        elif choice == '5':
            delete_note_mode(note_manager)
        elif choice == '6':
            print("\nThank you for using the Genetics Chatbot! Goodbye! 👋")
            break
        else:
            print("❌ Invalid choice. Please enter a number between 1-6.")

def chat_mode(chatbot, note_manager):
    """Handle chat interactions about genetics."""
    print("\n💬 Genetics Chat Mode (type 'back' to return to main menu)")
    print("You can ask about DNA, RNA, proteins, mutations, inheritance, etc.")
    
    while True:
        user_input = input("\n🧬 You: ").strip()
        
        if user_input.lower() in ['back', 'exit', 'quit']:
            break
        elif user_input.lower() == '':
            continue
            
        response = chatbot.get_response(user_input)
        print(f"🤖 Bot: {response}")
        
        # Offer to save as note if response is informative
        if len(response) > 50:  # If response is substantial
            save_option = input("\n💡 Would you like to save this information as a note? (y/n): ").strip().lower()
            if save_option in ['y', 'yes']:
                title = input("Enter a title for this note: ").strip() or f"Genetics Note about {user_input[:30]}..."
                note_manager.add_note(title, response, "chat_response")
                print("✅ Note saved successfully!")

def save_note_mode(note_manager):
    """Handle manual note saving."""
    print("\n📝 Create New Note")
    title = input("Note title: ").strip()
    if not title:
        print("❌ Title cannot be empty.")
        return
        
    print("Enter your note content (press Enter twice to finish):")
    content_lines = []
    while True:
        line = input()
        if line == '' and len(content_lines) > 0:
            # Check if last line was also empty (double enter to finish)
            if content_lines[-1] == '':
                content_lines.pop()  # Remove the last empty line
                break
        content_lines.append(line)
    
    content = '\n'.join(content_lines)
    if not content.strip():
        print("❌ Note content cannot be empty.")
        return
        
    category = input("Category (optional): ").strip() or "general"
    note_manager.add_note(title, content, category)
    print("✅ Note saved successfully!")

def view_notes_mode(note_manager):
    """Display all notes."""
    notes = note_manager.get_all_notes()
    if not notes:
        print("\n📭 No notes found.")
        return
        
    print(f"\n📚 Your Notes ({len(notes)} total):")
    for i, note in enumerate(notes, 1):
        print(f"\n{i}. [{note['category']}] {note['title']}")
        print(f"   📅 {note['timestamp']}")
        print(f"   {note['content'][:100]}..." if len(note['content']) > 100 else f"   {note['content']}")

def search_notes_mode(note_manager):
    """Search through notes."""
    query = input("\n🔍 Enter search term: ").strip()
    if not query:
        print("❌ Please enter a search term.")
        return
        
    results = note_manager.search_notes(query)
    if not results:
        print("🔍 No notes found matching your search.")
        return
        
    print(f"\n📖 Search Results ({len(results)} notes):")
    for i, note in enumerate(results, 1):
        print(f"\n{i}. [{note['category']}] {note['title']}")
        print(f"   📅 {note['timestamp']}")
        print(f"   {note['content'][:100]}..." if len(note['content']) > 100 else f"   {note['content']}")

def delete_note_mode(note_manager):
    """Delete a note."""
    notes = note_manager.get_all_notes()
    if not notes:
        print("\n📭 No notes to delete.")
        return
        
    print("\n🗑️ Delete Note")
    for i, note in enumerate(notes, 1):
        print(f"{i}. {note['title']}")
    
    try:
        choice = int(input(f"\nEnter note number to delete (1-{len(notes)}): "))
        if 1 <= choice <= len(notes):
            note_to_delete = notes[choice-1]
            confirm = input(f"Are you sure you want to delete '{note_to_delete['title']}'? (y/n): ").lower()
            if confirm in ['y', 'yes']:
                note_manager.delete_note(note_to_delete['id'])
                print("✅ Note deleted successfully!")
            else:
                print("❌ Deletion cancelled.")
        else:
            print("❌ Invalid note number.")
    except ValueError:
        print("❌ Please enter a valid number.")

if __name__ == "__main__":
    main()
