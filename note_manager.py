import json
import os
from datetime import datetime
import uuid

class NoteManager:
    def __init__(self, notes_file="data/notes.json"):
        self.notes_file = notes_file
        self.notes = self.load_notes()
    
    def load_notes(self):
        """Load notes from JSON file."""
        try:
            if os.path.exists(self.notes_file):
                with open(self.notes_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(self.notes_file), exist_ok=True)
                return []
        except Exception as e:
            print(f"Warning: Could not load notes: {e}")
            return []
    
    def save_notes(self):
        """Save notes to JSON file."""
        try:
            with open(self.notes_file, 'w', encoding='utf-8') as f:
                json.dump(self.notes, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving notes: {e}")
            return False
    
    def add_note(self, title, content, category="general"):
        """Add a new note."""
        note = {
            "id": str(uuid.uuid4()),
            "title": title,
            "content": content,
            "category": category,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.notes.append(note)
        self.save_notes()
        return note["id"]
    
    def get_all_notes(self):
        """Get all notes, sorted by timestamp (newest first)."""
        return sorted(self.notes, key=lambda x: x["timestamp"], reverse=True)
    
    def get_notes_by_category(self, category):
        """Get notes by category."""
        return [note for note in self.notes if note["category"].lower() == category.lower()]
    
    def search_notes(self, query):
        """Search notes by title or content."""
        query = query.lower()
        results = []
        
        for note in self.notes:
            if (query in note["title"].lower() or 
                query in note["content"].lower() or 
                query in note["category"].lower()):
                results.append(note)
        
        return sorted(results, key=lambda x: x["timestamp"], reverse=True)
    
    def delete_note(self, note_id):
        """Delete a note by ID."""
        initial_count = len(self.notes)
        self.notes = [note for note in self.notes if note["id"] != note_id]
        
        if len(self.notes) < initial_count:
            self.save_notes()
            return True
        return False
    
    def update_note(self, note_id, title=None, content=None, category=None):
        """Update an existing note."""
        for note in self.notes:
            if note["id"] == note_id:
                if title is not None:
                    note["title"] = title
                if content is not None:
                    note["content"] = content
                if category is not None:
                    note["category"] = category
                note["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.save_notes()
                return True
        return False
    
    def get_note_categories(self):
        """Get list of all categories used in notes."""
        categories = set(note["category"] for note in self.notes)
        return sorted(list(categories))
    
    def get_note_stats(self):
        """Get statistics about notes."""
        total_notes = len(self.notes)
        categories = self.get_note_categories()
        category_counts = {}
        
        for category in categories:
            category_counts[category] = len(self.get_notes_by_category(category))
        
        return {
            "total_notes": total_notes,
            "categories": categories,
            "category_counts": category_counts
        }
