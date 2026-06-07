"""
Protein Lookup Module
Provides sequence-to-ID and ID-to-sequence mappings for the drug discovery pipeline.
"""

import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOOKUP_FILE = os.path.join(BASE_DIR, 'protein_sequence_lookup.json')


class ProteinLookup:
    """Singleton class for protein sequence lookups"""
    
    _instance = None
    _sequence_to_id = None
    _id_to_sequence = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ProteinLookup, cls).__new__(cls)
            cls._instance._load_data()
        return cls._instance
    
    def _load_data(self):
        """Load lookup data from JSON file"""
        if not os.path.exists(LOOKUP_FILE):
            raise FileNotFoundError(
                f"Protein lookup file not found: {LOOKUP_FILE}\n"
                "Run extract_kiba_proteins.py first to generate mappings."
            )
        
        with open(LOOKUP_FILE, 'r') as f:
            data = json.load(f)
        
        self._sequence_to_id = data['sequence_to_id']
        self._id_to_sequence = data['id_to_sequence']
        
        print(f"✓ Loaded {len(self._sequence_to_id)} protein mappings")
    
    def get_id_from_sequence(self, sequence, allow_partial=True):
        """
        Get UniProt ID from protein sequence.
        
        Args:
            sequence: Amino acid sequence
            allow_partial: Allow partial matching for truncated sequences
            
        Returns:
            UniProt ID or None if not found
        """
        # Exact match
        if sequence in self._sequence_to_id:
            return self._sequence_to_id[sequence]
        
        # Partial match
        if allow_partial:
            for seq, pid in self._sequence_to_id.items():
                if sequence in seq or seq in sequence:
                    print(f"⚠ Partial match: using {pid}")
                    return pid
        
        return None
    
    def get_sequence_from_id(self, protein_id):
        """Get sequence from UniProt ID"""
        return self._id_to_sequence.get(protein_id)
    
    def get_all_ids(self):
        """Return all available UniProt IDs"""
        return list(self._id_to_sequence.keys())
    
    def get_stats(self):
        """Return statistics about the lookup data"""
        return {
            'total_proteins': len(self._sequence_to_id),
            'avg_sequence_length': sum(len(s) for s in self._sequence_to_id.keys()) / len(self._sequence_to_id)
        }


# Convenience functions for direct import
_lookup = None

def _get_lookup():
    global _lookup
    if _lookup is None:
        _lookup = ProteinLookup()
    return _lookup

def get_protein_id(sequence, allow_partial=True):
    """Get UniProt ID from sequence"""
    return _get_lookup().get_id_from_sequence(sequence, allow_partial)

def get_sequence(protein_id):
    """Get sequence from UniProt ID"""
    return _get_lookup().get_sequence_from_id(protein_id)

def get_all_protein_ids():
    """Get all available protein IDs"""
    return _get_lookup().get_all_ids()