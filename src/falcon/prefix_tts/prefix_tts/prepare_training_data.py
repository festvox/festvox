"""
prepare_training_data.py
========================

Prepare training data for Prefix-LM TTS by combining phoneme sequences 
and audio token sequences into a unified format suitable for language model training.

This script:
1. Reads phoneme data (with audio paths) from JSONL
2. Reads audio codec tokens from numpy files
3. Creates training examples with phoneme-to-audio-token sequences
4. Splits data into train/validation sets
5. Saves in formats suitable for transformer training

Usage:
    python prepare_training_data.py \
        --phonemes_jsonl data/phonemes_ljspeech/ljspeech_phonemes.jsonl \
        --codec_dir data/ljspeech_tokens/vq_codes \
        --output_dir data/training_data \
        --val_split 0.1
"""

import argparse
import json
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random
from collections import Counter
import pickle

class TrainingDataPreparator:
    def __init__(self, phoneme_vocab_path: Optional[str] = None):
        self.phoneme_to_id = {}
        self.id_to_phoneme = {}
        self.audio_token_vocab_size = 1024  # Typical VQ-VAE vocab size
        
        # Special tokens for language modeling
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<START>': 2,
            '<END>': 3,
            '<AUDIO_START>': 4,
            '<AUDIO_END>': 5,
        }
        
        if phoneme_vocab_path and os.path.exists(phoneme_vocab_path):
            self.load_phoneme_vocab(phoneme_vocab_path)
        
    def load_phoneme_vocab(self, vocab_path: str):
        """Load phoneme vocabulary from file."""
        with open(vocab_path, 'r') as f:
            phonemes = [line.strip() for line in f if line.strip()]
        
        # Start after special tokens
        start_id = len(self.special_tokens)
        
        for i, phoneme in enumerate(phonemes):
            if phoneme not in self.phoneme_to_id:
                self.phoneme_to_id[phoneme] = start_id + i
                self.id_to_phoneme[start_id + i] = phoneme
                
        print(f"Loaded {len(self.phoneme_to_id)} phonemes from vocabulary")
    
    def build_phoneme_vocab(self, phoneme_sequences: List[List[str]]):
        """Build phoneme vocabulary from sequences."""
        phoneme_counter = Counter()
        for seq in phoneme_sequences:
            phoneme_counter.update(seq)
        
        # Start after special tokens
        start_id = len(self.special_tokens)
        
        for i, (phoneme, count) in enumerate(phoneme_counter.most_common()):
            if phoneme not in self.phoneme_to_id:
                self.phoneme_to_id[phoneme] = start_id + i
                self.id_to_phoneme[start_id + i] = phoneme
        
        print(f"Built vocabulary with {len(self.phoneme_to_id)} phonemes")
    
    def encode_phonemes(self, phoneme_sequence: List[str]) -> List[int]:
        """Convert phoneme sequence to token IDs."""
        return [self.phoneme_to_id.get(p, self.special_tokens['<UNK>']) 
                for p in phoneme_sequence]
    
    def create_training_sequence(self, phoneme_ids: List[int], audio_tokens: List[int]) -> Dict:
        """Create a training sequence in prefix-LM format.
        
        Format: <START> phoneme1 phoneme2 ... <AUDIO_START> audio1 audio2 ... <AUDIO_END> <END>
        """
        # Offset audio tokens to avoid collision with phoneme vocabulary
        audio_offset = len(self.phoneme_to_id) + len(self.special_tokens)
        offset_audio_tokens = [token + audio_offset for token in audio_tokens]
        
        sequence = [
            self.special_tokens['<START>']
        ] + phoneme_ids + [
            self.special_tokens['<AUDIO_START>']
        ] + offset_audio_tokens + [
            self.special_tokens['<AUDIO_END>'],
            self.special_tokens['<END>']
        ]
        
        # Create labels for language modeling (shifted by 1)
        labels = sequence[1:] + [self.special_tokens['<PAD>']]
        
        return {
            'input_ids': sequence,
            'labels': labels,
            'phoneme_length': len(phoneme_ids),
            'audio_length': len(audio_tokens),
            'total_length': len(sequence)
        }
    
    def load_codec_tokens(self, codec_path: str) -> Optional[List[int]]:
        """Load audio codec tokens from numpy file."""
        try:
            if os.path.exists(codec_path):
                tokens = np.load(codec_path)
                return tokens.tolist()
            else:
                print(f"Warning: Codec file not found: {codec_path}")
                return None
        except Exception as e:
            print(f"Error loading codec file {codec_path}: {e}")
            return None
    
    def prepare_dataset(self, phonemes_jsonl: str, codec_dir: str, 
                       val_split: float = 0.1) -> Tuple[List[Dict], List[Dict]]:
        """Prepare complete dataset from phonemes and codec tokens."""
        
        print("Loading phoneme data...")
        phoneme_data = []
        with open(phonemes_jsonl, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                phoneme_data.append(data)
        
        print(f"Loaded {len(phoneme_data)} phoneme sequences")
        
        # Build vocabulary if not already loaded
        if not self.phoneme_to_id:
            all_phonemes = [data['phonemes'] for data in phoneme_data]
            self.build_phoneme_vocab(all_phonemes)
        
        # Prepare training examples
        training_examples = []
        skipped = 0
        
        print("Creating training examples...")
        for i, data in enumerate(phoneme_data):
            # Get audio path and construct codec path
            audio_path = data['audio_path']
            # Convert audio path to codec path
            # e.g., /path/to/LJ001-0001.wav -> /codec_dir/LJ001-0001.npy
            audio_filename = os.path.basename(audio_path)
            codec_filename = audio_filename.replace('.wav', '.npy')
            codec_path = os.path.join(codec_dir, codec_filename)
            
            # Load codec tokens
            audio_tokens = self.load_codec_tokens(codec_path)
            if audio_tokens is None:
                skipped += 1
                continue
            
            # Encode phonemes
            phoneme_ids = self.encode_phonemes(data['phonemes'])
            
            # Create training sequence
            example = self.create_training_sequence(phoneme_ids, audio_tokens)
            example['audio_path'] = audio_path
            example['text'] = data.get('text', '')
            
            training_examples.append(example)
            
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{len(phoneme_data)} examples")
        
        print(f"Created {len(training_examples)} training examples")
        print(f"Skipped {skipped} examples due to missing codec data")
        
        # Split into train/validation
        random.shuffle(training_examples)
        val_size = int(len(training_examples) * val_split)
        
        val_examples = training_examples[:val_size]
        train_examples = training_examples[val_size:]
        
        print(f"Split: {len(train_examples)} train, {len(val_examples)} validation")
        
        return train_examples, val_examples
    
    def save_dataset(self, train_examples: List[Dict], val_examples: List[Dict], 
                    output_dir: str):
        """Save dataset and vocabulary files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save training examples
        with open(os.path.join(output_dir, 'train.jsonl'), 'w') as f:
            for example in train_examples:
                f.write(json.dumps(example) + '\n')
        
        with open(os.path.join(output_dir, 'val.jsonl'), 'w') as f:
            for example in val_examples:
                f.write(json.dumps(example) + '\n')
        
        # Save vocabulary
        vocab_info = {
            'phoneme_to_id': self.phoneme_to_id,
            'id_to_phoneme': self.id_to_phoneme,
            'special_tokens': self.special_tokens,
            'audio_token_offset': len(self.phoneme_to_id) + len(self.special_tokens),
            'total_vocab_size': len(self.phoneme_to_id) + len(self.special_tokens) + self.audio_token_vocab_size
        }
        
        with open(os.path.join(output_dir, 'vocab.json'), 'w') as f:
            json.dump(vocab_info, f, indent=2)
        
        # Save as pickle for easy loading
        with open(os.path.join(output_dir, 'vocab.pkl'), 'wb') as f:
            pickle.dump(vocab_info, f)
        
        # Save dataset statistics
        train_lengths = [ex['total_length'] for ex in train_examples]
        val_lengths = [ex['total_length'] for ex in val_examples]
        
        stats = {
            'train_size': len(train_examples),
            'val_size': len(val_examples),
            'avg_train_length': sum(train_lengths) / len(train_lengths),
            'avg_val_length': sum(val_lengths) / len(val_lengths),
            'max_train_length': max(train_lengths),
            'max_val_length': max(val_lengths),
            'min_train_length': min(train_lengths),
            'min_val_length': min(val_lengths),
            'vocab_size': vocab_info['total_vocab_size']
        }
        
        with open(os.path.join(output_dir, 'dataset_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Dataset saved to {output_dir}")
        print(f"Vocabulary size: {vocab_info['total_vocab_size']}")
        print(f"Average sequence length: train={stats['avg_train_length']:.1f}, val={stats['avg_val_length']:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Prepare training data for Prefix-LM TTS")
    parser.add_argument("--phonemes_jsonl", type=str, required=True,
                       help="Path to phonemes JSONL file")
    parser.add_argument("--codec_dir", type=str, required=True,
                       help="Directory containing codec token files (.npy)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for training data")
    parser.add_argument("--phoneme_vocab", type=str, default=None,
                       help="Path to phoneme vocabulary file (optional)")
    parser.add_argument("--val_split", type=float, default=0.1,
                       help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible splits")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create preparator
    preparator = TrainingDataPreparator(args.phoneme_vocab)
    
    # Prepare dataset
    train_examples, val_examples = preparator.prepare_dataset(
        args.phonemes_jsonl, args.codec_dir, args.val_split
    )
    
    # Save dataset
    preparator.save_dataset(train_examples, val_examples, args.output_dir)
    
    print("Dataset preparation completed successfully!")


if __name__ == "__main__":
    main()
