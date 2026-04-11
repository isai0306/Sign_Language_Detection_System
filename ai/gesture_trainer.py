"""
SignAI - Enhanced Gesture Training Module
Improved CNN architecture and training process for better accuracy
"""

import numpy as np
import pickle
import os
from datetime import datetime
from collections import defaultdict

try:
    from sklearn.model_selection import train_test_split
    from sklearn.utils import class_weight
    from tensorflow import keras
    from tensorflow.keras import layers, models, regularizers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    import tensorflow as tf
    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False
    print("Warning: Training libraries not available")


class GestureTrainer:
    """
    Enhanced gesture training with better CNN architecture
    """
    
    def __init__(self, data_dir='training_data'):
        """Initialize trainer"""
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        self.samples = defaultdict(list)
        self.load_existing_data()
    
    def add_sample(self, gesture_name, landmarks):
        """
        Add a training sample with validation
        
        Args:
            gesture_name: Name of gesture
            landmarks: Normalized landmarks array (63 features)
        """
        if len(landmarks) != 63:
            print(f"Warning: Expected 63 landmarks, got {len(landmarks)}")
            return False
        
        # Validate landmarks are not all zeros
        if np.sum(np.abs(landmarks)) < 0.01:
            print("Warning: Landmarks appear to be all zeros")
            return False
        
        self.samples[gesture_name].append(landmarks)
        print(f"✅ Added sample for {gesture_name} (Total: {len(self.samples[gesture_name])})")
        return True
    
    def save_samples(self):
        """Save collected samples to disk"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(self.data_dir, f'samples_{timestamp}.pkl')
            
            with open(filename, 'wb') as f:
                pickle.dump(dict(self.samples), f)
            
            print(f"✅ Saved {len(self.samples)} gesture types to {filename}")
            return filename
        except Exception as e:
            print(f"❌ Error saving samples: {e}")
            return None
    
    def load_existing_data(self):
        """Load previously collected samples"""
        try:
            sample_files = [f for f in os.listdir(self.data_dir) if f.startswith('samples_')]
            
            if not sample_files:
                print("No existing training data found")
                return
            
            for filename in sample_files:
                filepath = os.path.join(self.data_dir, filename)
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    for gesture, samples in data.items():
                        self.samples[gesture].extend(samples)
            
            print(f"✅ Loaded existing data: {self.get_sample_counts()}")
        except Exception as e:
            print(f"Error loading existing data: {e}")
    
    def get_sample_counts(self):
        """Get count of samples per gesture"""
        return {gesture: len(samples) for gesture, samples in self.samples.items()}
    
    def clear_samples(self, gesture_name=None):
        """Clear samples for a gesture or all gestures"""
        if gesture_name:
            if gesture_name in self.samples:
                del self.samples[gesture_name]
                print(f"✅ Cleared samples for {gesture_name}")
        else:
            self.samples.clear()
            print("✅ Cleared all samples")
    
    def augment_data(self, landmarks):
        """
        Data augmentation for hand landmarks
        
        Args:
            landmarks: Original landmarks (21, 3)
            
        Returns:
            augmented_samples: List of augmented landmark arrays
        """
        augmented = []
        lm = landmarks.copy()
        
        # Original
        augmented.append(lm)
        
        # Small rotation (±10 degrees)
        for angle in [-10, 10]:
            rad = np.radians(angle)
            cos_a, sin_a = np.cos(rad), np.sin(rad)
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            lm_rotated = lm.copy()
            lm_rotated[:, :2] = lm[:, :2] @ rotation_matrix.T
            augmented.append(lm_rotated)
        
        # Small scaling (±10%)
        for scale in [0.9, 1.1]:
            lm_scaled = lm * scale
            augmented.append(lm_scaled)
        
        # Add small noise
        noise = np.random.normal(0, 0.01, lm.shape)
        lm_noisy = lm + noise
        augmented.append(lm_noisy)
        
        return augmented
    
    def train_model(self, model_type='neural_network', model_path='static/models/gesture_model.h5', 
                    use_augmentation=True, epochs=150, sequence_length=25):
        """
        Train gesture model (CNN or LSTM sequence classifier).
        
        Args:
            model_type: 'neural_network' (CNN), 'lstm' (sequence), or 'random_forest' (maps to CNN)
            model_path: Where to save trained model
            use_augmentation: Whether to use data augmentation
            epochs: Number of training epochs
            sequence_length: Timesteps for LSTM
            
        Returns:
            accuracy: Model accuracy on test set
        """
        if not TRAINING_AVAILABLE:
            print("❌ Training libraries not installed")
            return None

        if model_type == 'lstm':
            return self._train_lstm_model(
                model_path, use_augmentation, epochs, sequence_length
            )

        if model_type == 'random_forest':
            print("⚠️ Random Forest not implemented; training CNN instead.")
            model_type = 'neural_network'
        
        # Prepare data
        X, y, gesture_classes = self._prepare_training_data(use_augmentation)
        
        if len(X) < 20:
            print(f"❌ Not enough samples. Need at least 20, have {len(X)}.")
            return None
        
        if len(gesture_classes) < 2:
            print("❌ Need at least 2 different gestures to train")
            return None
        
        print(f"Training with {len(X)} samples across {len(gesture_classes)} gestures")
        print(f"Gesture classes: {gesture_classes}")
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training: {len(X_train)} samples, Testing: {len(X_test)} samples")
        
        # Train CNN
        accuracy = self._train_enhanced_cnn(
            X_train, X_test, y_train, y_test, model_path, epochs
        )
        
        # Save gesture classes
        classes_path = model_path.replace('.h5', '_classes.pkl')
        with open(classes_path, 'wb') as f:
            pickle.dump(gesture_classes, f)
        
        if accuracy:
            print(f"✅ Model trained with {accuracy*100:.2f}% accuracy")
            print(f"✅ Saved to {model_path}")
        
        return accuracy
    
    def _prepare_training_data(self, use_augmentation=True):
        """Convert samples to training arrays with optional augmentation"""
        X = []
        y = []
        gesture_classes = sorted(list(self.samples.keys()))
        
        for gesture_idx, gesture_name in enumerate(gesture_classes):
            for landmarks in self.samples[gesture_name]:
                # Reshape to (21, 3)
                lm = np.array(landmarks).reshape(21, 3)
                
                if use_augmentation:
                    # Generate augmented samples
                    augmented_samples = self.augment_data(lm)
                    for aug_lm in augmented_samples:
                        X.append(aug_lm)
                        y.append(gesture_idx)
                else:
                    X.append(lm)
                    y.append(gesture_idx)
        
        # Reshape for CNN: (samples, 21, 3, 1)
        X = np.array(X).reshape(-1, 21, 3, 1)
        y = np.array(y)
        
        return X, y, gesture_classes

    def _landmark_to_flat_normalized(self, lm_21_3):
        """Wrist-centered, scale-normalized flatten (63,) — matches runtime pipeline."""
        lm = np.array(lm_21_3, dtype=np.float32).reshape(21, 3)
        wrist = lm[0].copy()
        lm = lm - wrist
        d = np.max(np.linalg.norm(lm, axis=1))
        if d > 1e-6:
            lm = lm / d
        return lm.reshape(-1)

    def _prepare_flat_vectors(self, use_augmentation=True):
        """Per-frame flat vectors for sequence model training."""
        X = []
        y = []
        gesture_classes = sorted(list(self.samples.keys()))
        for gesture_idx, gesture_name in enumerate(gesture_classes):
            for landmarks in self.samples[gesture_name]:
                lm = np.array(landmarks).reshape(21, 3)
                if use_augmentation:
                    for aug_lm in self.augment_data(lm):
                        X.append(self._landmark_to_flat_normalized(aug_lm))
                        y.append(gesture_idx)
                else:
                    X.append(self._landmark_to_flat_normalized(lm))
                    y.append(gesture_idx)
        return np.array(X, dtype=np.float32), np.array(y), gesture_classes

    def _build_lstm_sequences(self, X_flat, y, seq_len, samples_per_class=96):
        rng = np.random.default_rng(42)
        by_class = defaultdict(list)
        for xi, yi in zip(X_flat, y):
            by_class[int(yi)].append(xi)
        X_seq, y_seq = [], []
        for cls, frames in by_class.items():
            arr = np.stack(frames) if len(frames) > 1 else np.stack([frames[0]] * 2)
            n_gen = max(samples_per_class, min(200, len(arr) * 10))
            for _ in range(n_gen):
                idx = rng.integers(0, len(arr), size=seq_len)
                clip = arr[idx].copy()
                clip += rng.normal(0, 0.008, clip.shape).astype(np.float32)
                X_seq.append(clip)
                y_seq.append(cls)
        return np.stack(X_seq, dtype=np.float32), np.array(y_seq)

    def _train_lstm_model(self, model_path, use_augmentation, epochs, sequence_length):
        X_flat, y_flat, gesture_classes = self._prepare_flat_vectors(use_augmentation)
        if len(X_flat) < 20 or len(gesture_classes) < 2:
            print("❌ Not enough data for LSTM training")
            return None
        X_seq, y_seq = self._build_lstm_sequences(X_flat, y_flat, sequence_length)
        X_train, X_test, y_train, y_test = train_test_split(
            X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
        )
        num_classes = len(np.unique(y_train))
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        model = models.Sequential([
            layers.Input(shape=(sequence_length, 63)),
            layers.Masking(),
            layers.LSTM(96, return_sequences=True),
            layers.Dropout(0.35),
            layers.LSTM(48),
            layers.Dropout(0.35),
            layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            layers.Dense(num_classes, activation='softmax'),
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=18, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=1),
            ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        ]
        class_weights = class_weight.compute_class_weight(
            'balanced', classes=np.unique(y_train), y=y_train
        )
        cw = dict(enumerate(class_weights))
        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=16,
            callbacks=callbacks,
            class_weight=cw,
            verbose=1,
        )
        _, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        classes_path = model_path.replace('.h5', '_classes.pkl')
        with open(classes_path, 'wb') as f:
            pickle.dump(gesture_classes, f)
        print(f"✅ LSTM saved to {model_path} accuracy={test_accuracy*100:.2f}%")
        return test_accuracy
    
    def _train_enhanced_cnn(self, X_train, X_test, y_train, y_test, model_path, epochs):
        """
        Train enhanced CNN with better architecture
        """
        num_classes = len(np.unique(y_train))
        
        print("Building enhanced CNN architecture...")
        
        # Build improved CNN model
        model = models.Sequential([
            # Input: (21, 3, 1)
            
            # First Conv Block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                         input_shape=(21, 3, 1)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 1)),
            layers.Dropout(0.25),
            
            # Second Conv Block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 1)),
            layers.Dropout(0.25),
            
            # Third Conv Block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            
            # Dense layers
            layers.Dense(256, activation='relu', 
                        kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(128, activation='relu',
                        kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile with optimizer and learning rate
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(model.summary())
        
        # Callbacks
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                model_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Calculate class weights for imbalanced data
        class_weights = class_weight.compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights_dict = dict(enumerate(class_weights))
        
        print("\nTraining CNN model...")
        print(f"Class weights: {class_weights_dict}")
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=16,
            callbacks=callbacks,
            class_weight=class_weights_dict,
            verbose=1
        )
        
        # Evaluate
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        print(f"\n📊 Final Results:")
        print(f"   Test Accuracy: {test_accuracy*100:.2f}%")
        print(f"   Test Loss: {test_loss:.4f}")
        
        # Show per-class accuracy
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        from sklearn.metrics import classification_report
        
        print("\n📈 Per-Class Performance:")
        print(classification_report(y_test, y_pred, target_names=[str(i) for i in range(num_classes)]))
        
        return test_accuracy
    
    def get_training_report(self):
        """Generate training report"""
        ready, message = self._is_ready_to_train()
        
        report = {
            'total_gestures': len(self.samples),
            'total_samples': sum(len(s) for s in self.samples.values()),
            'sample_counts': self.get_sample_counts(),
            'ready_to_train': (ready, message)
        }
        return report
    
    def _is_ready_to_train(self):
        """Check if enough data to train"""
        if len(self.samples) < 2:
            return False, "Need at least 2 different gestures"
        
        total_samples = sum(len(s) for s in self.samples.values())
        if total_samples < 20:
            return False, f"Need at least 20 total samples (have {total_samples})"
        
        # Check each gesture has minimum samples
        min_samples_per_gesture = 10
        for gesture, samples in self.samples.items():
            if len(samples) < min_samples_per_gesture:
                return False, f"{gesture} needs {min_samples_per_gesture - len(samples)} more samples"
        
        return True, "Ready to train! ✅"


# Testing function
def test_trainer():
    """Test trainer with dummy data"""
    print("Testing Enhanced Gesture Trainer")
    print("=" * 60)
    
    trainer = GestureTrainer()
    
    # Generate more realistic dummy samples
    gestures = ['HELLO', 'YES', 'HELP', 'THANK_YOU']
    
    for gesture in gestures:
        print(f"\nGenerating samples for {gesture}...")
        for i in range(15):
            # Create slightly varied dummy landmarks
            base = np.random.rand(21, 3)
            # Normalize
            base = base - base[0]
            max_dist = np.max(np.linalg.norm(base, axis=1))
            if max_dist > 0:
                base = base / max_dist
            
            landmarks = base.flatten().tolist()
            trainer.add_sample(gesture, landmarks)
    
    # Get report
    report = trainer.get_training_report()
    print(f"\n📊 Training Report:")
    print(f"   Total gestures: {report['total_gestures']}")
    print(f"   Total samples: {report['total_samples']}")
    print(f"   Sample counts: {report['sample_counts']}")
    print(f"   Ready to train: {report['ready_to_train']}")
    
    print("\n✅ Test complete")


if __name__ == "__main__":
    test_trainer()