import json
import duckdb
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import re
from datetime import datetime
import joblib
import os

from core.db import DEFAULT_CLIENT_ID

class EngagementPredictor:
    """
    Free ML model to predict video engagement using only metadata
    No expensive vision APIs needed!
    """
    
    def __init__(self, db_path="reels.duckdb", client_id=DEFAULT_CLIENT_ID):
        self.conn = duckdb.connect(db_path)
        self.client_id = client_id
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.feature_names = []
        self.model_trained = False
    
    def extract_features(self, post_data):
        """Extract features from post metadata"""
        features = {}
        
        # Basic metrics
        features['duration'] = post_data.get('videoDuration', 0) or 0
        features['caption_length'] = len(post_data.get('caption', '') or '')
        features['mentions_count'] = len(post_data.get('mentions', []) or [])
        features['hashtags_count'] = len(post_data.get('hashtags', []) or [])
        
        # Caption analysis
        caption = (post_data.get('caption', '') or '').lower()
        
        # Emotional indicators
        features['has_exclamation'] = 1 if '!' in caption else 0
        features['has_question'] = 1 if '?' in caption else 0
        features['has_emoji'] = 1 if any(ord(c) > 127 for c in caption) else 0
        
        # Content type indicators
        features['is_food_content'] = 1 if any(word in caption for word in ['food', 'eat', 'restaurant', 'bar', 'drink', 'ramen']) else 0
        features['is_event_content'] = 1 if any(word in caption for word in ['event', 'party', 'celebration', 'anniversary']) else 0
        features['is_location_content'] = 1 if post_data.get('locationName') else 0
        
        # Timing features (if timestamp available)
        timestamp = post_data.get('timestamp', '')
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                features['hour_of_day'] = dt.hour
                features['day_of_week'] = dt.weekday()
                features['is_weekend'] = 1 if dt.weekday() >= 5 else 0
            except:
                features['hour_of_day'] = 12  # Default
                features['day_of_week'] = 3   # Default
                features['is_weekend'] = 0
        else:
            features['hour_of_day'] = 12
            features['day_of_week'] = 3
            features['is_weekend'] = 0
        
        # Owner popularity (based on historical data)
        owner = post_data.get('ownerUsername', '')
        owner_posts = self.get_owner_avg_engagement(owner)
        features['owner_avg_engagement'] = owner_posts
        
        # Video quality indicators  
        features['has_music'] = 1 if post_data.get('musicInfo') else 0
        features['is_sponsored'] = 1 if post_data.get('isSponsored') else 0
        
        return features
    
    def get_owner_avg_engagement(self, owner):
        """Get average engagement rate for owner"""
        try:
            results = self.conn.execute("""
                SELECT raw FROM raw_scrapes
                WHERE client_id = ? AND raw LIKE ?
            """, [self.client_id, f'%{owner}%']).fetchall()
            
            engagements = []
            for (raw,) in results:
                try:
                    data = json.loads(raw)
                    if data.get('ownerUsername') == owner:
                        likes = data.get('likesCount', 0) or 0
                        views = data.get('videoViewCount', 0) or 0
                        if views > 0:
                            engagements.append(likes/views*100)
                except:
                    continue
            
            return sum(engagements) / len(engagements) if engagements else 5.0
        except:
            return 5.0  # Default engagement rate
    
    def prepare_training_data(self):
        """Prepare training data from database"""
        results = self.conn.execute(
            "SELECT raw FROM raw_scrapes WHERE client_id = ?",
            [self.client_id],
        ).fetchall()
        
        X = []
        y = []
        
        for (raw,) in results:
            try:
                data = json.loads(raw)
                
                # Calculate target (engagement rate)
                likes = data.get('likesCount', 0) or 0
                views = data.get('videoViewCount', 0) or 0
                
                if views > 0:  # Only include posts with views
                    engagement_rate = (likes / views) * 100
                    
                    # Extract features
                    features = self.extract_features(data)
                    
                    X.append(list(features.values()))
                    y.append(engagement_rate)
                    
                    # Store feature names (only once)
                    if not self.feature_names:
                        self.feature_names = list(features.keys())
                        
            except Exception as e:
                continue
        
        return pd.DataFrame(X, columns=self.feature_names), y
    
    def train_model(self):
        """Train the engagement prediction model"""
        print("🤖 Training engagement prediction model...")
        
        # Prepare data
        X, y = self.prepare_training_data()
        
        if len(X) < 5:
            print("❌ Not enough data to train model (need at least 5 posts)")
            return False
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"✅ Model trained successfully!")
        print(f"   Mean Absolute Error: {mae:.2f}%")
        print(f"   R² Score: {r2:.3f}")
        
        # Feature importance
        feature_importance = list(zip(self.feature_names, self.model.feature_importances_))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n🎯 Most Important Features:")
        for feature, importance in feature_importance[:5]:
            print(f"   {feature}: {importance:.3f}")
        
        self.model_trained = True
        
        # Save model
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names
        }, 'engagement_prediction_model.pkl')
        
        return True
    
    def predict_engagement(self, post_data):
        """Predict engagement rate for a post"""
        if not self.model_trained:
            return None
        
        features = self.extract_features(post_data)
        feature_vector = [features.get(name, 0) for name in self.feature_names]
        
        prediction = self.model.predict([feature_vector])[0]
        return max(0, prediction)  # Ensure non-negative
    
    def analyze_content_optimization(self, sample_post):
        """Analyze what changes could improve engagement"""
        if not self.model_trained:
            return {}
        
        base_prediction = self.predict_engagement(sample_post)
        optimizations = {}
        
        # Test different scenarios
        test_scenarios = {
            'add_location': {'locationName': 'Test Location'},
            'add_hashtags': {'hashtags': ['#test1', '#test2', '#test3']},
            'add_mentions': {'mentions': ['@user1', '@user2']},
            'optimal_duration_30s': {'videoDuration': 30},
            'optimal_duration_60s': {'videoDuration': 60},
            'add_emojis': {'caption': sample_post.get('caption', '') + ' 🔥✨'},
            'weekend_posting': {'timestamp': '2024-01-06T18:00:00.000Z'},  # Saturday 6PM
            'prime_time': {'timestamp': '2024-01-03T18:00:00.000Z'},  # Wednesday 6PM
        }
        
        for scenario, changes in test_scenarios.items():
            test_post = sample_post.copy()
            test_post.update(changes)
            
            new_prediction = self.predict_engagement(test_post)
            improvement = new_prediction - base_prediction
            
            optimizations[scenario] = {
                'predicted_engagement': new_prediction,
                'improvement': improvement,
                'improvement_percent': (improvement / base_prediction * 100) if base_prediction > 0 else 0
            }
        
        return optimizations
    
    def generate_content_recommendations(self):
        """Generate content strategy recommendations"""
        # Analyze historical high performers
        results = self.conn.execute(
            "SELECT raw FROM raw_scrapes WHERE client_id = ?",
            [self.client_id],
        ).fetchall()
        
        high_performers = []
        all_posts = []
        
        for (raw,) in results:
            try:
                data = json.loads(raw)
                likes = data.get('likesCount', 0) or 0
                views = data.get('videoViewCount', 0) or 0
                
                if views > 0:
                    engagement_rate = (likes / views) * 100
                    data['calculated_engagement'] = engagement_rate
                    all_posts.append(data)
                    
                    if engagement_rate > 10:  # High engagement threshold
                        high_performers.append(data)
            except:
                continue
        
        if not high_performers:
            return ["No high-performing posts found to analyze"]
        
        recommendations = []
        
        # Analyze high performers
        avg_duration = sum(p.get('videoDuration', 0) for p in high_performers) / len(high_performers)
        avg_caption_length = sum(len(p.get('caption', '') or '') for p in high_performers) / len(high_performers)
        
        recommendations.append(f"🎯 Optimal video duration: {avg_duration:.1f} seconds (based on top performers)")
        recommendations.append(f"📝 Ideal caption length: {avg_caption_length:.0f} characters")
        
        # Common hashtags in high performers
        all_hashtags = []
        for post in high_performers:
            caption = post.get('caption', '') or ''
            hashtags = re.findall(r'#\w+', caption.lower())
            all_hashtags.extend(hashtags)
        
        if all_hashtags:
            from collections import Counter
            common_hashtags = Counter(all_hashtags).most_common(3)
            hashtag_list = [h[0] for h in common_hashtags]
            recommendations.append(f"🏷️ High-performing hashtags: {', '.join(hashtag_list)}")
        
        # Timing analysis
        timestamps = [p.get('timestamp') for p in high_performers if p.get('timestamp')]
        if timestamps:
            hours = []
            days = []
            for ts in timestamps:
                try:
                    dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    hours.append(dt.hour)
                    days.append(dt.weekday())
                except:
                    continue
            
            if hours:
                avg_hour = sum(hours) / len(hours)
                recommendations.append(f"⏰ Best posting time: around {int(avg_hour)}:00")
            
            if days:
                weekend_posts = sum(1 for d in days if d >= 5)
                if weekend_posts / len(days) > 0.6:
                    recommendations.append("📅 Weekend posting performs better")
                else:
                    recommendations.append("📅 Weekday posting performs better")
        
        return recommendations

def main():
    predictor = EngagementPredictor()
    
    print("🎯 Free Engagement Prediction Model")
    print("=" * 50)
    
    # Train model
    if predictor.train_model():
        
        # Get sample post for testing
        sample_result = predictor.conn.execute(
            "SELECT raw FROM raw_scrapes WHERE client_id = ? LIMIT 1",
            [predictor.client_id],
        ).fetchone()
        
        if sample_result:
            sample_post = json.loads(sample_result[0])
            
            print(f"\n🔮 Testing prediction on sample post:")
            print(f"   Short Code: {sample_post.get('shortCode')}")
            print(f"   Actual Engagement: {((sample_post.get('likesCount', 0) or 0) / (sample_post.get('videoViewCount', 0) or 1) * 100):.2f}%")
            
            predicted = predictor.predict_engagement(sample_post)
            print(f"   Predicted Engagement: {predicted:.2f}%")
            
            # Content optimization analysis
            print(f"\n🚀 Content Optimization Suggestions:")
            optimizations = predictor.analyze_content_optimization(sample_post)
            
            for scenario, data in sorted(optimizations.items(), key=lambda x: x[1]['improvement'], reverse=True)[:5]:
                if data['improvement'] > 0.1:  # Only show meaningful improvements
                    print(f"   {scenario.replace('_', ' ').title()}: +{data['improvement']:.1f}% engagement")
        
        # Generate recommendations
        print(f"\n💡 Content Strategy Recommendations:")
        recommendations = predictor.generate_content_recommendations()
        for rec in recommendations:
            print(f"   {rec}")
        
        print(f"\n💾 Model saved as 'engagement_prediction_model.pkl'")
        print("You can now use this model to predict engagement for future posts!")

if __name__ == "__main__":
    main()
