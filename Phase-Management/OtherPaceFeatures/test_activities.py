#!/usr/bin/env python3
"""
Test script for pace management activities endpoints
"""

import requests
import json
import os
import tempfile
import numpy as np
import soundfile as sf

# Test server URL
BASE_URL = "http://localhost:8000"

def create_test_audio(duration=5, sample_rate=16000):
    """Create a test audio file for testing"""
    # Generate a simple sine wave with some pauses
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create speech-like signal with pauses
    signal = np.zeros_like(t)
    
    # Add speech segments (sine waves at different frequencies)
    speech_segments = [
        (0, 1, 440),      # 1 second at 440Hz
        (1.5, 2.5, 880),  # 1 second at 880Hz (pause from 1-1.5s)
        (3, 4, 660),      # 1 second at 660Hz (pause from 2.5-3s)
        (4.5, 5, 220)     # 0.5 seconds at 220Hz (pause from 4-4.5s)
    ]
    
    for start, end, freq in speech_segments:
        start_idx = int(start * sample_rate)
        end_idx = int(end * sample_rate)
        segment_t = t[start_idx:end_idx]
        signal[start_idx:end_idx] = 0.3 * np.sin(2 * np.pi * freq * segment_t)
    
    # Add some noise
    noise = 0.05 * np.random.randn(len(signal))
    signal += noise
    
    return signal, sample_rate

def test_activity_types():
    """Test getting activity types"""
    print("Testing activity types endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/activity-types/")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Activity types retrieved: {len(data['activity_types'])} types")
            print(f"   Available activities: {list(data['activity_types'].keys())}")
            print(f"   Available badges: {len(data['badges'])} badges")
            return True
        else:
            print(f"❌ Failed to get activity types: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error testing activity types: {e}")
        return False

def test_real_time_analysis():
    """Test real-time analysis endpoint"""
    print("\nTesting real-time analysis endpoint...")
    try:
        test_data = {
            "activityType": "pacing_curve",
            "duration": 30,
            "timestamp": 1234567890
        }
        
        response = requests.post(
            f"{BASE_URL}/real-time-analysis/",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Real-time analysis successful")
            print(f"   Score: {data.get('score', 'N/A')}")
            print(f"   Feedback: {data.get('feedback', 'N/A')}")
            return True
        else:
            print(f"❌ Real-time analysis failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error testing real-time analysis: {e}")
        return False

def test_analyze_activity():
    """Test activity analysis endpoint"""
    print("\nTesting activity analysis endpoint...")
    try:
        # Create test audio
        signal, sr = create_test_audio(duration=3)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            sf.write(tmp_file.name, signal, sr)
            
            # Test the endpoint
            with open(tmp_file.name, "rb") as audio_file:
                files = {"file": ("test.wav", audio_file, "audio/wav")}
                data = {"activityType": "pacing_curve"}
                
                response = requests.post(
                    f"{BASE_URL}/analyze-activity/",
                    files=files,
                    data=data
                )
            
            # Clean up
            os.unlink(tmp_file.name)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Activity analysis successful")
            print(f"   Activity: {result.get('activity_type', 'N/A')}")
            print(f"   Final Score: {result.get('final_score', 'N/A')}")
            print(f"   Duration: {result.get('duration', 'N/A'):.2f}s")
            print(f"   WPM: {result.get('average_wpm', 'N/A')}")
            print(f"   Badges: {len(result.get('new_badges', []))}")
            return True
        else:
            print(f"❌ Activity analysis failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error testing activity analysis: {e}")
        return False

def test_user_progress():
    """Test user progress endpoints"""
    print("\nTesting user progress endpoints...")
    try:
        # Test get progress
        response = requests.get(f"{BASE_URL}/user-progress/")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ User progress retrieved")
            print(f"   Total sessions: {data.get('total_sessions', 'N/A')}")
            print(f"   Current streak: {data.get('current_streak', 'N/A')}")
            print(f"   Current level: {data.get('current_level', 'N/A')}")
        else:
            print(f"❌ Failed to get user progress: {response.status_code}")
            return False
        
        # Test update progress
        progress_data = {
            "activity_type": "pacing_curve",
            "score": 85.5,
            "duration": 120,
            "badges_earned": ["steady_flow"]
        }
        
        response = requests.post(
            f"{BASE_URL}/update-progress/",
            json=progress_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            print(f"✅ Progress update successful")
            return True
        else:
            print(f"❌ Progress update failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error testing user progress: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Pace Management Activities API")
    print("=" * 50)
    
    tests = [
        test_activity_types,
        test_real_time_analysis,
        test_analyze_activity,
        test_user_progress
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The activities API is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the server logs for details.")
    
    return passed == total

if __name__ == "__main__":
    main()
