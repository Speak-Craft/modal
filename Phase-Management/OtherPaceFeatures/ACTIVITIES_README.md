# Pace Management Activities System

A comprehensive real-time activity system for improving speech pace management skills through interactive challenges, gamification, and AI-powered feedback.

## 🎯 Overview

The Pace Management Activities system provides 12 different interactive activities designed to improve various aspects of speech pace management:

### Speech Rate Activities
1. **Pacing Curve Challenge** - Visualize and improve pace stability
2. **Rate Match Drill** - Practice with metronome at target pace (100-150 WPM)
3. **Speed Shift Exercise** - Master intentional pace variation for expressiveness
4. **Consistency Tracker** - Long-term WPM consistency monitoring

### Speech Pause Activities
5. **Pause Timing Drill** - Practice strategic pause insertion
6. **Excessive Pause Elimination** - Remove awkward >5s pauses
7. **Pause for Impact Challenge** - Master dramatic emphasis pauses
8. **Pause Rhythm Training** - Develop consistent pause patterns
9. **Confidence Pause Coach** - Replace fillers with confident pauses

### Advanced Activities
10. **Golden Ratio Speech Challenge** - Achieve natural 1.618x pause ratios
11. **Pause Entropy Game** - Reduce pause randomness
12. **Cognitive Pause Mastery** - Master complex explanation pauses

## 🏆 Gamification System

### Badge System
- **Steady Flow** - WPM standard deviation < 20
- **Tempo Master** - 3 consecutive sessions at target pace
- **Dynamic Speaker** - Controlled pace variations
- **Consistency Streak** - 7 consecutive recordings
- **Pause Precision** - 70% optimal pause timing
- **No Dead Air** - Zero excessive pauses
- **Impact Pause** - High comprehension scores
- **Rhythm Speaker** - 70% rhythm consistency
- **Confident Voice** - <2 fillers per minute
- **Golden Ratio Master** - 80% golden ratio compliance
- **Entropy Controller** - Low pause randomness
- **Cognitive Master** - 85% cognitive pause effectiveness

### Level System
- **Bronze** - 30% variation threshold
- **Silver** - 20% variation threshold
- **Gold** - 10% variation threshold
- **Platinum** - 5% variation threshold

## 🚀 Features

### Real-time Analysis
- Live feedback during recording
- Instant score updates
- Real-time metric tracking
- Dynamic progress visualization

### Advanced Analytics
- Comprehensive performance metrics
- Detailed activity-specific scoring
- Progress tracking over time
- Personalized recommendations

### Interactive UI
- Modern, responsive design
- Smooth animations and transitions
- Real-time visual feedback
- Intuitive activity selection

## 📁 File Structure

```
modal/Phase-Management/OtherPaceFeatures/
├── activity_endpoints.py          # Backend API endpoints
├── test_activities.py            # Test script
├── ACTIVITIES_README.md          # This documentation
└── app.py                        # Main FastAPI app (updated)

frontend/src/components/
├── PaceManagementActivity.jsx    # Main activity component
├── PaceManagementHome.jsx        # Updated with activity link
└── App.jsx                       # Updated routing
```

## 🔧 API Endpoints

### Activity Management
- `GET /activity-types/` - Get all available activities
- `POST /real-time-analysis/` - Real-time analysis during recording
- `POST /analyze-activity/` - Complete activity analysis
- `GET /user-progress/` - Get user progress
- `POST /update-progress/` - Update user progress

### Request/Response Examples

#### Real-time Analysis
```json
POST /real-time-analysis/
{
  "activityType": "pacing_curve",
  "duration": 30,
  "timestamp": 1234567890
}

Response:
{
  "success": true,
  "score": 75,
  "feedback": "Good pace consistency detected",
  "timestamp": 1234567890,
  "duration": 30,
  "activity_type": "pacing_curve"
}
```

#### Activity Analysis
```json
POST /analyze-activity/
FormData: file (audio/wav), activityType (string)

Response:
{
  "success": true,
  "activity_type": "pacing_curve",
  "final_score": 85.5,
  "max_score": 100,
  "metrics": {...},
  "duration": 120.5,
  "word_count": 150,
  "average_wpm": 125.3,
  "consistency_score": 78.2,
  "pause_ratio": 12.5,
  "feedback": "Excellent performance!",
  "recommendations": [...],
  "new_badges": [...],
  "timestamp": "2024-01-01T12:00:00"
}
```

## 🎮 Usage

### Starting an Activity
1. Navigate to Pace Management → Interactive Activities
2. Select an activity from the list
3. Click "Start" to begin recording
4. Receive real-time feedback during recording
5. Click "Stop" to complete and analyze

### Understanding Scores
- **90-100**: Excellent performance
- **70-89**: Good performance
- **50-69**: Fair performance
- **0-49**: Needs practice

### Earning Badges
- Complete activities with specific performance criteria
- Badges are automatically awarded based on metrics
- View earned badges in the progress panel

## 🧪 Testing

Run the test script to verify all endpoints:

```bash
cd modal/Phase-Management/OtherPaceFeatures/
python test_activities.py
```

The test script will:
- Verify all API endpoints are working
- Test real-time analysis functionality
- Validate activity analysis with sample audio
- Check user progress tracking

## 🔧 Setup

### Backend Setup
1. Ensure all dependencies are installed
2. Start the FastAPI server:
   ```bash
   cd modal/Phase-Management/OtherPaceFeatures/
   uvicorn app:app --reload --port 8000
   ```

### Frontend Setup
1. Install React dependencies
2. Start the development server:
   ```bash
   cd frontend/
   npm start
   ```

## 📊 Activity-Specific Scoring

Each activity has unique scoring criteria:

### Pacing Curve Challenge
- WPM standard deviation (lower = better)
- WPM consistency score
- Target: <20 WPM variation

### Rate Match Drill
- Proximity to target WPM (100-150)
- Consistency over time
- Target: 125 WPM ±10%

### Pause Timing Drill
- Optimal pause ratio (0.5-2s pauses)
- Precision of pause timing
- Target: 70% optimal pauses

### And more...

## 🎯 Future Enhancements

- Multiplayer challenges
- Leaderboards
- Advanced AI coaching
- Custom activity creation
- Integration with external speech analysis tools
- Mobile app support

## 🤝 Contributing

To add new activities:
1. Define activity in `ACTIVITY_TYPES` dictionary
2. Implement scoring logic in `calculate_activity_score()`
3. Add badge requirements in `check_badge_eligibility()`
4. Update frontend activity list
5. Add tests for new functionality

## 📝 License

This project is part of the SpeaKraft speech improvement platform.
