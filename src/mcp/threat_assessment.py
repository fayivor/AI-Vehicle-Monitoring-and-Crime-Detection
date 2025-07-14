"""Threat level assessment system for dynamic context adjustment."""

import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from ..models.vehicle import Vehicle
from ..models.incident import Incident, SeverityLevel, IncidentType
from ..models.user import UserSession
from ..utils.logging import get_logger
from ..utils.config import get_settings

logger = get_logger(__name__)
settings = get_settings()


class ThreatLevel(Enum):
    """Threat level enumeration."""
    NORMAL = 1      # Normal operations
    ELEVATED = 2    # Elevated awareness
    HEIGHTENED = 3  # Heightened alert
    HIGH = 4        # High threat
    CRITICAL = 5    # Critical threat


@dataclass
class ThreatIndicator:
    """Individual threat indicator."""
    indicator_type: str
    value: float
    weight: float
    description: str
    timestamp: datetime
    source: str


@dataclass
class ThreatAssessment:
    """Comprehensive threat assessment."""
    overall_threat_level: ThreatLevel
    threat_score: float
    indicators: List[ThreatIndicator]
    assessment_time: datetime
    location: Optional[str] = None
    confidence: float = 0.0
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []


class ThreatAssessor:
    """
    Threat level assessment system that analyzes current conditions
    and adjusts context prioritization accordingly.
    """
    
    def __init__(self):
        self.indicator_weights = self._initialize_indicator_weights()
        self.threat_thresholds = self._initialize_threat_thresholds()
        self.recent_assessments: Dict[str, ThreatAssessment] = {}
    
    def _initialize_indicator_weights(self) -> Dict[str, float]:
        """Initialize weights for different threat indicators."""
        return {
            'high_risk_vehicles': 0.25,
            'recent_incidents': 0.30,
            'incident_severity': 0.20,
            'incident_frequency': 0.15,
            'suspicious_patterns': 0.10,
            'border_activity': 0.20,
            'time_of_day': 0.05,
            'location_risk': 0.15,
            'vehicle_concentration': 0.10,
            'alert_flags': 0.15
        }
    
    def _initialize_threat_thresholds(self) -> Dict[ThreatLevel, Tuple[float, float]]:
        """Initialize threat level thresholds (min, max)."""
        return {
            ThreatLevel.NORMAL: (0.0, 0.3),
            ThreatLevel.ELEVATED: (0.3, 0.5),
            ThreatLevel.HEIGHTENED: (0.5, 0.7),
            ThreatLevel.HIGH: (0.7, 0.85),
            ThreatLevel.CRITICAL: (0.85, 1.0)
        }
    
    def assess_threat_level(
        self,
        user_session: UserSession,
        recent_vehicles: List[Vehicle] = None,
        recent_incidents: List[Incident] = None,
        location: Optional[str] = None
    ) -> ThreatAssessment:
        """
        Assess current threat level based on multiple indicators.
        
        Args:
            user_session: Current user session
            recent_vehicles: Recent vehicle data
            recent_incidents: Recent incident data
            location: Location context
            
        Returns:
            Comprehensive threat assessment
        """
        logger.info(
            "Starting threat assessment",
            user_id=user_session.user.user_id,
            location=location
        )
        
        indicators = []
        
        # Analyze vehicle-based indicators
        if recent_vehicles:
            vehicle_indicators = self._assess_vehicle_indicators(recent_vehicles)
            indicators.extend(vehicle_indicators)
        
        # Analyze incident-based indicators
        if recent_incidents:
            incident_indicators = self._assess_incident_indicators(recent_incidents)
            indicators.extend(incident_indicators)
        
        # Analyze temporal indicators
        temporal_indicators = self._assess_temporal_indicators()
        indicators.extend(temporal_indicators)
        
        # Analyze location-based indicators
        if location:
            location_indicators = self._assess_location_indicators(location)
            indicators.extend(location_indicators)
        
        # Calculate overall threat score
        threat_score = self._calculate_threat_score(indicators)
        
        # Determine threat level
        threat_level = self._determine_threat_level(threat_score)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(threat_level, indicators)
        
        # Calculate confidence
        confidence = self._calculate_confidence(indicators)
        
        assessment = ThreatAssessment(
            overall_threat_level=threat_level,
            threat_score=threat_score,
            indicators=indicators,
            assessment_time=datetime.utcnow(),
            location=location,
            confidence=confidence,
            recommendations=recommendations
        )
        
        # Cache assessment
        cache_key = f"{user_session.session_id}_{location or 'global'}"
        self.recent_assessments[cache_key] = assessment
        
        logger.info(
            "Threat assessment completed",
            threat_level=threat_level.name,
            threat_score=threat_score,
            confidence=confidence,
            indicators_count=len(indicators)
        )
        
        return assessment
    
    def _assess_vehicle_indicators(self, vehicles: List[Vehicle]) -> List[ThreatIndicator]:
        """Assess threat indicators from vehicle data."""
        indicators = []
        
        if not vehicles:
            return indicators
        
        # High-risk vehicle concentration
        high_risk_count = sum(1 for v in vehicles if v.risk_score > 0.7)
        high_risk_ratio = high_risk_count / len(vehicles)
        
        if high_risk_ratio > 0.3:
            indicators.append(ThreatIndicator(
                indicator_type='high_risk_vehicles',
                value=high_risk_ratio,
                weight=self.indicator_weights['high_risk_vehicles'],
                description=f'{high_risk_count} high-risk vehicles detected ({high_risk_ratio:.1%})',
                timestamp=datetime.utcnow(),
                source='vehicle_analysis'
            ))
        
        # Alert flags concentration
        total_alerts = sum(len(v.alert_flags) for v in vehicles)
        if total_alerts > 0:
            alert_density = total_alerts / len(vehicles)
            if alert_density > 1.0:  # More than 1 alert per vehicle on average
                indicators.append(ThreatIndicator(
                    indicator_type='alert_flags',
                    value=min(alert_density / 3.0, 1.0),  # Normalize to 0-1
                    weight=self.indicator_weights['alert_flags'],
                    description=f'High alert flag density: {alert_density:.1f} alerts per vehicle',
                    timestamp=datetime.utcnow(),
                    source='vehicle_analysis'
                ))
        
        # Vehicle concentration in area
        if len(vehicles) > 50:  # Arbitrary threshold for high concentration
            concentration_score = min(len(vehicles) / 100.0, 1.0)
            indicators.append(ThreatIndicator(
                indicator_type='vehicle_concentration',
                value=concentration_score,
                weight=self.indicator_weights['vehicle_concentration'],
                description=f'High vehicle concentration: {len(vehicles)} vehicles',
                timestamp=datetime.utcnow(),
                source='vehicle_analysis'
            ))
        
        return indicators
    
    def _assess_incident_indicators(self, incidents: List[Incident]) -> List[ThreatIndicator]:
        """Assess threat indicators from incident data."""
        indicators = []
        
        if not incidents:
            return indicators
        
        # Recent incident frequency
        recent_cutoff = datetime.utcnow() - timedelta(hours=24)
        recent_incidents = [
            i for i in incidents 
            if i.timestamp >= recent_cutoff
        ]
        
        if recent_incidents:
            frequency_score = min(len(recent_incidents) / 10.0, 1.0)  # Normalize to 10 incidents/day
            indicators.append(ThreatIndicator(
                indicator_type='recent_incidents',
                value=frequency_score,
                weight=self.indicator_weights['recent_incidents'],
                description=f'{len(recent_incidents)} incidents in last 24 hours',
                timestamp=datetime.utcnow(),
                source='incident_analysis'
            ))
        
        # Incident severity analysis
        high_severity_count = sum(
            1 for i in incidents 
            if i.severity_level >= SeverityLevel.MEDIUM_HIGH
        )
        
        if high_severity_count > 0:
            severity_ratio = high_severity_count / len(incidents)
            indicators.append(ThreatIndicator(
                indicator_type='incident_severity',
                value=severity_ratio,
                weight=self.indicator_weights['incident_severity'],
                description=f'{high_severity_count} high-severity incidents ({severity_ratio:.1%})',
                timestamp=datetime.utcnow(),
                source='incident_analysis'
            ))
        
        # Suspicious pattern detection
        smuggling_incidents = [
            i for i in recent_incidents 
            if i.incident_type in [IncidentType.SMUGGLING, IncidentType.SUSPICIOUS_ACTIVITY]
        ]
        
        if len(smuggling_incidents) >= 3:  # Pattern threshold
            pattern_score = min(len(smuggling_incidents) / 5.0, 1.0)
            indicators.append(ThreatIndicator(
                indicator_type='suspicious_patterns',
                value=pattern_score,
                weight=self.indicator_weights['suspicious_patterns'],
                description=f'Suspicious activity pattern: {len(smuggling_incidents)} related incidents',
                timestamp=datetime.utcnow(),
                source='pattern_analysis'
            ))
        
        # Border crossing activity
        border_incidents = [
            i for i in recent_incidents 
            if i.incident_type == IncidentType.BORDER_CROSSING
        ]
        
        if border_incidents:
            border_score = min(len(border_incidents) / 5.0, 1.0)
            indicators.append(ThreatIndicator(
                indicator_type='border_activity',
                value=border_score,
                weight=self.indicator_weights['border_activity'],
                description=f'Increased border activity: {len(border_incidents)} incidents',
                timestamp=datetime.utcnow(),
                source='border_analysis'
            ))
        
        return indicators
    
    def _assess_temporal_indicators(self) -> List[ThreatIndicator]:
        """Assess time-based threat indicators."""
        indicators = []
        
        current_hour = datetime.utcnow().hour
        
        # High-risk time periods (late night/early morning)
        if current_hour >= 22 or current_hour <= 5:
            time_risk_score = 0.3  # Moderate increase for high-risk hours
            indicators.append(ThreatIndicator(
                indicator_type='time_of_day',
                value=time_risk_score,
                weight=self.indicator_weights['time_of_day'],
                description=f'High-risk time period: {current_hour:02d}:00',
                timestamp=datetime.utcnow(),
                source='temporal_analysis'
            ))
        
        return indicators
    
    def _assess_location_indicators(self, location: str) -> List[ThreatIndicator]:
        """Assess location-based threat indicators."""
        indicators = []
        
        # High-risk locations (simplified - would use actual risk database)
        high_risk_locations = [
            'tema port', 'border crossing', 'airport', 'kumasi market'
        ]
        
        location_lower = location.lower()
        for risk_location in high_risk_locations:
            if risk_location in location_lower:
                location_risk_score = 0.4
                indicators.append(ThreatIndicator(
                    indicator_type='location_risk',
                    value=location_risk_score,
                    weight=self.indicator_weights['location_risk'],
                    description=f'High-risk location: {location}',
                    timestamp=datetime.utcnow(),
                    source='location_analysis'
                ))
                break
        
        return indicators
    
    def _calculate_threat_score(self, indicators: List[ThreatIndicator]) -> float:
        """Calculate overall threat score from indicators."""
        if not indicators:
            return 0.0
        
        weighted_sum = sum(
            indicator.value * indicator.weight 
            for indicator in indicators
        )
        
        total_weight = sum(indicator.weight for indicator in indicators)
        
        if total_weight == 0:
            return 0.0
        
        # Normalize by total weight and apply scaling
        base_score = weighted_sum / total_weight
        
        # Apply non-linear scaling to emphasize higher threat levels
        scaled_score = base_score ** 0.8
        
        return min(scaled_score, 1.0)
    
    def _determine_threat_level(self, threat_score: float) -> ThreatLevel:
        """Determine threat level from threat score."""
        for level, (min_score, max_score) in self.threat_thresholds.items():
            if min_score <= threat_score <= max_score:
                return level
        
        # Default to highest level if score exceeds all thresholds
        return ThreatLevel.CRITICAL
    
    def _calculate_confidence(self, indicators: List[ThreatIndicator]) -> float:
        """Calculate confidence in the threat assessment."""
        if not indicators:
            return 0.0
        
        # Confidence based on number and diversity of indicators
        base_confidence = min(len(indicators) / 5.0, 1.0)
        
        # Boost confidence for diverse indicator types
        indicator_types = set(indicator.indicator_type for indicator in indicators)
        diversity_bonus = min(len(indicator_types) / 8.0, 0.3)
        
        return min(base_confidence + diversity_bonus, 1.0)
    
    def _generate_recommendations(
        self, 
        threat_level: ThreatLevel, 
        indicators: List[ThreatIndicator]
    ) -> List[str]:
        """Generate recommendations based on threat level and indicators."""
        recommendations = []
        
        if threat_level == ThreatLevel.CRITICAL:
            recommendations.extend([
                "Activate emergency response protocols",
                "Increase patrol frequency in affected areas",
                "Coordinate with border security",
                "Alert senior command immediately"
            ])
        
        elif threat_level == ThreatLevel.HIGH:
            recommendations.extend([
                "Increase surveillance in high-risk areas",
                "Deploy additional units to key locations",
                "Monitor vehicle movements closely",
                "Prepare rapid response teams"
            ])
        
        elif threat_level == ThreatLevel.HEIGHTENED:
            recommendations.extend([
                "Enhance monitoring of flagged vehicles",
                "Increase checkpoint frequency",
                "Review recent incident patterns",
                "Brief officers on current threats"
            ])
        
        elif threat_level == ThreatLevel.ELEVATED:
            recommendations.extend([
                "Maintain heightened awareness",
                "Monitor developing situations",
                "Review security protocols"
            ])
        
        # Add specific recommendations based on indicators
        for indicator in indicators:
            if indicator.indicator_type == 'high_risk_vehicles' and indicator.value > 0.5:
                recommendations.append("Focus on high-risk vehicle monitoring")
            
            elif indicator.indicator_type == 'border_activity' and indicator.value > 0.3:
                recommendations.append("Coordinate with border control units")
            
            elif indicator.indicator_type == 'suspicious_patterns' and indicator.value > 0.4:
                recommendations.append("Investigate potential criminal networks")
        
        return list(set(recommendations))  # Remove duplicates
    
    def get_cached_assessment(
        self, 
        session_id: str, 
        location: Optional[str] = None
    ) -> Optional[ThreatAssessment]:
        """Get cached threat assessment if available and recent."""
        cache_key = f"{session_id}_{location or 'global'}"
        assessment = self.recent_assessments.get(cache_key)
        
        if assessment:
            # Check if assessment is still fresh (within 15 minutes)
            age = (datetime.utcnow() - assessment.assessment_time).total_seconds()
            if age <= 900:  # 15 minutes
                return assessment
            else:
                # Remove stale assessment
                del self.recent_assessments[cache_key]
        
        return None
    
    def cleanup_old_assessments(self) -> None:
        """Clean up old cached assessments."""
        current_time = datetime.utcnow()
        expired_keys = []
        
        for key, assessment in self.recent_assessments.items():
            age = (current_time - assessment.assessment_time).total_seconds()
            if age > 3600:  # 1 hour
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.recent_assessments[key]
        
        if expired_keys:
            logger.info("Cleaned up old threat assessments", count=len(expired_keys))
