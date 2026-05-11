from typing import Dict, Any, Optional
import numpy as np

class EvaluationService:
    def evaluate_price(
        self, 
        symbol: str, 
        current_price: float, 
        analysis: Dict[str, Any],
        tolerance_percent: float = 0.5
    ) -> Dict[str, Any]:
        """
        Evaluate if the current price is near a level or in a specific zone.
        """
        supports = analysis.get('supports', [])
        resistances = analysis.get('resistances', [])
        
        all_levels = supports + resistances
        if not all_levels:
            return {"error": "No levels detected for this symbol"}

        # Find nearest level
        nearest_level = min(all_levels, key=lambda x: abs(x - current_price))
        distance = abs(current_price - nearest_level)
        distance_pct = (distance / current_price) * 100

        zone_type = "neutral"
        suggested_action = "wait"

        is_near = distance_pct <= tolerance_percent

        if is_near:
            if nearest_level in supports:
                zone_type = "support"
                suggested_action = "possible_buy"
            else:
                zone_type = "resistance"
                suggested_action = "possible_sell"
        else:
            # Check for potential breakout
            # Simple logic: if above all resistances or below all supports
            if resistances and current_price > max(resistances):
                zone_type = "breakout_high"
                suggested_action = "bullish_momentum"
            elif supports and current_price < min(supports):
                zone_type = "breakout_low"
                suggested_action = "bearish_momentum"

        return {
            "symbol": symbol,
            "current_price": round(current_price, 5),
            "zone_type": zone_type,
            "nearest_level": round(nearest_level, 5),
            "distance_percent": round(distance_pct, 4),
            "confidence": analysis.get('confidence', 0),
            "suggested_action": suggested_action
        }
