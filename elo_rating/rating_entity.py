from __future__ import annotations

"""
The mechanics of rating changes.
"""

SCALE_FACTOR = 400

class RatingEntity:
    """Represents an entity that has an Elo-like numerical rating."""
    
    def __init__(self, rating: float, K: int):
        """
        Args:
            rating (float): A numerical rating that represents their skill level.
            K (int): The K-factor is a constant that determines how much a player's rating will change based on the game outcome. A higher K-factor allows ratings to change more rapidly.
        """        
        self.rating = rating
        self.K = K
        self.num_battle = 0
        
    def expected_score(self, other: RatingEntity) -> float:
        """Compute the expected score(winrate) when facing the other rating entity."""
        return 1 / (1 + pow(10, (other.rating - self.rating) / SCALE_FACTOR))
    
    def rating_delta(self, expected_score: float, actual_score: float) -> float:
        """Compute how much the rating would change according to the give scores."""
        return self.K * (actual_score - expected_score)
    
    def update_rating(self, expected_score: float, actual_score: float) -> float:
        """Update the rating according to the give scores."""
        # print("rating delta: ", self.rating_delta(expected_score, actual_score))
        self.rating += self.rating_delta(expected_score, actual_score)
        
    
    def __str__(self):
        return f"RatingEntity({self.rating}, {self.K})"
    
    
if __name__ == "__main__":
    # Wikipedia example
    A = RatingEntity(1613, 32)
    print(A)
    print(A.expected_score(RatingEntity(1609, 32))) # Should give approximately .51
    print(A.rating_delta(2.88, 2.5)) # Should give approximately -12.
    
