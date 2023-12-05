from __future__ import annotations

"""
The mechanics of rating changes.
"""

SCALE_FACTOR = 400

class RatingEntity:
    """
    Represents an entity that has an Elo-like numerical rating.
    
    Attributes:
        rating (float): A numerical rating that represents their skill level.
        K (int): The K-factor is a constant that determines how much a player's rating will change based on the game outcome. A higher K-factor allows ratings to change more rapidly.
        num_battle (int): The number of battles the entity has participated in.
    """
    
    def __init__(self, initial_rating: float, K: int):
        """
        Initializes a new RatingEntity object.
        
        Args:
            initial_rating (float): The initial rating of the entity.
            K (int): The K-factor for the entity.
        """    
        self.rating = initial_rating
        self.K = K
        self.num_battle = 0
        
    def expected_score(self, other: RatingEntity) -> float:
        """
        Computes the expected score (winrate) when facing the other rating entity.
        
        Args:
            other (RatingEntity): The other rating entity.
        
        Returns:
            float: The expected score (winrate) when facing the other rating entity.
        """
        return 1 / (1 + pow(10, (other.rating - self.rating) / SCALE_FACTOR))
    
    def rating_delta(self, expected_score: float, actual_score: float) -> float:
        """
        Computes how much the rating would change according to the given scores.
        
        Args:
            expected_score (float): The expected score (winrate).
            actual_score (float): The actual score (winrate).
        
        Returns:
            float: The rating change based on the given scores.
        """
        return self.K * (actual_score - expected_score)
    
    def update_rating(self, expected_score: float, actual_score: float):
        """
        Updates the rating according to the given scores.
        
        Args:
            expected_score (float): The expected score (winrate).
            actual_score (float): The actual score (winrate).
        """
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
    
