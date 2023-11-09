from elo_rating.rating_entity import RatingEntity
from enum import Enum

class PairwiseBattleScore(Enum):
    WINNER_IS_A = 1 
    TIE = 2
    WINNER_IS_B = 3
    
class PairwiseRatingEntity:
    """Represents an pairwise entity has Elo-like numerical rating."""
    def __init__(self, entity_a: RatingEntity, entity_b: RatingEntity) -> None:
        self.entity_a = entity_a
        self.entity_b = entity_b
    
    def battle(self, winner: PairwiseBattleScore):
        """Compute rating delta of pairwise and update rating."""
        # cache expected score of a and b before battle and updating rating.
        expected_score_a = self.entity_a.expected_score(self.entity_b)
        expected_score_b = self.entity_b.expected_score(self.entity_a)
        
        # get actual score of both sides by battle winner        
        if winner == PairwiseBattleScore.WINNER_IS_A:
            actual_score_a = 1
            actual_score_b = 0
        elif winner == PairwiseBattleScore.TIE:
            actual_score_a = 0.5
            actual_score_b = 0.5
        elif winner == PairwiseBattleScore.WINNER_IS_B:
            actual_score_a = 0
            actual_score_b = 1
        else:
            raise Exception("unexpected winner: ", winner)

        # update rating of two sides
        self.entity_a.update_rating(expected_score_a, actual_score_a)
        self.entity_b.update_rating(expected_score_b, actual_score_b)
        
        self.entity_a.num_battle+=1
        self.entity_b.num_battle+=1
        