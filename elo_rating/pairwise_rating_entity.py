from elo_rating.rating_entity import RatingEntity
from enum import Enum

class PairwiseBattleWinner(Enum):
    WINNER_IS_A = 1 
    TIE = 2
    WINNER_IS_B = 3
    
class PairwiseRatingEntity:
    """Represents an pairwise entity with Elo-like numerical rating.
    
        Attributes:
        entity_a (RatingEntity): The first entity in the pairwise rating.
        entity_b (RatingEntity): The second entity in the pairwise rating.
    """
    def __init__(self, entity_a: RatingEntity, entity_b: RatingEntity) -> None:
        self.entity_a = entity_a
        self.entity_b = entity_b
    
    def battle(self, winner: PairwiseBattleWinner):
        """Compute rating delta of pairwise and update rating.

        Args:
            winner (PairwiseBattleWinner): The winner of the pairwise battle.

        Raises:
            Exception: If an unexpected winner is provided.
        """
        # cache expected score of a and b before battle and updating rating.
        expected_score_a = self.entity_a.expected_score(self.entity_b)
        expected_score_b = self.entity_b.expected_score(self.entity_a)

        # get actual score of both sides by battle winner        
        if winner == PairwiseBattleWinner.WINNER_IS_A:
            actual_score_a = 1.0
            actual_score_b = 0.0
        elif winner == PairwiseBattleWinner.TIE:
            actual_score_a = 0.5
            actual_score_b = 0.5
        elif winner == PairwiseBattleWinner.WINNER_IS_B:
            actual_score_a = 0.0
            actual_score_b = 1.0
        else:
            raise Exception("unexpected winner: ", winner)

        # update rating of two sides
        self.entity_a.update_rating(expected_score_a, actual_score_a)
        self.entity_b.update_rating(expected_score_b, actual_score_b)
        
        self.entity_a.num_battle += 1
        self.entity_b.num_battle += 1
        