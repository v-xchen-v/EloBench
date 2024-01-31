from __future__ import annotations
from elo_rating.rating_entity import RatingEntity
from abc import ABC, abstractmethod
from enum import Enum

INITIAL_PLAYER_K = 4 #32
FINAL_PLAYER_K = 4 #16
INITIAL_LLMPLAYER_RATING = 1000

class K_ADJUSTMENT_STRATEGY(Enum):
    SOLID=1
    BASE_ON_RATING_LEVEL=2
    BASE_ON_NUM_OF_BATTLE=3
    BASE_ON_MATCH_IMPORTANCE=4 # TODO: the difficulties of question

class Player(RatingEntity, ABC):
    """Represents a player with Elo rating.
    
    This class is an abstract base class that defines the common behavior and attributes of a player.
    """
    def __init__(self, rating: float, K: int):
        super().__init__(rating, K)
        
    def update_K(self):
        """Define the strategy to evolve K over time."""
        # self.K = max(FINAL_PLAYER_K, self.K -1)
        # turn off K evolving temporally
        return self.K
    
class LLMPlayer(Player):
    """Represents a large language model player with Elo rating.
    
    This class inherits from the `Player` class and adds additional functionality specific to large language models.
    """
    def __init__(self, id: str, initial_rating: float= INITIAL_LLMPLAYER_RATING, K: int = INITIAL_PLAYER_K, evolve_K: K_ADJUSTMENT_STRATEGY=K_ADJUSTMENT_STRATEGY.SOLID):
        self.id = id
        super().__init__(initial_rating, K)
        self.evolve_K = evolve_K
            
    def __str__(self):
        return f'model_id:{self.id} rating:{self.rating}'
    
    def __repr__(self) -> str:
        return f'model_id:{self.id} rating:{self.rating}'
    
    def update_K(self):
        """Update the K factor based on the specified strategy."""
        if self.evolve_K == K_ADJUSTMENT_STRATEGY.SOLID:
            pass
        elif self.evolve_K == K_ADJUSTMENT_STRATEGY.BASE_ON_RATING_LEVEL:
            self.adjust_K_by_rating_level()
        elif self.evolve_K == K_ADJUSTMENT_STRATEGY.BASE_ON_NUM_OF_BATTLE:
            self.adjust_K_by_num_of_battle()
        else:
            raise Exception("Not implemented")
        
    def adjust_K_by_rating_level(self):
        """Adjust the K factor based on the rating level of the player."""
        if self.rating > 1100:
            self.K = INITIAL_PLAYER_K//2
        else:
            self.K = INITIAL_PLAYER_K
    
    def adjust_K_by_num_of_battle(self):
        """Adjust the K factor based on the number of battles the player has participated in."""
        if self.num_battle > 10:
            self.K = INITIAL_PLAYER_K//2
        else:
            self.K = INITIAL_PLAYER_K
            
    def update_rating_by_winner(self, another_player: Player, as_winner: bool):
        """Update the rating of the player based on the result of a battle."""
        # cache expected score of a and b before battle and updating rating.
        expected_score_a = self.expected_score(another_player)
        expected_score_b = another_player.expected_score(self)
        
        # get actual score of both sides by battle winner  
        if as_winner:
            actual_score_a = 1.0
            actual_score_b = 0.0
        else:
            actual_score_a = 0.0
            actual_score_b = 1.0      

        # update rating of two sides
        self.update_rating(expected_score_a, actual_score_a)
        another_player.update_rating(expected_score_b, actual_score_b)
        
        self.num_battle += 1
        another_player.num_battle += 1
        
        # self.update_K()
        # another_player.update_K()