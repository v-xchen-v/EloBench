# elo_ranking_system_simulation
A library of elo ranking system with simulation apps using python and gradio

## carings
1. stability
how to set K-factor and（initial score=1000）
2. timely
Can we save the battle result for some large scale(~70B) models to get the same ranking?
3. accurate
- enough observation
- more battles for high-level players, because more tie outcomes for larger model.
4. visualizations for investigat
5. evaluate metric for elo ratings.

## Feature

- Rating Calculation: The core feature, which involves updating the Elo ratings of players based on their game results. This includes calculating expected scores and adjusting ratings after each game.

- Initial Rating Assignment: Facilities for assigning initial ratings to new players entering the system.

- Match Predictions: Using current ratings to predict the outcomes of matches.

- API for Battle Results Input: An easy-to-use API for inputting battle results (win, loss, draw) and updating ratings accordingly.

- TODO: Support for Different K-Factors: The K-factor in Elo rating determines the sensitivity of the rating system. Libraries may allow different K-factors for different player levels (e.g., beginners vs. advanced players).

- TODO: Rating Scale Customization: Some libraries allow customization of the rating scale, including the base rating (often set at 1500) and the scale factor determining how much ratings can change per game.

- TODO: Rating Decay Over Time: Some libraries implement a feature where inactive players' ratings decay over time.

- TODO: History Tracking: Ability to track and visualize the history of players' ratings over time.

- TODO: Import/Export Ratings: Facilities to import and export player ratings, which is useful for maintaining persistent player databases.

- TODO: Performance Metrics: Calculation of performance metrics such as win/loss ratio, average opponent rating, highest rating achieved, etc.

- TODO: Documentation and Examples: Comprehensive documentation and examples to help users implement the Elo system in their applications.

- TODO: Multiplayer Support: While traditional Elo is designed for two-player games, some libraries extend the system to support multiplayer games.