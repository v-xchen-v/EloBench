import sys, os
sys.path.append(os.path.dirname(os.path.dirname((__file__))))

from battle_pipe import BattlePipeline
from datamodel.question_and_answers_collection import QuestionAndAnswersCollection, LLMAnswer
from datamodel import QuestionCollection, PairwiseBattleArrangement, ArrangementStrategy
import pandas as pd
from datamodel import MODEL_A_COLUMN_NAME, MODEL_B_COLUMN_NAME
from pathlib import Path

question_file = Path('data')/'arena_data'/'chatbot_arena_conversations'/'questions.csv'
q_and_as_file = Path('data')/'arena_data'/'chatbot_arena_conversations'/'q_and_as.csv'
battle_arrangement_file = Path('data')/'arena_data'/'chatbot_arena_conversations'/'battle_arrangement.csv'
battled_pairs_file = Path('data')/'arena_data/chatbot_arena_conversations'/'battled_pairs.csv'

class BattleArenaWithGPT4AsJudger(BattlePipeline):
    def gen_model_answers(self) -> None:
        battle_arrangement = pd.read_csv(battle_arrangement_file)
        question_and_answers = pd.read_csv(q_and_as_file)
        
        def pick_answer(question: str, model_a_or_b: str):
            model_name = battle_arrangement[battle_arrangement['question'] == question].iloc[0][model_a_or_b]

            answer = question_and_answers[question_and_answers['question']==question].iloc[0][model_name]
            return answer

        for rnd in self.battle_arrangements.battles_in_order:
            q = rnd.question
            m_a = rnd.model_a
            m_b = rnd.model_b
            
            # generate ans
            ans_a = pick_answer(q, MODEL_A_COLUMN_NAME)
            ans_b = pick_answer(q, MODEL_B_COLUMN_NAME)

            self.question_and_answers_collection.add_answer(q, LLMAnswer(m_a, ans_a))
            self.question_and_answers_collection.add_answer(q, LLMAnswer(m_b, ans_b))
            
if __name__ == '__main__':          
    start=0
    end=1010 
    bp = BattleArenaWithGPT4AsJudger(tempcache_dir=Path('results')/'chatbot_arena_conversations_0_1000',start=start, end=end, no_cache=False)
    
    # qs = ["Give me a question about animal.", "Is Landon rainy?", 'Is Austrilia always sunny?']
    # qs = QuestionCollection.read_csv(question_file).questions
    qs = PairwiseBattleArrangement.read_csv(battle_arrangement_file).questions
    bp.register_questions(qs)
    
    # models = ['huggyllama/llama-7b', 'meta-llama/Llama-2-7b-hf', 'mosaicml/mpt-7b']
    models = PairwiseBattleArrangement.read_csv(battle_arrangement_file).models
    bp.register_models(models)
    
    bp.arrange_battles(arrange_strategy=ArrangementStrategy.Reload_Existing_Arrangement, battle_arrangement_file=Path('results')/'chatbot_arena_conversations_0_1000'/'battle_arrangement.csv')
    
    bp.gen_model_answers()
    bp.battle()
    bp.gen_elo()
    pass